"""
用户长期记忆存储管理器

为 LangGraph agent 提供用户画像持久化存储。
单例模式管理 AsyncConnectionPool + AsyncPostgresStore，
并提供 user_profiles 表的 CRUD 操作。
"""
import asyncio
import json
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres import AsyncPostgresStore

from app.config import settings
from app.utils.logger import app_logger

# =============================================================================
# DDL
# =============================================================================

CREATE_USER_PROFILES_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(64) PRIMARY KEY,
    preferred_transport VARCHAR(20),
    budget_level VARCHAR(20),
    travel_styles JSONB DEFAULT '[]',
    favorite_destinations JSONB DEFAULT '[]',
    dietary_preferences JSONB DEFAULT '[]',
    total_trips INTEGER DEFAULT 0,
    last_destination VARCHAR(100),
    last_travel_date DATE,
    extensions JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

UPSERT_PROFILE_SQL = """
INSERT INTO user_profiles (
    user_id, preferred_transport, budget_level,
    travel_styles, favorite_destinations, dietary_preferences,
    total_trips, last_destination, last_travel_date,
    extensions, created_at, updated_at
)
VALUES (
    %s, %s, %s,
    %s::jsonb, %s::jsonb, %s::jsonb,
    %s, %s, %s,
    %s::jsonb, NOW(), NOW()
)
ON CONFLICT (user_id) DO UPDATE SET
    preferred_transport = COALESCE(EXCLUDED.preferred_transport, user_profiles.preferred_transport),
    budget_level = COALESCE(EXCLUDED.budget_level, user_profiles.budget_level),
    travel_styles = EXCLUDED.travel_styles,
    favorite_destinations = EXCLUDED.favorite_destinations,
    dietary_preferences = EXCLUDED.dietary_preferences,
    total_trips = COALESCE(EXCLUDED.total_trips, user_profiles.total_trips),
    last_destination = COALESCE(EXCLUDED.last_destination, user_profiles.last_destination),
    last_travel_date = COALESCE(EXCLUDED.last_travel_date, user_profiles.last_travel_date),
    extensions = EXCLUDED.extensions,
    updated_at = NOW();
"""

# =============================================================================
# MemoryStoreManager（单例模式）
# =============================================================================


class MemoryStoreManager:
    """用户长期记忆存储管理器（单例模式）"""

    _instance: Optional["MemoryStoreManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None
        self.store: Optional[AsyncPostgresStore] = None

    # -------------------------------------------------------------------------
    # 单例生命周期
    # -------------------------------------------------------------------------

    @classmethod
    async def get_instance(cls) -> "MemoryStoreManager":
        """获取单例实例（异步安全，双重检查锁定）"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    await instance.initialize()
                    cls._instance = instance
        return cls._instance

    async def initialize(self):
        """初始化连接池、创建表、创建 AsyncPostgresStore"""
        if self.store is not None:
            app_logger.warning("MemoryStoreManager 已初始化，跳过")
            return

        try:
            app_logger.info("初始化 PostgreSQL MemoryStore...")

            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_min_conn,
                max_size=settings.db_max_conn,
                timeout=settings.db_timeout,
            )

            await self.pool.open()

            # 创建 user_profiles 表
            try:
                async with self.pool.connection() as conn:
                    await conn.execute(CREATE_USER_PROFILES_SQL)

                # 创建 LangGraph AsyncPostgresStore
                self.store = AsyncPostgresStore(conn=self.pool)
                await self.store.setup()

                app_logger.info("MemoryStoreManager 初始化完成")
            except Exception:
                await self.pool.close()
                self.pool = None
                raise
        except Exception as e:
            app_logger.error(f"MemoryStoreManager 初始化失败: {e}")
            raise

    async def close(self):
        """关闭连接池，重置单例"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.store = None
            MemoryStoreManager._instance = None
            app_logger.info("MemoryStoreManager 连接池已关闭")

    def get_store(self) -> AsyncPostgresStore:
        """获取 AsyncPostgresStore 实例"""
        if self.store is None:
            raise RuntimeError("MemoryStoreManager 未初始化，请先调用 initialize()")
        return self.store

    # -------------------------------------------------------------------------
    # CRUD 操作（降级不崩溃，失败时记录 warning 并返回安全默认值）
    # -------------------------------------------------------------------------

    async def get_profile(self, user_id: str) -> Optional[dict]:
        """获取用户画像，不存在时返回 None"""
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT * FROM user_profiles WHERE user_id = %s",
                        (user_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        return None
                    columns = [desc[0] for desc in cur.description]
                    profile = dict(zip(columns, row))
                    # 转换日期等不可 JSON 序列化的字段
                    if profile.get("last_travel_date"):
                        profile["last_travel_date"] = str(profile["last_travel_date"])
                    if profile.get("created_at"):
                        profile["created_at"] = str(profile["created_at"])
                    if profile.get("updated_at"):
                        profile["updated_at"] = str(profile["updated_at"])
                    return profile
        except Exception as e:
            app_logger.warning(f"get_profile 失败 (user_id={user_id}): {e}")
            return None

    async def upsert_profile(self, user_id: str, fields: dict) -> Optional[dict]:
        """
        创建或更新用户画像。

        合并规则：
        - 数组字段（travel_styles, favorite_destinations, dietary_preferences）：
          新旧合并 → 去重（保序）→ 最多 10 项
        - 标量字段（preferred_transport, budget_level）：
          新值覆盖旧值；SQL 层 COALESCE 保证 NULL 不覆写已有数据
        - 统计字段（total_trips）：累加；last_destination / last_travel_date 有则更新
        - extensions：新旧 dict 浅合并
        """
        try:
            # 读取旧画像
            old = await self.get_profile(user_id) or {}

            # ---- 数组字段：合并 + 去重 + 截断 ----
            array_fields = ["travel_styles", "favorite_destinations", "dietary_preferences"]
            for field in array_fields:
                old_vals = old.get(field, []) or []
                new_vals = fields.get(field) or []
                if new_vals:
                    merged = list(dict.fromkeys(old_vals + new_vals))[:10]
                else:
                    merged = old_vals
                fields[field] = merged

            # ---- extensions：dict 浅合并 ----
            old_ext = old.get("extensions", {}) or {}
            new_ext = fields.get("extensions") or {}
            if new_ext:
                fields["extensions"] = {**old_ext, **new_ext}
            else:
                fields["extensions"] = old_ext

            # ---- 统计字段：total_trips 累加 ----
            if "total_trips" in fields:
                old_trips = old.get("total_trips", 0) or 0
                fields["total_trips"] = old_trips + int(fields["total_trips"])

            # ---- 执行 UPSERT ----
            async with self.pool.connection() as conn:
                await conn.execute(
                    UPSERT_PROFILE_SQL,
                    [
                        user_id,
                        fields.get("preferred_transport"),
                        fields.get("budget_level"),
                        json.dumps(fields.get("travel_styles", []), ensure_ascii=False),
                        json.dumps(fields.get("favorite_destinations", []), ensure_ascii=False),
                        json.dumps(fields.get("dietary_preferences", []), ensure_ascii=False),
                        fields.get("total_trips"),
                        fields.get("last_destination"),
                        fields.get("last_travel_date"),
                        json.dumps(fields.get("extensions", {}), ensure_ascii=False),
                    ],
                )

            # ---- 读回完整画像 ----
            return await self.get_profile(user_id)
        except Exception as e:
            app_logger.warning(f"upsert_profile 失败 (user_id={user_id}): {e}")
            return None

    async def delete_profile(self, user_id: str) -> bool:
        """删除用户画像，返回是否实际删除了行"""
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM user_profiles WHERE user_id = %s",
                        (user_id,),
                    )
                    deleted = cur.rowcount > 0
                    return deleted
        except Exception as e:
            app_logger.warning(f"delete_profile 失败 (user_id={user_id}): {e}")
            return False

    async def list_user_ids(self) -> list[str]:
        """返回所有已存储的用户 ID 列表"""
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT user_id FROM user_profiles")
                    rows = await cur.fetchall()
                    return [row[0] for row in rows]
        except Exception as e:
            app_logger.warning(f"list_user_ids 失败: {e}")
            return []


# =============================================================================
# 便捷函数
# =============================================================================


async def get_memory_store_manager() -> MemoryStoreManager:
    """获取全局 MemoryStoreManager 实例"""
    return await MemoryStoreManager.get_instance()

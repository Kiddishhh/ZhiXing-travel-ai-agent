import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.config import settings
from app.utils.logger import app_logger


class CheckpointerManager:
    """Checkpointer 管理器（单例模式）"""
    _instance: Optional['CheckpointerManager'] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    @classmethod
    async def get_instance(cls) -> 'CheckpointerManager':
        """获取单例实例（并发安全）"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """初始化连接池 + Checkpointer"""
        if self.checkpointer is not None:
            app_logger.warning("⚠️ Checkpointer 已初始化，跳过")
            return

        try:
            app_logger.info("初始化 PostgreSQL Checkpointer...")
            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_min_conn,
                max_size=settings.db_max_conn,
                timeout=settings.db_timeout,
                open=False,
            )
            await self.pool.open()
            self.checkpointer = AsyncPostgresSaver(self.pool)
            app_logger.info("✅ Checkpointer 初始化完成")
        except Exception as e:
            app_logger.error(f"❌ Checkpointer 初始化失败: {e}")
            raise

    async def close(self):
        """关闭数据库连接池，释放资源"""
        if self.pool:
            await self.pool.close()
            app_logger.info("Checkpointer 连接池已关闭")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """底层安全获取实例（同步，仅内存操作）"""
        if self.checkpointer is None:
            raise RuntimeError("Checkpointer 未初始化，请先调用 initialize()")
        return self.checkpointer


# ===================== 对外便捷顶层函数 =====================
# 业务层、服务层统一唯一入口，不是冗余！封装底层所有细节
async def get_checkpointer() -> AsyncPostgresSaver:
    """
    全局统一获取持久化器
    全项目所有地方统一用这一个函数
    """
    manager = await CheckpointerManager.get_instance()
    return manager.get_checkpointer()


# ===================== FastAPI 生命周期管理器 =====================
@asynccontextmanager
async def checkpointer_lifespan():
    """
    全局资源生命周期托管
    服务启动自动初始化，服务关闭自动释放
    """
    manager = await CheckpointerManager.get_instance()
    try:
        yield manager.get_checkpointer()
    finally:
        await manager.close()
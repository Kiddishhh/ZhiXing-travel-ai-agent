"""
业务表连接池管理器

管理 users / conversations / messages 三张业务表的连接池。
单例模式，与 CheckpointerManager 一致。
"""
import asyncio
from typing import Optional
from uuid import UUID

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def dict_row_str(cursor):
    """dict_row 变体 —— 自动将 UUID 转为字符串，兼容 Pydantic 校验"""
    _make_row = dict_row(cursor)

    def make_row_str(values):
        row = _make_row(values)
        if row is None:
            return None
        return {k: str(v) if isinstance(v, UUID) else v for k, v in row.items()}

    return make_row_str

from app.config import settings
from app.utils.logger import app_logger


CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    avatar_url VARCHAR(512),
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    preferences JSONB DEFAULT '{}',
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_CONVERSATIONS_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(256) DEFAULT '新对话',
    current_model VARCHAR(64),
    system_prompt TEXT,
    summary TEXT,
    total_tokens INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_MESSAGES_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    content_type VARCHAR(20) DEFAULT 'text',
    token_count INTEGER DEFAULT 0,
    feedback INTEGER DEFAULT 0,
    is_error BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);",
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);",
    "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(conversation_id, created_at);",
]


class DatabaseManager:
    """业务表连接池管理器（单例）"""

    _instance: Optional["DatabaseManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None

    @classmethod
    async def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    await instance.initialize()
                    cls._instance = instance
        return cls._instance

    async def initialize(self):
        if self.pool is not None:
            return
        try:
            app_logger.info("初始化业务表连接池...")
            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_min_conn,
                max_size=settings.db_max_conn,
                timeout=settings.db_timeout,
                kwargs={"row_factory": dict_row_str},
            )
            await self.pool.open()

            # 建表
            async with self.pool.connection() as conn:
                await conn.execute(CREATE_USERS_SQL)
                await conn.execute(CREATE_CONVERSATIONS_SQL)
                await conn.execute(CREATE_MESSAGES_SQL)
                for idx_sql in CREATE_INDEXES_SQL:
                    await conn.execute(idx_sql)

            app_logger.info("业务表初始化完成")
        except Exception as e:
            app_logger.error(f"业务表初始化失败: {e}")
            if self.pool:
                await self.pool.close()
                self.pool = None
            raise

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
            DatabaseManager._instance = None
            app_logger.info("业务表连接池已关闭")

    def get_pool(self) -> AsyncConnectionPool:
        if self.pool is None:
            raise RuntimeError("DatabaseManager 未初始化")
        return self.pool


async def get_db_pool() -> AsyncConnectionPool:
    """获取业务表连接池（用于 FastAPI 依赖注入）"""
    manager = await DatabaseManager.get_instance()
    return manager.get_pool()

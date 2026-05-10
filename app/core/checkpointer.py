"""
PostgreSQL Checkpointer 管理器

为 LangGraph agent 提供短期会话记忆（对话状态持久化）。
单例模式管理 AsyncConnectionPool + AsyncPostgresSaver。
"""
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.utils.logger import app_logger


class CheckpointerManager:
    """Checkpointer 管理器（单例模式）"""

    _instance: Optional["CheckpointerManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    @classmethod
    async def get_instance(cls) -> "CheckpointerManager":
        """获取单例实例（异步安全）"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    await instance.initialize()
                    cls._instance = instance
        return cls._instance

    async def initialize(self):
        """初始化连接池和 Checkpointer"""
        if self.checkpointer is not None:
            app_logger.warning("Checkpointer 已初始化，跳过")
            return

        try:
            app_logger.info("初始化 PostgreSQL Checkpointer...")

            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_min_conn,
                max_size=settings.db_max_conn,
                timeout=settings.db_timeout,
            )

            await self.pool.open()

            try:
                self.checkpointer = AsyncPostgresSaver(self.pool)
                await self.checkpointer.setup()

                app_logger.info("Checkpointer 初始化完成")
            except Exception:
                await self.pool.close()
                self.pool = None
                raise
        except Exception as e:
            app_logger.error(f"Checkpointer 初始化失败: {e}")
            raise

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.checkpointer = None
            app_logger.info("Checkpointer 连接池已关闭")

    def get_checkpointer(self) -> AsyncPostgresSaver:
        """获取 Checkpointer 实例"""
        if self.checkpointer is None:
            raise RuntimeError("Checkpointer 未初始化，请先调用 initialize()")
        return self.checkpointer


async def get_checkpointer() -> AsyncPostgresSaver:
    """获取全局 Checkpointer 实例"""
    manager = await CheckpointerManager.get_instance()
    return manager.get_checkpointer()


@asynccontextmanager
async def checkpointer_lifespan():
    """Checkpointer 生命周期管理器（用于 FastAPI lifespan）"""
    manager = await CheckpointerManager.get_instance()
    try:
        yield manager.get_checkpointer()
    finally:
        await manager.close()

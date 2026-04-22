"""
数据库初始化脚本
"""
import asyncio
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from app.config import settings
from app.utils.logger import app_logger


async def init_database():
    """初始化所有数据库表"""
    db_url = settings.database_url
    app_logger.info(f"连接数据库: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")

    try:
        # 1. 初始化业务表（用户、会话、消息）在第九章会使用
        # app_logger.info("初始化业务表...")
        # await init_db()
        # app_logger.info("✅ 业务表创建成功")

        # 2. 初始化 LangGraph Checkpointer 表（存储对话状态）
        async with AsyncConnectionPool(conninfo=db_url, min_size=2, max_size=10) as pool:
            app_logger.info("初始化 Checkpointer 表...")
            async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
                await checkpointer.setup()
                app_logger.info("[SUCCESS] LangGraph Checkpointer 表创建成功")

            # 3. 初始化 LangGraph Store 表（存储持久化数据）
            app_logger.info("初始化 Store 表...")
            async with AsyncPostgresStore.from_conn_string(db_url) as store:
                await store.setup()
                app_logger.info("[SUCCESS] LangGraph Store 表创建成功")

        app_logger.info("[SUCCESS] 所有数据库表初始化完成！")

    except Exception as e:
        app_logger.error(f"[ERROR] 数据库初始化失败: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database())
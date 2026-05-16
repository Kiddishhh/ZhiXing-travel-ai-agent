"""
FastAPI 应用工厂

lifespan 统一管理 checkpointer / memory_store / database 生命周期。
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.core.checkpointer import CheckpointerManager
from app.core.memory_store import MemoryStoreManager
from app.core.database import DatabaseManager
from app.config import settings
from app.utils.logger import app_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    app_logger.info("=== 启动服务 ===")

    checkpointer_mgr = await CheckpointerManager.get_instance()
    app_logger.info("Checkpointer 已就绪")

    memory_mgr = await MemoryStoreManager.get_instance()
    app_logger.info("MemoryStore 已就绪")

    db_mgr = await DatabaseManager.get_instance()
    app_logger.info("Database 已就绪")

    yield

    await db_mgr.close()
    await memory_mgr.close()
    await checkpointer_mgr.close()
    app_logger.info("=== 服务已关闭 ===")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="知行智能旅游规划助手",
        description="AI-driven travel planning assistant API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(v1_router, prefix="/api/v1")

    return app

"""知行智能旅游规划助手 · 启动入口

使用方法:
    python main.py
    或
    uv run python main.py
"""

import sys
import os

# Windows 兼容
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from app.config import settings


def main():
    print("=" * 56)
    print("  知行智能旅游规划助手  v0.1.0")
    print("  AI-Powered Travel Planning Assistant")
    print("=" * 56)
    print()
    print(f"  前端页面:  http://localhost:{settings.app_port}")
    print(f"  API 文档:  http://localhost:{settings.app_port}/docs")
    print(f"  ReDoc:     http://localhost:{settings.app_port}/redoc")
    print()
    print("  按 Ctrl+C 停止服务")
    print("=" * 56)

    uvicorn.run(
        "app.api.app:create_app",
        host=settings.app_host,
        port=settings.app_port,
        factory=True,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()

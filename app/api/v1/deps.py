"""
FastAPI 依赖注入

提供: get_current_user (JWT 解析), get_db (业务表连接池 + user_id),
      get_memory_manager (长期画像查询)
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from psycopg_pool import AsyncConnectionPool

from app.config import settings
from app.core.database import get_db_pool
from app.core.memory_store import get_memory_store_manager, MemoryStoreManager
from app.utils.logger import app_logger


JWT_SECRET = settings.dashscope_api_key[:32]
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = timedelta(days=7)

security = HTTPBearer()


def create_access_token(user_id: str, role: str = "user") -> str:
    """签发 JWT"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + JWT_EXPIRATION,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """解析 JWT，返回 payload"""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """从 JWT 解析当前用户"""
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌"
            )
        return {"user_id": user_id, "role": payload.get("role", "user")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌已过期"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌"
        )


async def get_db(
    current_user: dict = Depends(get_current_user),
) -> tuple[AsyncConnectionPool, str]:
    """获取业务表连接池 + 当前 user_id"""
    pool = await get_db_pool()
    return pool, current_user["user_id"]


async def get_memory_manager() -> MemoryStoreManager:
    """获取 MemoryStoreManager 实例"""
    return await get_memory_store_manager()

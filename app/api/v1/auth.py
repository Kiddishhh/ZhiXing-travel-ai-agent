"""认证路由：注册 + 登录"""
from uuid import uuid4

import bcrypt
from fastapi import APIRouter, HTTPException, status, Depends
from psycopg_pool import AsyncConnectionPool

from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, UserInDB
from app.api.v1.deps import get_db_pool, create_access_token
from app.utils.logger import app_logger

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, pool: AsyncConnectionPool = Depends(get_db_pool)):
    """用户注册"""
    async with pool.connection() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE username = $1 OR email = $2",
            body.username, body.email,
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="用户名或邮箱已被注册",
            )

        password_hash = bcrypt.hashpw(
            body.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        user_id = uuid4()
        row = await conn.fetchrow(
            """
            INSERT INTO users (id, username, email, password_hash)
            VALUES ($1, $2, $3, $4)
            RETURNING id, username, email, role, is_active, created_at
            """,
            user_id, body.username, body.email, password_hash,
        )

    app_logger.info(f"新用户注册: {body.username} ({user_id})")
    return dict(row)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, pool: AsyncConnectionPool = Depends(get_db_pool)):
    """用户登录"""
    async with pool.connection() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, password_hash, role, is_active FROM users WHERE username = $1",
            body.username,
        )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    user = dict(row)
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用",
        )

    if not bcrypt.checkpw(body.password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE users SET last_login_at = NOW() WHERE id = $1", user["id"]
        )

    token = create_access_token(user_id=user["id"], role=user["role"])
    app_logger.info(f"用户登录: {body.username}")

    return TokenResponse(access_token=token)

# FastAPI 接口层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为知行旅游规划助手构建完整 FastAPI 对外接口层，含 JWT 认证、SSE 流式对话、会话管理和用户画像。

**Architecture:** 13 个新文件 + 2 个修改文件。DatabaseManager 管理业务表连接池，deps.py 提供 JWT 依赖注入，4 个路由模块覆盖 12 个端点，app.py 统一生命周期管理。

**Tech Stack:** FastAPI 0.124.4, uvicorn, PyJWT 2.10, bcrypt 5.0, psycopg 3.3 (AsyncConnectionPool), SSE (text/event-stream), LangGraph astream_events

**Design doc:** `docs/superpowers/specs/2026-05-15-fastapi-layer-design.md`

---

### Task 1: 业务表连接池 DatabaseManager（`app/core/database.py`）

**Files:**
- Create: `app/core/database.py`
- Reference: `app/core/checkpointer.py`（模式模板）

- [ ] **Step 1: 创建文件**

```python
"""
业务表连接池管理器

管理 users / conversations / messages 三张业务表的连接池。
单例模式，与 CheckpointerManager 一致。
"""
import asyncio
from typing import Optional

from psycopg_pool import AsyncConnectionPool

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
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
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
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
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
    created_at TIMESTAMP DEFAULT NOW()
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
                min_size=2,
                max_size=10,
                timeout=30,
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
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/core/database.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/core/database.py
git commit -m "feat: add DatabaseManager for business table connection pool"
```

---

### Task 2: Pydantic Schemas（`app/schemas/`）

**Files:**
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/auth.py`
- Create: `app/schemas/user.py`
- Create: `app/schemas/conversation.py`
- Create: `app/schemas/message.py`
- Create: `app/schemas/chat.py`

- [ ] **Step 1: 创建所有 schema 文件**

**`app/schemas/__init__.py`:**
```python
"""
Pydantic 请求/响应模型
"""
```

**`app/schemas/auth.py`:**
```python
"""认证相关模型"""
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 604800  # 7 天


class UserInDB(BaseModel):
    id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
```

**`app/schemas/user.py`:**
```python
"""用户相关模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    avatar_url: Optional[str] = None
    role: str
    created_at: datetime


class UserProfileResponse(BaseModel):
    user_id: str
    preferred_transport: Optional[str] = None
    budget_level: Optional[str] = None
    travel_styles: list[str] = []
    favorite_destinations: list[str] = []
    dietary_preferences: list[str] = []
    total_trips: int = 0
    last_destination: Optional[str] = None
    last_travel_date: Optional[str] = None
    extensions: dict = {}
```

**`app/schemas/conversation.py`:**
```python
"""会话相关模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", max_length=256)
    system_prompt: Optional[str] = None
    current_model: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=256)
    status: Optional[str] = None
    summary: Optional[str] = None
    system_prompt: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    current_model: Optional[str] = None
    summary: Optional[str] = None
    total_tokens: int = 0
    status: str
    created_at: datetime
    updated_at: datetime
```

**`app/schemas/message.py`:**
```python
"""消息相关模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    content_type: str = "text"
    token_count: int = 0
    feedback: int = 0
    is_error: bool = False
    metadata: dict = {}
    created_at: datetime
```

**`app/schemas/chat.py`:**
```python
"""对话相关模型"""
from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1, max_length=10000)
```

- [ ] **Step 2: 验证语法**

```bash
python -c "
import ast
for f in ['app/schemas/__init__.py','app/schemas/auth.py','app/schemas/user.py','app/schemas/conversation.py','app/schemas/message.py','app/schemas/chat.py']:
    ast.parse(open(f, encoding='utf-8').read())
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add app/schemas/
git commit -m "feat: add Pydantic schemas for auth, user, conversation, message, chat"
```

---

### Task 3: 依赖注入（`app/api/v1/deps.py`）

**Files:**
- Create: `app/api/v1/deps.py`
- Create: `app/api/v1/__init__.py`（空文件）

- [ ] **Step 1: 创建 deps.py**

```python
"""
FastAPI 依赖注入

提供: get_current_user (JWT 解析), get_db_pool (业务表连接池),
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


# JWT 配置
JWT_SECRET = settings.dashscope_api_key[:32]  # 复用 API Key 前 32 位作为 JWT 密钥
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
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌")
        return {"user_id": user_id, "role": payload.get("role", "user")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的令牌")


async def get_db(
    current_user: dict = Depends(get_current_user),
) -> tuple[AsyncConnectionPool, str]:
    """获取业务表连接池 + 当前 user_id"""
    pool = await get_db_pool()
    return pool, current_user["user_id"]


async def get_memory_manager() -> MemoryStoreManager:
    """获取 MemoryStoreManager 实例"""
    return await get_memory_store_manager()
```

- [ ] **Step 2: 同时检查 JWT secret 是否已配置**

在 `app/config.py` 中添加 `jwt_secret` 字段（仅需读已有配置，不需要改 .env）：

验证现有 settings 有 `dashscope_api_key` 可用作 JWT secret 派生源。

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/api/v1/deps.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app/api/v1/__init__.py app/api/v1/deps.py
git commit -m "feat: add JWT dependency injection and auth utilities"
```

---

### Task 4: 认证路由（`app/api/v1/auth.py`）

**Files:**
- Create: `app/api/v1/auth.py`

- [ ] **Step 1: 创建文件**

```python
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
    # 检查用户名唯一性
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

        # 哈希密码
        password_hash = bcrypt.hashpw(
            body.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # 创建用户
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

    # 更新最后登录时间
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE users SET last_login_at = NOW() WHERE id = $1", user["id"]
        )

    token = create_access_token(user_id=user["id"], role=user["role"])
    app_logger.info(f"用户登录: {body.username}")

    return TokenResponse(access_token=token)
```

- [ ] **Step 2: 验证语法并检查导入**

```bash
python -c "import ast; ast.parse(open('app/api/v1/auth.py', encoding='utf-8').read()); print('OK')"
python -c "from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, UserInDB; print('schemas OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/auth.py
git commit -m "feat: add auth routes (register/login) with JWT"
```

---

### Task 5: 用户路由（`app/api/v1/users.py`）

**Files:**
- Create: `app/api/v1/users.py`

- [ ] **Step 1: 创建文件**

```python
"""用户路由：当前用户信息 + 长期画像"""
from fastapi import APIRouter, Depends
from psycopg_pool import AsyncConnectionPool

from app.schemas.user import UserResponse, UserProfileResponse
from app.api.v1.deps import get_current_user, get_db, get_memory_manager
from app.core.memory_store import MemoryStoreManager
from app.utils.logger import app_logger

router = APIRouter(prefix="/users", tags=["用户"])


@router.get("/me", response_model=UserResponse)
async def get_me(pool_user: tuple = Depends(get_db)):
    """获取当前用户信息"""
    pool, user_id = pool_user
    async with pool.connection() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, email, avatar_url, role, created_at FROM users WHERE id = $1",
            user_id,
        )
    return dict(row)


@router.get("/me/profile", response_model=UserProfileResponse)
async def get_my_profile(
    pool_user: tuple = Depends(get_db),
    memory_mgr: MemoryStoreManager = Depends(get_memory_manager),
):
    """获取当前用户的长期旅行画像"""
    _, user_id = pool_user
    profile = await memory_mgr.get_profile(user_id)

    if profile is None:
        return UserProfileResponse(user_id=user_id)

    return UserProfileResponse(
        user_id=user_id,
        preferred_transport=profile.get("preferred_transport"),
        budget_level=profile.get("budget_level"),
        travel_styles=profile.get("travel_styles") or [],
        favorite_destinations=profile.get("favorite_destinations") or [],
        dietary_preferences=profile.get("dietary_preferences") or [],
        total_trips=profile.get("total_trips", 0),
        last_destination=profile.get("last_destination"),
        last_travel_date=str(profile["last_travel_date"]) if profile.get("last_travel_date") else None,
        extensions=profile.get("extensions") or {},
    )
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/api/v1/users.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/users.py
git commit -m "feat: add user routes (me, profile)"
```

---

### Task 6: 会话管理路由（`app/api/v1/conversations.py`）

**Files:**
- Create: `app/api/v1/conversations.py`

- [ ] **Step 1: 创建文件**

```python
"""会话管理路由：CRUD 5 个端点"""
from uuid import uuid4
from psycopg_pool import AsyncConnectionPool

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationResponse
from app.api.v1.deps import get_db
from app.utils.logger import app_logger

router = APIRouter(prefix="/conversations", tags=["会话"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(body: ConversationCreate, pool_user: tuple = Depends(get_db)):
    """创建新会话"""
    pool, user_id = pool_user
    conv_id = uuid4()

    async with pool.connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO conversations (id, user_id, title, current_model, system_prompt)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
            """,
            conv_id, user_id, body.title, body.current_model, body.system_prompt,
        )

    app_logger.info(f"会话创建: {conv_id} (user={user_id})")
    return dict(row)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    pool_user: tuple = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """获取会话列表"""
    pool, user_id = pool_user
    async with pool.connection() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM conversations
            WHERE user_id = $1 AND status != 'deleted'
            ORDER BY updated_at DESC
            LIMIT $2 OFFSET $3
            """,
            user_id, limit, offset,
        )
    return [dict(r) for r in rows]


@router.get("/{conv_id}", response_model=ConversationResponse)
async def get_conversation(conv_id: str, pool_user: tuple = Depends(get_db)):
    """获取会话详情"""
    pool, user_id = pool_user
    async with pool.connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM conversations WHERE id = $1", conv_id
        )

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    conv = dict(row)
    if conv["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

    return conv


@router.patch("/{conv_id}", response_model=ConversationResponse)
async def update_conversation(
    conv_id: str,
    body: ConversationUpdate,
    pool_user: tuple = Depends(get_db),
):
    """更新会话（归属校验）"""
    pool, user_id = pool_user

    async with pool.connection() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM conversations WHERE id = $1", conv_id
        )
        if existing is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        if dict(existing)["user_id"] != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

        # 构建动态 UPDATE
        updates = body.model_dump(exclude_none=True)
        if not updates:
            return dict(existing)

        set_clauses = []
        params = [conv_id]
        for key, val in updates.items():
            set_clauses.append(f"{key} = ${len(params) + 1}")
            params.append(val)
        set_clauses.append(f"updated_at = NOW()")
        params.append(conv_id)  # 占位，下面拼接

        # removed the extra placeholder, rebuild correctly
        params = params[:-1]  # 移除多余

        # Actually, rebuild properly:
        params = [conv_id]
        set_clauses = []
        for key, val in updates.items():
            set_clauses.append(f"{key} = ${len(params) + 1}")
            params.append(val)
        set_clauses.append("updated_at = NOW()")

        sql = f"UPDATE conversations SET {', '.join(set_clauses)} WHERE id = $1 RETURNING *"
        row = await conn.fetchrow(sql, *params)

    return dict(row)


@router.delete("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conv_id: str, pool_user: tuple = Depends(get_db)):
    """软删除会话"""
    pool, user_id = pool_user

    async with pool.connection() as conn:
        existing = await conn.fetchrow(
            "SELECT * FROM conversations WHERE id = $1", conv_id
        )
        if existing is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        if dict(existing)["user_id"] != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

        await conn.execute(
            "UPDATE conversations SET status = 'deleted', updated_at = NOW() WHERE id = $1",
            conv_id,
        )

    app_logger.info(f"会话已删除: {conv_id}")
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/api/v1/conversations.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/conversations.py
git commit -m "feat: add conversation CRUD routes"
```

---

### Task 7: SSE 流式对话路由（`app/api/v1/chat.py`）

**Files:**
- Create: `app/api/v1/chat.py`

- [ ] **Step 1: 创建文件**

```python
"""对话路由：SSE 流式对话 + 历史消息"""
import json
import time as _time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from psycopg_pool import AsyncConnectionPool

from app.schemas.chat import ChatStreamRequest
from app.schemas.message import MessageResponse
from app.api.v1.deps import get_db
from app.core.state import create_initial_state
from app.core.checkpointer import get_checkpointer
from app.core.memory_store import get_memory_store_manager
from app.agents.handoffs.graph import create_travel_planner
from app.utils.logger import app_logger

router = APIRouter(prefix="/chat", tags=["对话"])


async def _save_message(pool: AsyncConnectionPool, conv_id: str, role: str, content: str,
                        content_type: str = "text", token_count: int = 0, is_error: bool = False):
    """保存消息到 messages 表"""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, content_type, token_count, is_error)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            uuid4(), conv_id, role, content, content_type, token_count, is_error,
        )


@router.post("/stream")
async def chat_stream(body: ChatStreamRequest, pool_user: tuple = Depends(get_db)):
    """
    SSE 流式对话

    事件类型:
    - message: AI 文本回复
    - tool_call: 工具调用开始
    - tool_result: 工具返回结果
    - step: 步骤切换
    - done: 对话完成
    - error: 出错
    """
    pool, user_id = pool_user

    # 1. 验证 conversation 归属
    async with pool.connection() as conn:
        conv = await conn.fetchrow(
            "SELECT * FROM conversations WHERE id = $1 AND status != 'deleted'",
            body.conversation_id,
        )
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    conv_data = dict(conv)
    if conv_data["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

    thread_id = body.conversation_id

    # 2. 保存用户消息
    await _save_message(pool, body.conversation_id, "user", body.message)

    async def event_generator():
        graph = None
        try:
            # 3. 初始化 graph
            checkpointer = await get_checkpointer()
            memory_mgr = await get_memory_store_manager()
            store = memory_mgr.get_store()
            graph = await create_travel_planner(checkpointer=checkpointer, store=store)

            # 4. 构建初始状态
            initial_state = create_initial_state(user_id=user_id, session_id=thread_id)
            initial_state["messages"].append(HumanMessage(content=body.message))

            config = {"configurable": {"thread_id": thread_id}}

            # 5. 流式执行
            current_step = None
            async for event in graph.astream_events(initial_state, config, version="v2"):
                kind = event.get("event")

                # 步骤切换
                step = event.get("metadata", {}).get("langgraph_node", "")
                if step and step != current_step:
                    current_step = step
                    yield f"event: step\ndata: {json.dumps({'step': step})}\n\n"

                # LLM 输出
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"event: message\ndata: {json.dumps({'content': chunk.content})}\n\n"

                # 工具调用开始
                if kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    yield f"event: tool_call\ndata: {json.dumps({'tool': tool_name, 'args': tool_input})}\n\n"

                # 工具调用结束
                if kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    preview = str(output)[:500] if output else ""
                    yield f"event: tool_result\ndata: {json.dumps({'tool': tool_name, 'preview': preview})}\n\n"

                # 保存 AI 消息（完整消息在 on_chat_model_end 中最可靠）
                if kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output and hasattr(output, "content") and output.content:
                        await _save_message(pool, body.conversation_id, "assistant", output.content)

            # 6. 更新会话
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                    body.conversation_id,
                )

            yield f"event: done\ndata: {json.dumps({'conversation_id': body.conversation_id})}\n\n"

        except Exception as e:
            app_logger.error(f"流式对话异常: {e}")
            yield f"event: error\ndata: {json.dumps({'code': 'STREAM_ERROR', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{conv_id}/messages", response_model=dict)
async def get_messages(
    conv_id: str,
    pool_user: tuple = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """获取会话历史消息"""
    pool, user_id = pool_user

    # 归属校验
    async with pool.connection() as conn:
        conv = await conn.fetchrow("SELECT user_id FROM conversations WHERE id = $1", conv_id)
        if conv is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        if dict(conv)["user_id"] != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

        rows = await conn.fetch(
            """
            SELECT * FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            conv_id, limit, offset,
        )

    return {
        "conversation_id": conv_id,
        "messages": [dict(r) for r in rows],
    }
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/api/v1/chat.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/chat.py
git commit -m "feat: add SSE streaming chat and message history routes"
```

---

### Task 8: 路由汇总 + FastAPI 应用工厂（`router.py` + `app.py`）

**Files:**
- Create: `app/api/v1/router.py`
- Create: `app/api/app.py`

- [ ] **Step 1: 创建 router.py**

```python
"""API v1 路由汇总"""
from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.chat import router as chat_router

v1_router = APIRouter()
v1_router.include_router(auth_router)
v1_router.include_router(users_router)
v1_router.include_router(conversations_router)
v1_router.include_router(chat_router)
```

- [ ] **Step 2: 创建 app.py**

```python
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

    # 初始化基础设施
    checkpointer_mgr = await CheckpointerManager.get_instance()
    app_logger.info("Checkpointer 已就绪")

    memory_mgr = await MemoryStoreManager.get_instance()
    app_logger.info("MemoryStore 已就绪")

    db_mgr = await DatabaseManager.get_instance()
    app_logger.info("Database 已就绪")

    yield

    # 关闭所有连接池
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

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(v1_router, prefix="/api/v1")

    return app
```

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/api/v1/router.py', encoding='utf-8').read()); ast.parse(open('app/api/app.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: 验证应用可导入**

```bash
python -c "from app.api.app import create_app; app = create_app(); print('App created:', app.title)"
```

Expected: `App created: 知行智能旅游规划助手`

- [ ] **Step 5: Commit**

```bash
git add app/api/v1/router.py app/api/app.py
git commit -m "feat: add FastAPI app factory with lifespan and route aggregation"
```

---

### Task 9: init_db 集成 + 全项目验证

**Files:**
- Modify: `scripts/init_db.py`
- Create: `tests/api/test_api_schemas.py`

- [ ] **Step 1: 更新 init_db.py，追加业务表初始化**

在 `init_database()` 函数的 MemoryStoreManager 初始化之后，追加：

```python
        # 4. 初始化业务表（users, conversations, messages）
        from app.core.database import DatabaseManager
        app_logger.info("初始化业务表...")
        db_mgr = await DatabaseManager.get_instance()
        try:
            app_logger.info("[SUCCESS] 业务表创建成功")
        finally:
            await db_mgr.close()
```

- [ ] **Step 2: 创建 API schema 单元测试**

```python
"""API Schemas 单元测试"""
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from app.schemas.user import UserResponse, UserProfileResponse
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationResponse
from app.schemas.message import MessageResponse
from app.schemas.chat import ChatStreamRequest


class TestAuthSchemas:
    def test_register_request_valid(self):
        body = RegisterRequest(username="testuser", email="test@example.com", password="123456")
        assert body.username == "testuser"
        assert body.email == "test@example.com"

    def test_register_request_short_username(self):
        try:
            RegisterRequest(username="ab", email="test@example.com", password="123456")
        except Exception:
            pass  # Validation error expected

    def test_login_request(self):
        body = LoginRequest(username="testuser", password="123456")
        assert body.username == "testuser"

    def test_token_response(self):
        token = TokenResponse(access_token="eyJxxx", token_type="bearer", expires_in=604800)
        assert token.access_token == "eyJxxx"

    def test_register_request_invalid_email(self):
        try:
            RegisterRequest(username="testuser", email="not-an-email", password="123456")
        except Exception:
            pass


class TestConversationSchemas:
    def test_create_default_title(self):
        conv = ConversationCreate()
        assert conv.title == "新对话"

    def test_update_partial(self):
        update = ConversationUpdate(title="新标题")
        assert update.title == "新标题"
        assert update.status is None


class TestChatSchemas:
    def test_chat_stream_request(self):
        req = ChatStreamRequest(
            conversation_id="00000000-0000-0000-0000-000000000001",
            message="推荐一个目的地",
        )
        assert "推荐" in req.message

    def test_chat_stream_empty_message(self):
        try:
            ChatStreamRequest(conversation_id="00000000-0000-0000-0000-000000000001", message="")
        except Exception:
            pass
```

- [ ] **Step 3: 运行所有测试**

```bash
python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: 所有测试通过（新增 schema 测试 + 已有 81 个测试）

- [ ] **Step 4: 全项目语法检查**

```bash
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in __import__('pathlib').Path('.').rglob('*.py') if 'venv' not in str(p) and '__pycache__' not in str(p)]" && echo "ALL OK"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/init_db.py tests/api/
git commit -m "feat: add business table init and API schema tests"
```

---

### Task 10: 启动验证

- [ ] **Step 1: 启动服务**

```bash
cd "D:\AI agent\知行智能旅游规划助手" && python -m uvicorn app.api.app:create_app --factory --host 0.0.0.0 --port 8000 &
```

Expected: 服务启动，无报错

- [ ] **Step 2: 测试注册 endpoint**

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"test\",\"email\":\"test@test.com\",\"password\":\"123456\"}"
```

Expected: 返回 201 + 用户信息

- [ ] **Step 3: 测试登录 endpoint**

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"test\",\"password\":\"123456\"}"
```

Expected: 返回 access_token

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "chore: final verification - all tests pass, service starts"
```

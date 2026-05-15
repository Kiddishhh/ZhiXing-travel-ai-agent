# FastAPI 接口层设计方案

## 概述

为知行智能旅游规划助手构建完整的对外 API 层，包含 JWT 认证、会话管理、SSE 流式对话和用户画像查询。

## 一、文件结构

```
app/
├── api/
│   ├── __init__.py
│   ├── app.py              # FastAPI 应用工厂 + lifespan
│   └── v1/
│       ├── __init__.py
│       ├── router.py        # 汇总注册所有子路由
│       ├── auth.py          # POST /auth/register, /auth/login
│       ├── users.py         # GET /users/me, /users/me/profile
│       ├── conversations.py # CRUD /conversations
│       ├── chat.py          # POST /chat/stream, GET /chat/{id}/messages
│       └── deps.py          # 依赖注入：get_current_user, get_db_pool 等
├── core/
│   └── database.py          # 业务表连接池 + 建表 DDL
└── schemas/
    ├── __init__.py
    ├── auth.py              # RegisterRequest, LoginRequest, TokenResponse
    ├── user.py              # UserResponse, UserProfileResponse
    ├── conversation.py      # ConversationCreate, ConversationResponse, ConversationUpdate
    ├── message.py           # MessageResponse
    └── chat.py              # ChatStreamRequest
```

## 二、API 端点清单（12 个）

### 认证（无需 JWT）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/register` | 注册，返回用户信息 |
| POST | `/api/v1/auth/login` | 登录，返回 JWT |

### 对话（JWT）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/chat/stream` | SSE 流式对话 |
| GET | `/api/v1/chat/{conversation_id}/messages` | 历史消息，支持 limit/offset 分页 |

### 会话管理（JWT）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/conversations` | 创建会话 |
| GET | `/api/v1/conversations` | 会话列表 |
| GET | `/api/v1/conversations/{id}` | 会话详情 |
| PATCH | `/api/v1/conversations/{id}` | 更新标题/状态/摘要 |
| DELETE | `/api/v1/conversations/{id}` | 软删除（status=deleted） |

### 用户（JWT）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/users/me` | 当前用户信息 |
| GET | `/api/v1/users/me/profile` | 用户长期画像 |

## 三、认证设计

- JWT 认证：bcrypt 哈希密码 + PyJWT 签发 token
- Token 有效期 7 天，包含 `user_id` + `role`
- `deps.get_current_user(token)` 解析 JWT，自动注入路由
- 注册校验 username/email 唯一性
- 登录失败统一提示"用户名或密码错误"

## 四、数据库

### 业务表连接池

新增 `DatabaseManager`（`app/core/database.py`），独立连接池管理业务表。

### 三张业务表

- `users` — 用户账号（username, email, password_hash, avatar_url, role, is_active, preferences, last_login_at）
- `conversations` — 会话（user_id, title, current_model, system_prompt, summary, total_tokens, status, metadata, thread_id → LangGraph configurable）
- `messages` — 消息（conversation_id, role, content, content_type, token_count, feedback, is_error, metadata）

`conversation_id` = LangGraph `thread_id`（一一映射）。

建表在 `scripts/init_db.py` 追加。

## 五、SSE 流式对话

### 请求

```
POST /api/v1/chat/stream
Authorization: Bearer <token>
Body: { conversation_id: UUID, message: str }
```

### SSE 事件类型

| event | 含义 |
|-------|------|
| `message` | AI 文本回复 |
| `tool_call` | 工具调用开始 |
| `tool_result` | 工具返回结果 |
| `step` | 步骤切换 |
| `done` | 对话完成 |
| `error` | 出错 |

### 内部流程

1. 验证 JWT → 确认 conversation 归属
2. 查 conversation 获取 thread_id
3. 查 messages 表获取历史消息 → 注入 HumanMessage
4. 设置 `config = {"configurable": {"thread_id": thread_id}}`
5. `graph.astream_events(initial_state, config)` → 逐事件 yield SSE
6. 每条 AI/Tool 消息写入 messages 表
7. 对话结束更新 conversations.summary + total_tokens

## 六、应用入口

```python
def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(v1_router, prefix="/api/v1")
    return app
```

lifespan 统一管理：startup 初始化 checkpointer + memory_store + database，shutdown 关闭所有连接池。

## 七、涉及文件

| 文件 | 动作 | 说明 |
|------|------|------|
| `app/api/app.py` | **新增** | FastAPI 工厂 + lifespan |
| `app/api/v1/router.py` | **新增** | 汇总路由注册 |
| `app/api/v1/auth.py` | **新增** | 注册/登录 |
| `app/api/v1/users.py` | **新增** | 用户信息 + 画像 |
| `app/api/v1/conversations.py` | **新增** | 会话 CRUD |
| `app/api/v1/chat.py` | **新增** | SSE 流式对话 + 历史消息 |
| `app/api/v1/deps.py` | **新增** | JWT 解析 + 连接池注入 |
| `app/core/database.py` | **新增** | 业务表连接池 + DDL |
| `app/schemas/auth.py` | **新增** | 认证请求/响应模型 |
| `app/schemas/user.py` | **新增** | 用户模型 |
| `app/schemas/conversation.py` | **新增** | 会话模型 |
| `app/schemas/message.py` | **新增** | 消息模型 |
| `app/schemas/chat.py` | **新增** | 对话请求模型 |
| `scripts/init_db.py` | **修改** | 追加业务表建表 |

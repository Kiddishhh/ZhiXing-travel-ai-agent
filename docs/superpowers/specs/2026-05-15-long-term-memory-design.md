# 用户长期记忆系统设计方案

## 概述

为旅行规划 Agent 增加跨会话的用户长期记忆能力，使用 PostgreSQL 持久化存储用户偏好画像，通过中间件自动注入 + 工具主动写入的模式实现记忆的全生命周期管理。

## 一、数据库表设计

### user_profiles 表

```sql
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(64) PRIMARY KEY,
    -- 核心偏好（高频查询，固定列）
    preferred_transport VARCHAR(20),           -- flight / train / driving
    budget_level VARCHAR(20),                  -- economy / comfort / luxury
    travel_styles JSONB DEFAULT '[]',          -- ["culture", "food", "relaxation"]
    favorite_destinations JSONB DEFAULT '[]',  -- ["西安", "成都", "东京"]
    dietary_preferences JSONB DEFAULT '[]',    -- ["川菜", "清真", "不吃香菜"]
    -- 历史统计
    total_trips INTEGER DEFAULT 0,
    last_destination VARCHAR(100),
    last_travel_date DATE,
    -- 扩展字段
    extensions JSONB DEFAULT '{}',             -- {"special_needs": "需要无障碍设施"}
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

- 6 个固定列覆盖高频查询场景（交通偏好、预算档位、旅行风格、偏好目的地、饮食偏好、扩展字段）
- `JSONB` 数组存储多值维度，避免关联表
- `extensions` 兜底未来扩展
- 保持极简——只有一张表，不拆分

## 二、Store 管理模块

### MemoryStoreManager（`app/core/memory_store.py`）

仿照 `CheckpointerManager` 单例模式设计：

```python
class MemoryStoreManager:
    _instance = None
    _lock = asyncio.Lock

    async def initialize(pool_kwargs)    # 建表 + 注入 AsyncPostgresStore
    async def close()                    # 关闭连接池
    async def get_store() -> AsyncPostgresStore

    # 记忆 CRUD
    async def get_profile(user_id) -> dict | None
    async def upsert_profile(user_id, fields: dict) -> dict
    async def delete_profile(user_id)
    async def list_user_ids() -> list[str]

    # 行程完成统计更新
    async def record_trip(user_id, trip_summary: dict)
```

**关键设计：**

- `get_profile(user_id)` 返回 `None` 时表示新用户，调用方静默跳过画像注入
- `upsert_profile` 使用 `INSERT ... ON CONFLICT (user_id) DO UPDATE`，单条 SQL 原子操作
- `record_trip` 自动更新 `total_trips + 1`、`last_destination`、`last_travel_date`
- 连接池复用现有的 `AsyncConnectionPool` 模式（`psycopg_pool`）

### 初始化

`scripts/init_db.py` 增加 `await memory_store.initialize()` 调用，与 checkpointer 一并初始化。

## 三、中间件注入

### 改 `StepConfigResolver.resolve()`

在 `app/core/middleware.py` 的 `resolve()` 方法末尾增加记忆注入步骤：

```
步骤 ④（新增）：调用 MemoryStoreManager.get_profile(user_id)
    ↓
格式化画像文本 → 追加到 step_prompt 末尾
```

**注入格式：**

```
[用户长期画像]
- 交通偏好: 高铁
- 预算档位: 舒适
- 旅行风格: 文化、美食
- 偏好目的地: 西安、成都
- 饮食偏好: 川菜
- 历史出行: 共3次，最近一次2026-03-15去西安
```

**关键约束：**

- `get_profile` 返回 `None` 时不注入，静默跳过
- 只在新会话首次 resolve 时查 DB（不缓存、不主动重载）
- 当前会话内不刷新画像——等到用户下一轮对话再实时拉取最新数据
- `MemoryStoreManager` 通过构造器依赖注入到 `StepConfigResolver`

## 四、记忆写入工具

### 4.1 `save_user_preference`

`app/tools/memory_tools.py`

```python
@tool
async def save_user_preference(
    preference_type: str,   # "transport" | "food" | "budget" | "style" | "custom"
    value: str,             # 偏好内容
    runtime: ToolRuntime = None,
) -> str:
```

- 用户显式表达偏好时由 Agent 调用（如 "我比较爱吃川菜"）
- `preference_type` 映射到 `user_profiles` 表对应列
- 内部调用 `MemoryStoreManager.upsert_profile(user_id, fields)`
- 返回确认信息

可用步骤：`requirement_collection` 起即可用（让用户一开始就能表达偏好）。

### 4.2 `auto_save_from_state`

```python
@tool
async def auto_save_from_state(
    runtime: ToolRuntime = None,
) -> str:
```

- 在 `order_generation` 步骤由 Agent 在订单完成后调用
- 从当前 TravelState 提取结构化画像：

| state 来源 | → user_profiles 字段 |
|---|---|
| `user_requirement.budget_level` | `budget_level` |
| `user_requirement.travel_styles` | `travel_styles`（合并去重） |
| `selected_transport` | `preferred_transport` |
| `selected_destination` | 追加到 `favorite_destinations` |
| `food_options[].type` | 推断 `dietary_preferences` |
| `travel_days`, `departure_date` | 更新统计字段 |

- 调用 `upsert_profile` 写入
- 返回简洁摘要

### 4.3 注册

两个工具注册到 `TOOL_REGISTRY`，`save_user_preference` 在所有步骤可用，`auto_save_from_state` 仅在 `order_generation` 可用。

## 五、画像字段合并策略

- **数组字段**（`travel_styles`、`favorite_destinations`、`dietary_preferences`）：新旧合并去重，最多保留 10 条
- **标量字段**（`budget_level`、`preferred_transport`）：新值覆盖旧值
- **统计字段**（`total_trips`、`last_destination`、`last_travel_date`）：累计 +1，更新日期

## 六、错误处理

| 场景 | 处理 |
|------|------|
| 数据库不可用 | 记 warning 日志，当作新用户，不阻塞主流程 |
| 新用户首次访问 | `get_profile` 返回 `None`，不注入画像，Agent 正常收集需求 |
| 写操作失败 | 记 warning 日志，不影响对话继续 |
| 并发写同一条画像 | PostgreSQL 行锁保证串行执行 |

**原则：记忆是增强功能，不是核心路径。挂了就降级，不影响规划主链路。**

## 七、涉及文件

| 文件 | 动作 | 说明 |
|------|------|------|
| `app/core/memory_store.py` | **新增** | MemoryStoreManager 单例 |
| `app/tools/memory_tools.py` | **新增** | `save_user_preference` + `auto_save_from_state` |
| `app/core/middleware.py` | **修改** | `resolve()` 增加画像注入 |
| `app/tools/__init__.py` | **修改** | 注册 2 个记忆工具 |
| `app/agents/handoffs/step_config.py` | **修改** | `order_generation` 的 tools 增加 `auto_save_from_state` |
| `scripts/init_db.py` | **修改** | 初始化时创建 user_profiles 表 |
| `app/agents/handoffs/graph.py` | **修改** | 注入 MemoryStoreManager 到 StepConfigResolver |
| `tests/tools/test_memory_tools.py` | **新增** | 记忆工具单元测试 |

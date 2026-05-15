# 用户长期记忆系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为旅行规划 Agent 增加跨会话用户长期记忆能力，通过中间件注入+工具写入模式实现用户偏好画像的持久化。

**Architecture:** 新增 `MemoryStoreManager` 单例管理 user_profiles 表，`StepConfigResolver.resolve()` 中注入画像文本到 prompt，`save_user_preference` 和 `auto_save_from_state` 两个工具覆盖显式保存和自动保存场景。

**Tech Stack:** PostgreSQL (psycopg_pool AsyncConnectionPool), LangGraph (AsyncPostgresStore), Python 3.11+

**Design doc:** `docs/superpowers/specs/2026-05-15-long-term-memory-design.md`

---

### Task 1: 创建 MemoryStoreManager（`app/core/memory_store.py`）

**Files:**
- Create: `app/core/memory_store.py`
- Reference: `app/core/checkpointer.py`（模式模板）

- [ ] **Step 1: 创建文件，写入完整实现**

```python
"""
用户长期记忆存储管理器

仿照 CheckpointerManager 单例模式。
管理 user_profiles 业务表 + LangGraph AsyncPostgresStore。
"""
import asyncio
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres import AsyncPostgresStore

from app.config import settings
from app.utils.logger import app_logger


CREATE_USER_PROFILES_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(64) PRIMARY KEY,
    preferred_transport VARCHAR(20),
    budget_level VARCHAR(20),
    travel_styles JSONB DEFAULT '[]',
    favorite_destinations JSONB DEFAULT '[]',
    dietary_preferences JSONB DEFAULT '[]',
    total_trips INTEGER DEFAULT 0,
    last_destination VARCHAR(100),
    last_travel_date DATE,
    extensions JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

UPSERT_PROFILE_SQL = """
INSERT INTO user_profiles (
    user_id, preferred_transport, budget_level,
    travel_styles, favorite_destinations, dietary_preferences,
    total_trips, last_destination, last_travel_date,
    extensions, updated_at
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
ON CONFLICT (user_id) DO UPDATE SET
    preferred_transport = COALESCE(EXCLUDED.preferred_transport, user_profiles.preferred_transport),
    budget_level = COALESCE(EXCLUDED.budget_level, user_profiles.budget_level),
    travel_styles = EXCLUDED.travel_styles,
    favorite_destinations = EXCLUDED.favorite_destinations,
    dietary_preferences = EXCLUDED.dietary_preferences,
    total_trips = EXCLUDED.total_trips,
    last_destination = COALESCE(EXCLUDED.last_destination, user_profiles.last_destination),
    last_travel_date = COALESCE(EXCLUDED.last_travel_date, user_profiles.last_travel_date),
    extensions = EXCLUDED.extensions,
    updated_at = NOW();
"""


class MemoryStoreManager:
    """用户长期记忆存储管理器（单例）"""

    _instance: Optional["MemoryStoreManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.pool: Optional[AsyncConnectionPool] = None
        self.store: Optional[AsyncPostgresStore] = None

    @classmethod
    async def get_instance(cls) -> "MemoryStoreManager":
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
            app_logger.info("初始化 MemoryStoreManager...")

            self.pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                min_size=settings.db_min_conn,
                max_size=settings.db_max_conn,
                timeout=settings.db_timeout,
            )
            await self.pool.open()

            # 建业务表
            async with self.pool.connection() as conn:
                await conn.execute(CREATE_USER_PROFILES_SQL)

            # 注入 LangGraph Store（给 graph compile 用）
            self.store = AsyncPostgresStore(self.pool)
            await self.store.setup()

            app_logger.info("MemoryStoreManager 初始化完成")
        except Exception as e:
            app_logger.error(f"MemoryStoreManager 初始化失败: {e}")
            if self.pool:
                await self.pool.close()
                self.pool = None
            raise

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
            self.store = None
            MemoryStoreManager._instance = None
            app_logger.info("MemoryStoreManager 连接池已关闭")

    def get_store(self) -> AsyncPostgresStore:
        if self.store is None:
            raise RuntimeError("MemoryStoreManager 未初始化")
        return self.store

    async def get_profile(self, user_id: str) -> Optional[dict]:
        """读取用户画像，不存在返回 None"""
        try:
            async with self.pool.connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE user_id = $1", user_id
                )
            if row is None:
                return None
            return dict(row)
        except Exception as e:
            app_logger.warning(f"读取用户画像失败 (user_id={user_id}): {e}")
            return None

    async def upsert_profile(self, user_id: str, fields: dict) -> dict:
        """创建或更新用户画像，返回写入后的完整画像"""
        try:
            async with self.pool.connection() as conn:
                # 先读旧画像（用于数组合并）
                old = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE user_id = $1", user_id
                )
                old_dict = dict(old) if old else {}

                # 数组合并去重，最多 10 条
                def merge_array(old_val, new_val):
                    if not new_val:
                        return old_val
                    merged = list(dict.fromkeys((new_val or []) + (old_val or [])))
                    return merged[:10]

                travel_styles = fields.get("travel_styles")
                favorite_destinations = fields.get("favorite_destinations")
                dietary_preferences = fields.get("dietary_preferences")

                if travel_styles is not None:
                    travel_styles = merge_array(
                        old_dict.get("travel_styles"), travel_styles
                    )
                else:
                    travel_styles = old_dict.get("travel_styles", [])

                if favorite_destinations is not None:
                    favorite_destinations = merge_array(
                        old_dict.get("favorite_destinations"), favorite_destinations
                    )
                else:
                    favorite_destinations = old_dict.get("favorite_destinations", [])

                if dietary_preferences is not None:
                    dietary_preferences = merge_array(
                        old_dict.get("dietary_preferences"), dietary_preferences
                    )
                else:
                    dietary_preferences = old_dict.get("dietary_preferences", [])

                # 标量字段：新值覆盖旧值
                preferred_transport = fields.get(
                    "preferred_transport", old_dict.get("preferred_transport")
                )
                budget_level = fields.get(
                    "budget_level", old_dict.get("budget_level")
                )

                # 统计字段
                total_trips = fields.get(
                    "total_trips", old_dict.get("total_trips", 0)
                )
                last_destination = fields.get(
                    "last_destination", old_dict.get("last_destination")
                )
                last_travel_date = fields.get(
                    "last_travel_date", old_dict.get("last_travel_date")
                )

                # 扩展字段
                new_extensions = fields.get("extensions") or {}
                old_extensions = old_dict.get("extensions") or {}
                extensions = {**old_extensions, **new_extensions}

                await conn.execute(
                    UPSERT_PROFILE_SQL,
                    user_id,
                    preferred_transport,
                    budget_level,
                    travel_styles,
                    favorite_destinations,
                    dietary_preferences,
                    total_trips,
                    last_destination,
                    last_travel_date,
                    extensions,
                )

                # 读回完整画像
                row = await conn.fetchrow(
                    "SELECT * FROM user_profiles WHERE user_id = $1", user_id
                )
                app_logger.info(f"画像已更新 (user_id={user_id})")
                return dict(row)
        except Exception as e:
            app_logger.warning(f"写入用户画像失败 (user_id={user_id}): {e}")
            return {}

    async def delete_profile(self, user_id: str):
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    "DELETE FROM user_profiles WHERE user_id = $1", user_id
                )
            app_logger.info(f"画像已删除 (user_id={user_id})")
        except Exception as e:
            app_logger.warning(f"删除用户画像失败 (user_id={user_id}): {e}")

    async def list_user_ids(self) -> list[str]:
        try:
            async with self.pool.connection() as conn:
                rows = await conn.fetch("SELECT user_id FROM user_profiles")
            return [r["user_id"] for r in rows]
        except Exception as e:
            app_logger.warning(f"列出用户ID失败: {e}")
            return []


async def get_memory_store_manager() -> MemoryStoreManager:
    """获取全局 MemoryStoreManager 实例"""
    return await MemoryStoreManager.get_instance()
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/core/memory_store.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/core/memory_store.py
git commit -m "feat: add MemoryStoreManager singleton for user profile persistence"
```

---

### Task 2: 集成到 init_db.py

**Files:**
- Modify: `scripts/init_db.py`

- [ ] **Step 1: 修改 init_database()，增加 user_profiles 表创建**

Replace the entire `init_database` function body to add MemoryStoreManager initialization:

```python
async def init_database():
    """初始化所有数据库表"""
    db_url = settings.database_url
    app_logger.info(f"连接数据库: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")

    try:
        # 1. 初始化 LangGraph Checkpointer 表（存储对话状态）
        async with AsyncConnectionPool(conninfo=db_url, min_size=2, max_size=10) as pool:
            app_logger.info("初始化 Checkpointer 表...")
            async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
                await checkpointer.setup()
                app_logger.info("[SUCCESS] LangGraph Checkpointer 表创建成功")

            # 2. 初始化 LangGraph Store 表（存储持久化数据）
            app_logger.info("初始化 Store 表...")
            async with AsyncPostgresStore.from_conn_string(db_url) as store:
                await store.setup()
                app_logger.info("[SUCCESS] LangGraph Store 表创建成功")

        # 3. 初始化用户长期记忆表（user_profiles）
        from app.core.memory_store import MemoryStoreManager
        app_logger.info("初始化用户长期记忆表...")
        memory_manager = await MemoryStoreManager.get_instance()
        try:
            app_logger.info("[SUCCESS] 用户长期记忆表创建成功")
        finally:
            await memory_manager.close()

        app_logger.info("[SUCCESS] 所有数据库表初始化完成！")

    except Exception as e:
        app_logger.error(f"[ERROR] 数据库初始化失败: {str(e)}")
        raise
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('scripts/init_db.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/init_db.py
git commit -m "feat: add user_profiles table init to init_db.py"
```

---

### Task 3: 创建记忆工具（`app/tools/memory_tools.py`）

**Files:**
- Create: `app/tools/memory_tools.py`

- [ ] **Step 1: 创建文件，实现 save_user_preference 和 auto_save_from_state**

```python
"""
用户长期记忆写入工具

save_user_preference — 用户显式声明偏好时由 Agent 调用
auto_save_from_state — 订单生成完成后从 TravelState 自动提取画像
"""
from langchain.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from app.core.state import TravelState
from app.core.memory_store import get_memory_store_manager
from app.utils.logger import app_logger


PREFERENCE_TYPE_MAP = {
    "transport": "preferred_transport",
    "food": "dietary_preferences",
    "budget": "budget_level",
    "style": "travel_styles",
    "custom": "extensions",
}


@tool
async def save_user_preference(
    preference_type: str,
    value: str,
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    保存用户的长期偏好到记忆库。

    参数说明:
    - preference_type: 偏好类型。可选值:
      "transport" — 交通偏好 (如 "高铁")
      "food" — 饮食偏好 (如 "川菜")
      "budget" — 预算偏好 (如 "舒适")
      "style" — 旅行风格 (如 "文化")
      "custom" — 自定义偏好 (如 "需要无障碍设施")
    - value: 偏好内容

    返回确认信息。
    """
    user_id = runtime.state.get("user_id", "unknown") if runtime else "unknown"
    app_logger.info(f"保存用户偏好: user_id={user_id}, type={preference_type}, value={value}")

    col = PREFERENCE_TYPE_MAP.get(preference_type)
    if col is None:
        return f"未知的偏好类型: {preference_type}，可选: {', '.join(PREFERENCE_TYPE_MAP.keys())}"

    manager = await get_memory_store_manager()

    if preference_type in ("food", "style"):
        # 数组字段：包装为列表后合并
        fields = {col: [value]}
    elif preference_type == "custom":
        # 扩展字段
        fields = {"extensions": {preference_type: value}}
    else:
        # 标量字段
        fields = {col: value}

    await manager.upsert_profile(user_id, fields)
    return f"已保存您的{preference_type}偏好: {value}"


@tool
async def auto_save_from_state(
    runtime: ToolRuntime[None, TravelState] = None,
) -> str:
    """
    从当前旅行规划状态中自动提取并保存用户偏好画像。

    在订单生成完成后调用，将本次行程的偏好和历史写入长期记忆。
    无需参数，自动从当前状态中提取。

    返回保存摘要。
    """
    if runtime is None:
        return "无法读取旅行状态，画像未保存。"

    state = runtime.state
    user_id = state.get("user_id", "unknown")
    req = state.get("user_requirement", {}) or {}

    app_logger.info(f"自动保存用户画像: user_id={user_id}")

    # 从 state 提取字段
    fields = {}

    # budget_level
    budget_level = req.get("budget_level")
    if budget_level:
        fields["budget_level"] = budget_level

    # travel_styles
    travel_styles = req.get("travel_styles", []) or []
    if travel_styles:
        fields["travel_styles"] = list(travel_styles)

    # preferred_transport
    selected_transport = state.get("selected_transport")
    if selected_transport:
        fields["preferred_transport"] = selected_transport

    # favorite_destinations
    selected_destination = state.get("selected_destination")
    if selected_destination:
        fields["favorite_destinations"] = [selected_destination]

    # dietary_preferences（从 food_options 推断）
    food_options = state.get("food_options", []) or []
    food_types = [f.get("type", "") for f in food_options if f.get("type")]
    if food_types:
        fields["dietary_preferences"] = food_types

    # 统计字段
    travel_days = req.get("travel_days", 0)
    departure_date = req.get("departure_date")
    if travel_days:
        fields["total_trips"] = 1  # upsert 时会递增
    if selected_destination:
        fields["last_destination"] = selected_destination
    if departure_date:
        fields["last_travel_date"] = departure_date

    manager = await get_memory_store_manager()
    result = await manager.upsert_profile(user_id, fields)

    if result:
        total_trips = result.get("total_trips", 0)
        last_dest = result.get("last_destination", "未知")
        return (
            f"已更新您的旅行画像：共 {total_trips} 次出行，"
            f"最近目的地 {last_dest}。"
        )
    return "画像保存完成。"
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/tools/memory_tools.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/tools/memory_tools.py
git commit -m "feat: add save_user_preference and auto_save_from_state tools"
```

---

### Task 4: 注册记忆工具到 TOOL_REGISTRY

**Files:**
- Modify: `app/tools/__init__.py`

- [ ] **Step 1: 在文件末尾添加 memory_tools 的 import 和注册**

在现有的 `register_tool("get_current_date", get_current_date)` 之后追加：

```python
# ── 注册记忆工具 ──
from .memory_tools import save_user_preference, auto_save_from_state

register_tool("save_user_preference", save_user_preference)
register_tool("auto_save_from_state", auto_save_from_state)
```

同时在文件顶部的 `__all__` 追加（如果有的话），或直接在末尾加即可。

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/tools/__init__.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/tools/__init__.py
git commit -m "feat: register memory tools in TOOL_REGISTRY"
```

---

### Task 5: 将 save_user_preference 加入各步骤的 tools 列表

**Files:**
- Modify: `app/agents/handoffs/step_config.py`

- [ ] **Step 1: import save_user_preference**

在文件顶部的 import 区域（`from app.tools.utility_tools import get_current_date` 下方）追加：

```python
from app.tools.memory_tools import save_user_preference
```

- [ ] **Step 2: 在步骤 1-8 的 tools 列表中追加 save_user_preference**

每个步骤的 `"tools"` 列表末尾追加 `save_user_preference`。例如步骤 1 的 tools 改为：

```python
"tools": [
    record_requirement_tool,
    check_current_progress,
    get_current_date,
    save_user_preference,
],
```

对所有 8 个步骤重复此操作。`order_generation`（步骤 8）还需要额外追加 `auto_save_from_state`：

```python
from app.tools.memory_tools import save_user_preference, auto_save_from_state

# ... 在 order_generation 的 tools 列表中:
"tools": [
    create_order,
    generate_order_tool,
    go_back_to_budget, go_back_to_itinerary, go_back_to_food,
    go_back_to_accommodation, go_back_to_transport,
    go_back_to_destination, go_back_to_requirement,
    go_back_to_step,
    check_current_progress,
    get_current_date,
    save_user_preference,
    auto_save_from_state,
],
```

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/agents/handoffs/step_config.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/agents/handoffs/step_config.py
git commit -m "feat: add memory tools to step configs"
```

---

### Task 6: 修改 StepConfigResolver 增加画像注入

**Files:**
- Modify: `app/core/middleware.py`

- [ ] **Step 1: 改造 resolve() 方法的签名和实现**

将 `resolve` 方法改为 async，接收 `user_id`，在返回 prompt 前查询并注入画像。

完整替换 `resolve` 方法和 `__init__`：

```python
from app.core.memory_store import get_memory_store_manager

class StepConfigResolver:
    """步骤配置解析器 - 根据 current_step 返回对应的 prompt 和 tools"""

    def __init__(self, step_config: dict):
        self._step_config = step_config

    # 状态注入 current_step, 返回对应步骤的 system_prompt and tools
    async def resolve(self, state: dict) -> tuple:
        current_step = state.get("current_step", "requirement_collection")

        app_logger.info(f"当前步骤: {current_step}")

        if current_step not in self._step_config:
            app_logger.error(f"未知步骤: {current_step}")
            raise ValueError(f"未知步骤: {current_step}")

        step_config = self._step_config[current_step]

        # ── 验证前置依赖 ──
        for required_field in step_config["requires"]:
            val = state.get(required_field)
            if val is None:
                error_msg = (
                    f"步骤 {current_step} 需要 '{required_field}' 字段, "
                    f"但当前未设置"
                )
                app_logger.error(f"前置依赖缺失: {error_msg}")
                raise ValueError(error_msg)
            app_logger.debug(f"前置依赖满足: {required_field}")

        # ── 渲染 prompt ──
        try:
            system_prompt = step_config["prompt"].format(**state)
        except KeyError as e:
            app_logger.warning(f"prompt 占位符无法渲染: {e}, 使用原始模板")
            system_prompt = step_config["prompt"]

        # ── 注入用户长期记忆 ──
        user_id = state.get("user_id")
        if user_id:
            try:
                manager = await get_memory_store_manager()
                profile = await manager.get_profile(user_id)
                if profile:
                    profile_text = _format_profile_for_prompt(profile)
                    system_prompt += f"\n\n{profile_text}"
                    app_logger.info(f"已注入用户画像 (user_id={user_id})")
            except Exception as e:
                app_logger.warning(f"画像注入失败，跳过: {e}")

        app_logger.info(f"已解析步骤配置: {len(step_config['tools'])} 个工具")
        return system_prompt, step_config["tools"]


def _format_profile_for_prompt(profile: dict) -> str:
    """将 user_profiles 行格式化为 prompt 可用的画像文本"""
    lines = ["[用户长期画像]"]

    transport = profile.get("preferred_transport")
    if transport:
        lines.append(f"- 交通偏好: {transport}")

    budget = profile.get("budget_level")
    if budget:
        lines.append(f"- 预算档位: {budget}")

    styles = profile.get("travel_styles") or []
    if styles:
        lines.append(f"- 旅行风格: {', '.join(styles)}")

    dests = profile.get("favorite_destinations") or []
    if dests:
        lines.append(f"- 偏好目的地: {', '.join(dests)}")

    diets = profile.get("dietary_preferences") or []
    if diets:
        lines.append(f"- 饮食偏好: {', '.join(diets)}")

    total = profile.get("total_trips", 0)
    if total:
        last_dest = profile.get("last_destination", "") or ""
        last_date = profile.get("last_travel_date", "") or ""
        parts = [f"共{total}次"]
        if last_date:
            parts.append(f"最近一次{last_date}")
        if last_dest:
            parts.append(f"去{last_dest}")
        lines.append(f"- 历史出行: {'，'.join(parts)}")

    extensions = profile.get("extensions") or {}
    for k, v in extensions.items():
        if v:
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)
```

**注意：** `resolve` 方法签名从 `def resolve(self, state: dict) -> tuple:` 改为 `async def resolve(self, state: dict) -> tuple:`。

- [ ] **Step 2: 更新 agent_node 中的 resolve 调用**

`graph.py` 的 `_make_agent_node` 中 `resolver.resolve(state)` 需要加 `await`：

在 `app/agents/handoffs/graph.py` 的 `agent_node` 函数中，将：

```python
system_prompt, tools = resolver.resolve(state)
```

改为：

```python
system_prompt, tools = await resolver.resolve(state)
```

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/core/middleware.py', encoding='utf-8').read()); print('OK')"
python -c "import ast; ast.parse(open('app/agents/handoffs/graph.py', encoding='utf-8').read()); print('OK')"
```

Expected: `OK` `OK`

- [ ] **Step 4: Commit**

```bash
git add app/core/middleware.py app/agents/handoffs/graph.py
git commit -m "feat: inject user profile into step prompt via middleware"
```

---

### Task 7: 编写单元测试

**Files:**
- Create: `tests/tools/test_memory_tools.py`

- [ ] **Step 1: 创建测试文件**

```python
"""用户长期记忆工具测试"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.tools.memory_tools import save_user_preference, auto_save_from_state


def _print_stage(stage: str, total: int, current: int):
    print(f"\n{'─'*50}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'─'*50}")


def make_mock_runtime(state_overrides=None):
    """构造模拟 ToolRuntime"""
    state = {
        "current_step": "order_generation",
        "user_id": "test-user-001",
        "session_id": "test-session-001",
        "user_requirement": {
            "departure_city": "成都",
            "destination": "重庆",
            "departure_date": "2026-06-15",
            "travel_days": 2,
            "adult_count": 2,
            "children_count": 0,
            "budget_min": 1000,
            "budget_max": 3000,
            "budget_level": "comfort",
            "travel_styles": ["food", "culture"],
            "special_needs": None,
        },
        "selected_destination": "重庆",
        "selected_transport": "train",
        "selected_accommodation_types": ["star_hotel"],
        "selected_food_types": ["specialty"],
        "transport_options": [
            {"transport_type": "train", "details": "G1234", "price": 150.0}
        ],
        "accommodation_options": [
            {"name": "重庆解放碑酒店", "price_per_night": 350.0}
        ],
        "food_options": [
            {"type": "specialty", "estimated_daily_cost": 120.0}
        ],
    }
    if state_overrides:
        state.update(state_overrides)

    mock_runtime = MagicMock()
    mock_runtime.state = state
    mock_runtime.tool_call_id = "test-memory-001"
    return mock_runtime


class TestSaveUserPreference:
    """测试显式偏好保存工具"""

    @pytest.mark.asyncio
    async def test_save_transport_preference(self):
        """测试保存交通偏好"""
        _print_stage("save_user_preference — transport", 4, 1)
        print("[注入] preference_type='transport', value='高铁'")
        runtime = make_mock_runtime()

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.upsert_profile = AsyncMock(return_value={
                "user_id": "test-user-001",
                "preferred_transport": "高铁",
                "total_trips": 0,
            })
            mock_get_mgr.return_value = mock_mgr

            result = await save_user_preference.func(
                preference_type="transport",
                value="高铁",
                runtime=runtime,
            )

        assert isinstance(result, str)
        assert "高铁" in result
        print(f"[OK] 返回: {result}")

    @pytest.mark.asyncio
    async def test_save_food_preference(self):
        """测试保存饮食偏好（数组字段）"""
        _print_stage("save_user_preference — food", 4, 2)
        print("[注入] preference_type='food', value='川菜'")
        runtime = make_mock_runtime()

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.upsert_profile = AsyncMock(return_value={
                "user_id": "test-user-001",
                "dietary_preferences": ["川菜"],
            })
            mock_get_mgr.return_value = mock_mgr

            result = await save_user_preference.func(
                preference_type="food",
                value="川菜",
                runtime=runtime,
            )

        assert "川菜" in result
        print(f"[OK] 返回: {result}")

    @pytest.mark.asyncio
    async def test_save_invalid_preference_type(self):
        """测试无效偏好类型"""
        _print_stage("save_user_preference — invalid type", 4, 3)
        print("[注入] preference_type='invalid', value='xxx'")
        runtime = make_mock_runtime()

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_get_mgr.return_value = mock_mgr

            result = await save_user_preference.func(
                preference_type="invalid",
                value="xxx",
                runtime=runtime,
            )

        assert "未知" in result
        print(f"[OK] 返回: {result}")


class TestAutoSaveFromState:
    """测试自动保存工具"""

    @pytest.mark.asyncio
    async def test_auto_save_full_state(self):
        """测试从完整 state 自动提取并保存画像"""
        _print_stage("auto_save_from_state", 2, 1)
        runtime = make_mock_runtime()
        print("[注入] 2人重庆2日游 comfort 档，train + specialty")

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.upsert_profile = AsyncMock(return_value={
                "user_id": "test-user-001",
                "total_trips": 5,
                "last_destination": "重庆",
                "last_travel_date": "2026-06-15",
            })
            mock_get_mgr.return_value = mock_mgr

            result = await auto_save_from_state.func(runtime=runtime)

        assert isinstance(result, str)
        assert "5" in result or "重庆" in result
        print(f"[OK] 返回: {result}")

    @pytest.mark.asyncio
    async def test_auto_save_db_failure_graceful(self):
        """测试数据库不可用时降级不崩溃"""
        _print_stage("auto_save_from_state — DB failure", 2, 2)
        runtime = make_mock_runtime()
        print("[注入] get_memory_store_manager 抛出异常")

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            side_effect=Exception("数据库连接失败"),
        ):
            result = await auto_save_from_state.func(runtime=runtime)

        # 不抛异常，返回错误提示
        assert isinstance(result, str)
        print(f"[OK] 降级处理: {result}")
```

- [ ] **Step 2: 运行测试，验证全部通过**

Run: `python -m pytest tests/tools/test_memory_tools.py -v -s`
Expected: 5 passed

- [ ] **Step 3: Commit**

```bash
git add tests/tools/test_memory_tools.py
git commit -m "test: add memory tools unit tests"
```

---

### Task 8: 端到端验证

**Files:**
- 无新建，用现有 interactive_flow.py 测试

- [ ] **Step 1: 确保 db 已初始化**

```bash
python scripts/init_db.py
```

Expected: 所有表创建成功，包括 `user_profiles`

- [ ] **Step 2: 验证语法完整性（全项目）**

```bash
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in __import__('pathlib').Path('.').rglob('*.py') if 'venv' not in str(p) and '__pycache__' not in str(p)]" && echo "ALL OK"
```

Expected: `ALL OK`

- [ ] **Step 3: 运行所有已存在测试确保无回归**

```bash
python -m pytest tests/tools/ -v -s
```

Expected: 全部通过（含新增的 5 个 memory tools 测试）

- [ ] **Step 4: Commit 最终确认**

```bash
git add -A
git commit -m "chore: final syntax check and test pass after memory system integration"
```

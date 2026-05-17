# 中间件注入 Prompt 和 Tools 优化设计

## 目标

将当前自定义 StateGraph 中 `StepConfigResolver.resolve()` + 手动消息组装的方式，迁移到 `langchain.agents.create_agent` + `AgentMiddleware.awrap_model_call` 标准模式，同时保留上下文压缩能力。

## 动机

当前 `graph.py` 中 `_make_agent_node` 和 `_make_guard_node` 两个闭包承担了四个职责：上下文压缩、步骤配置解析、prompt 渲染、消息组装。这些与 LLM 调用的核心逻辑耦合在自定义图结构中。`langchain.agents` 提供了 `AgentMiddleware` 标准扩展点，可以让这些横切关注点独立、可测试、可组合。

## 架构对比

### 当前

```
START → guard(压缩) → agent(解析+拼消息+bind_tools) → tools → guard → ...
```

自定义 StateGraph，两个闭包 `_make_guard_node` 和 `_make_agent_node` 手动管理所有逻辑。

### 迁移后

```
create_agent 内部循环:
  abefore_model(压缩) → awrap_model_call(注入prompt+tools) → LLM → ToolNode → ...
```

`create_agent` 管理图结构，`TravelPlannerMiddleware` 通过两个钩子注入业务逻辑。

## 组件设计

### TravelPlannerMiddleware(AgentMiddleware)

继承 `langchain.agents.middleware.AgentMiddleware`，一个类承担两个职责：

| 钩子 | 职责 | 对应旧代码 |
|------|------|-----------|
| `abefore_model` | token 计数 + 上下文压缩，返回 `RemoveMessage` + `context_summary` | `_make_guard_node` |
| `awrap_model_call` | 读 `current_step` → 渲染 prompt → 注入画像 → `request.override(system_message, tools)` | `StepConfigResolver.resolve()` + agent_node 消息组装 |

#### abefore_model

```python
async def abefore_model(self, state: TravelState, runtime) -> dict | None:
    messages = list(state["messages"])
    token_count = count_tokens_approximately(messages)

    if token_count <= COMPRESSION_MAX_TOKENS or len(messages) <= COMPRESSION_KEEP_RECENT:
        return None

    old_msgs = messages[:-COMPRESSION_KEEP_RECENT]
    # 构建压缩请求 → LLM 生成摘要
    # 如已有 context_summary，合并重压缩
    # 失败时降级为简单截断

    return {
        "messages": [RemoveMessage(id=...) for m in old_msgs],
        "context_summary": summary,
    }
```

#### awrap_model_call

```python
async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
    current_step = request.state.get("current_step", "requirement_collection")
    cfg = self._step_config[current_step]

    # 1. 验证前置依赖
    for field in cfg["requires"]:
        if request.state.get(field) is None:
            raise ValueError(f"步骤 {current_step} 需要 '{field}'")

    # 2. 渲染 prompt（缺失占位符降级为原始模板）
    try:
        system_prompt = cfg["prompt"].format(**request.state)
    except KeyError:
        system_prompt = cfg["prompt"]

    # 3. 追加 context_summary 和用户画像
    if request.state.get("context_summary"):
        system_prompt += f"\n\n[已收集的旅行信息]\n\n{request.state['context_summary']}"
    system_prompt = await self._inject_profile(request.state, system_prompt)

    # 4. 不可变覆盖，交给 handler
    modified = request.override(
        system_message=SystemMessage(content=system_prompt),
        tools=cfg["tools"],
    )
    return await handler(modified)
```

### create_travel_planner

```python
async def create_travel_planner(checkpointer=None, store=None):
    middleware = await create_travel_planner_middleware()

    llm = ChatOpenAI(
        model=settings.qwen_model_name,
        api_key=settings.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    return create_agent(
        model=llm,
        tools=list(TOOL_REGISTRY.values()),
        middleware=[middleware],
        state_schema=TravelState,
        checkpointer=checkpointer,
        store=store,
    )
```

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/agents/handoffs/graph.py` | 重写 | 从 ~200 行缩减为 ~50 行，删除 `_make_agent_node`、`_make_guard_node`、手动图构建，改为调用 `create_agent` |
| `app/core/middleware.py` | 重写 | 从 `StepConfigResolver` 类变为 `TravelPlannerMiddleware(AgentMiddleware)` 子类 |
| `app/agents/handoffs/step_config.py` | 不改 | 8 步配置（prompt 模板 + 工具列表 + 前置依赖）保持不变 |
| `app/tools/state_transition.py` | 不改 | 17 个 Command 工具不变 |
| `app/core/state.py` | 不改 | TravelState 不变 |
| `app/tools/__init__.py` | 不改 | TOOL_REGISTRY 不变 |

## 错误处理

| 场景 | 策略 |
|------|------|
| 未知步骤 | `ValueError` 直接抛出 |
| 前置依赖缺失 | `ValueError` 直接抛出 |
| prompt 占位符无法渲染 | 降级为原始模板（保留 `{field}` 花括号） |
| 画像注入失败 | 静默跳过，不阻塞 LLM 调用 |
| handler(LLM) 异常 | 不捕获，向上传播 |

## 测试策略

### 单元测试

- `test_middleware_injects_prompt_and_tools` — mock handler，验证 `request.override` 参数
- `test_middleware_rejects_missing_prerequisite` — 缺前置依赖时抛异常
- `test_compression_skips_under_threshold` — token 不足时不压缩，返回 None
- `test_compression_triggers_over_threshold` — 超阈值时返回 `RemoveMessage` + `context_summary`
- `test_prompt_rendering_fallback` — `{missing_field}` 降级为原始模板
- 遍历 8 步骤：正常渲染、前置依赖、工具列表正确性

### 集成测试

- 用 mock LLM 跑完整 8 步流程，验证 `create_agent` + middleware 端到端可用

## 设计决策

1. **单个中间件类而非多个** — 压缩和配置注入共享 `step_config` 和 `memory_manager`，拆成两个类会增加不必要的复杂度
2. **用 `awrap_model_call` 而非 `abefore_model` 注入 prompt/tools** — `awrap_model_call` 提供 `ModelRequest` 对象，可以通过 `override()` 精确控制 system_message 和 tools，语义更清晰
3. **保留 `step_config.py` 不做改动** — 配置数据层的稳定性能降低迁移风险，未来可以进一步优化配置格式（如 YAML 文件）
4. **`handler` 回调不捕获异常** — LLM 调用失败应由上层（Tenacity 重试、调用方）处理，中间件保持单一职责

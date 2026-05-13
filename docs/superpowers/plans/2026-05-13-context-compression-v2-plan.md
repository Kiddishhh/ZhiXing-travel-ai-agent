# 上下文压缩 v2 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans.

**Goal:** 修复消息顺序错乱导致 AI 不跟随步骤 prompt 的问题，简化 guard 节点，优化压缩参数。

**Architecture:** 三层消息隔离 — 步骤 prompt（指令层）→ 记忆（事实数据层）→ 对话（交互历史层）。guard 移除 SystemMessage 分离逻辑。单文件改动：`graph.py`。

**Tech Stack:** LangGraph, LangChain messages, count_tokens_approximately

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `app/agents/handoffs/graph.py` | 修改 | 参数 + agent_node + guard_node + prompt |
| `tests/test_agents/test_context_compression.py` | 修改 | 删除过时测试 + 适配新 guard 逻辑 |

---

### Task 1: 修改 graph.py（参数 + agent_node + guard_node + prompt）

**Files:**
- Modify: `app/agents/handoffs/graph.py`

- [ ] **Step 1: 调整参数**

```python
# Line 30: 8000 → 12000
COMPRESSION_MAX_TOKENS = 12000
```

- [ ] **Step 2: 优化压缩 prompt（500→800字 + 禁止描述 AI 行为）**

```python
COMPRESSION_SYSTEM_PROMPT = """你是一个对话摘要专家。请将以下旅行规划对话压缩为简洁摘要。

压缩规则：
1. 只提取事实数据：日期、目的地、人数、预算、已选选项（交通/住宿/餐饮）
2. 只提取用户偏好和特殊需求
3. 只提取工具返回的具体数据（航班号、酒店名、价格、菜品等）
4. 禁止描述 AI 做过什么、问过什么、建议过什么
5. 去除闲聊、重复确认、冗余表达
6. 使用中文，不超过 800 字

待压缩对话：
{messages_to_compress}

请生成摘要："""
```

- [ ] **Step 3: 简化 guard_node — 移除 SystemMessage 分离**

将 guard_node 内的这段代码（lines 60-74）：
```python
        # 分离 SystemMessage（保留不动）和可压缩消息
        system_msgs = []
        compressible = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                compressible.append(msg)

        # 如果可压缩消息不足，不压缩
        if len(compressible) <= COMPRESSION_KEEP_RECENT:
            return {}

        # 分离旧消息和最近消息
        old_msgs = compressible[:-COMPRESSION_KEEP_RECENT]
```

改为：
```python
        # 如果消息不足，不压缩
        if len(messages) <= COMPRESSION_KEEP_RECENT:
            return {}

        # 分离：旧消息(压缩) + 最近消息(保留)
        old_msgs = messages[:-COMPRESSION_KEEP_RECENT]
```

- [ ] **Step 4: 修复 agent_node 消息顺序 — 三步分层**

将 agent_node 内的消息拼接（lines 168-182）：
```python
        # 构建注入 LLM 的消息列表（临时，不存入 state）
        messages = []

        # 1. 上下文摘要（由 guard 节点生成，临时注入）
        context_summary = state.get("context_summary")
        if context_summary:
            messages.append(
                SystemMessage(content=f"[对话历史摘要]\n\n{context_summary}")
            )

        # 2. 步骤 prompt（由 StepConfigResolver 生成，临时注入）
        messages.append(SystemMessage(content=system_prompt))

        # 3. 对话历史（Human/AI/Tool 消息，来自 state）
        messages.extend(state["messages"])
```

改为：
```python
        # 构建注入 LLM 的三层消息（临时，不存入 state）
        messages = []

        # 第1层: 指令层 — 当前步骤 prompt（优先级最高，最先被 LLM 读取）
        messages.append(SystemMessage(content=system_prompt))

        # 第2层: 记忆层 — 已收集的旅行事实数据（辅助参考）
        context_summary = state.get("context_summary")
        if context_summary:
            messages.append(
                SystemMessage(content=f"[已收集的旅行信息]\n\n{context_summary}")
            )

        # 第3层: 对话层 — 交互历史
        messages.extend(state["messages"])
```

- [ ] **Step 5: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/agents/handoffs/graph.py', encoding='utf-8').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

---

### Task 2: 更新测试

**Files:**
- Modify: `tests/test_agents/test_context_compression.py`

- [ ] **Step 1: 删除两个过时测试**

删除 `test_all_system_messages_no_compression` 和 `test_preserves_system_messages`。

`test_all_system_messages_no_compression` 删除原因：guard 不再分离 SystemMessage，全 SystemMessage 场景不可能出现。

`test_preserves_system_messages` 删除原因：guard 不再需要区分消息类型。

- [ ] **Step 2: 从测试 state 中移除 SystemMessage**

以下 4 个测试函数中的 `state = {"messages": [system_msg] + old_msgs + ...}` 去掉 `system_msg`：

`test_few_messages_no_compression`：
```python
# 改为纯对话消息
state = {
    "messages": [
        HumanMessage(content="我想去北京"),
        AIMessage(content="好的，北京是个不错的选择"),
    ]
}
```

`test_compresses_when_exceeds_threshold`：
```python
# 去掉 system_msg 变量
state = {"messages": old_msgs + recent_msgs}
```

`test_keeps_recent_messages`：
```python
# 去掉 sys_msg 变量
state = {"messages": old_msgs + recent}
```

`test_fallback_on_llm_error`：
```python
# 去掉 sys_msg 变量
state = {"messages": old_msgs + recent_msgs}
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_agents/test_context_compression.py -v
```

Expected: 4/4 tests PASS（删除了 2 个，剩余 4 个）

- [ ] **Step 4: 运行全量测试**

```bash
python -m pytest tests/ -v --ignore=tests/test_api --ignore=scripts -q
```

Expected: 0 新增失败

---

### Task 3: Commit

- [ ] **Step 1: 提交**

```bash
git add app/agents/handoffs/graph.py tests/test_agents/test_context_compression.py docs/superpowers/specs/2026-05-13-context-compression-v2-design.md docs/superpowers/plans/2026-05-13-context-compression-v2-plan.md
git commit -m "$(cat <<'EOF'
fix: reorder message layers in agent_node, simplify guard, optimize compression params

- agent_node: step prompt first (instruction priority), then summary (facts), then conversation
- guard_node: remove obsolete SystemMessage filtering
- COMPRESSION_MAX_TOKENS: 8000 → 12000
- summary limit: 500 → 800 chars, forbid describing AI behavior
EOF
)"
```

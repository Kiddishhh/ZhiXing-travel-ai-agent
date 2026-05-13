# 上下文压缩设计

## 概述

在主流程 agent 调用前，基于 token 阈值自动触发 LLM 压缩，将历史对话压缩为摘要，降低后续 LLM 调用的 token 消耗，同时保持关键信息不丢失。

## 触发条件

- **自动触发**：消息列表 token 数超过 `COMPRESSION_MAX_TOKENS`（默认 8000）
- **不压缩**：SystemMessage（步骤 prompt + 摘要）始终排除在压缩候选之外

## 压缩范围

| 消息类型 | 是否压缩 | 说明 |
|----------|----------|------|
| SystemMessage（步骤 prompt） | 否 | 始终保留 |
| SystemMessage（历史摘要） | 否 | 已压缩过的摘要不再重复压缩 |
| HumanMessage | 是 | 用户输入和工具结果 |
| AIMessage | 是 | LLM 回复（含 tool_calls） |
| ToolMessage | 是 | 工具执行结果 |

## 图结构

```
START → guard ──→ agent ⇄ tools → END
```

- 从当前 `START → agent` 改为经过 guard 节点
- guard 负责 token 检测和压缩，agent 逻辑不变

## 数据流

```
state.messages ──→ guard_node
                       │
              count_tokens_approximately(messages)
                       │
              ┌────────┴────────┐
           ≤阈值              >阈值
              │                  │
           透传 {}       1. 分离 SystemMessage（保留）
              │          2. 分离最近 4 条消息（保留）
              │          3. 其余 → 压缩候选
              │          4. qwen-max 生成摘要（≤500字）
              │          5. 返回 RemoveMessage(旧消息) + SystemMessage(摘要)
              │
         agent_node ←─────────────────
              │
    StepConfigResolver.resolve(state)
    注入步骤 SystemMessage
              │
    llm.bind_tools(tools).ainvoke(messages)
              │
    return {"messages": [response]}
```

## 压缩 Prompt

```
你是一个对话摘要专家。请将以下旅行规划对话压缩为简洁摘要。

压缩规则：
1. 保留所有关键事实：日期、目的地、人数、预算、已选选项（交通/住宿/餐饮）
2. 保留用户的特殊需求和偏好
3. 保留工具调用返回的具体数据（航班号、酒店名、价格、菜品等）
4. 去除闲聊、重复确认、冗余表达
5. 使用中文，不超过 500 字

待压缩对话：
{messages_to_compress}

请生成摘要：
```

## 实现位置

全部实现在 `app/agents/handoffs/graph.py`：

- `COMPRESSION_SYSTEM_PROMPT` — 压缩 prompt 常量
- `COMPRESSION_MAX_TOKENS` — token 阈值（8000）
- `COMPRESSION_KEEP_RECENT` — 保留最近消息数（4）
- `_make_guard_node(llm)` — 创建 guard 节点闭包（传入 ChatTongyi 实例用于压缩）
- `create_travel_planner()` — 图中注册 guard 节点

## 关键依赖

- `langchain_core.messages.utils.count_tokens_approximately` — token 近似估算
- `langgraph.graph.message.RemoveMessage` — 移除旧消息
- `ChatTongyi(model="qwen-max")` — 用于压缩的 LLM（与主流程复用同一实例）

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| COMPRESSION_MAX_TOKENS | 8000 | 触发压缩的 token 阈值 |
| COMPRESSION_KEEP_RECENT | 4 | 压缩后保留最近 N 条消息 |
| 摘要字数上限 | 500 | 压缩 prompt 中约束 |

## 错误处理

- 压缩 LLM 调用失败 → 降级为简单截断（保留最近 N 条消息 + 删除更早的消息）
- 降级时不生成摘要，仅做 RemoveMessage 清理

## 测试要点

- 低于阈值时不触发压缩（透传）
- 高于阈值时正确分离和压缩
- SystemMessage 不被压缩
- RemoveMessage 正确删除旧消息
- 摘要 SystemMessage 正确注入
- 压缩 LLM 失败时的降级行为

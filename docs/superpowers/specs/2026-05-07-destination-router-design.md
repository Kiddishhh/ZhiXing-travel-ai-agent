# Destination Router — 并行探索/天气 Agent 工作流

## 背景

在旅游规划助手中，用户查询通常涉及多个维度：景点攻略、天气信息等。需要一个 Router 层将查询分类后并行分发给不同 Agent，最后汇总结果。

## 架构

```
用户查询 → classifier_node (LLM 分类)
                ↓
         route_to_agents (Send 并行)
              ↙         ↘
    explore_agent    weather_agent
      (RAG 检索)      (占位)
              ↘         ↙
            agent_results 累加
                  ↓
            final_report 汇总
```

## State 定义

使用 `TypedDict` + `Annotated[list, add]` 实现结果累加：

- **`Classification`**: `agent` (explore/weather) + `query` (子查询)
- **`AgentOutput`**: `agent_name` + `result`
- **`DestinationRouterState`**: 原始查询、目的地、分类列表、Agent 结果（累加）、综合报告

## 组件

| 组件 | 输入 | 输出 | 实现 |
|------|------|------|------|
| `classifier_node` | state | 更新 classifications | ChatTongyi 结构化输出 `ClassificationResult` |
| `route_to_agents` | state | `Send` 列表 | 遍历 classifications 构造并行任务 |
| `agent_node` | state | 追加 AgentOutput | explore→HybridRetriever，weather→占位 |
| `build_router_graph` | - | CompiledStateGraph | 编译 worklow |

## LangGraph 流程

```python
graph.add_sequence([classifier_node, route_to_agents, agent_node])
```

`route_to_agents` 返回 `Send("agent_node", ...)` 列表，LangGraph 自动并行执行多个 `agent_node`，结果通过 `add` reducer 累加到 `agent_results`。

## 关键依赖

- `langgraph.graph.StateGraph`, `Send`
- `ChatTongyi` (qwen-turbo, 与 reranker 保持一致)
- `HybridRetriever` (已有 RAG 检索器)
- `settings`, `app_logger`

## 未决事项

- `weather_agent` 当前占位，后续接入高德天气 API
- Agent 节点未来可拆分到 `subagents/` 目录

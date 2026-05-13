# 工具返回口径统一设计

## 概述

统一 `app/tools/` 下所有工具的返回模式：路由决策从工具收回到图定义中，消除 `Command(goto="agent")` 绕过 guard 节点的问题。

## 当前问题

| 工具类型 | 返回 | 流程 |
|----------|------|------|
| 状态转换 (15) | `Command(update={...}, goto="agent")` | 跳过 guard |
| 终端 (1) | `Command(update={...}, goto="__end__")` | 正常 |
| 查询/计算 (8) | `str` | 走边 `tools → guard → agent` |

状态转换工具的 `goto="agent"` 绕过了 guard 节点，导致上下文压缩在步骤转换后不生效。

## 目标

- 所有工具统一经由图边控制流程
- 工具只负责状态更新，不负责路由
- guard 节点对每个工具返回都执行

## 改动

**文件**: `app/tools/state_transition.py`

**操作**: 15 个工具删除 `, goto="agent"`：

| 工具 | 改动 |
|------|------|
| `record_requirement_tool` | `Command(update={...})` |
| `select_destination_tool` | 同上 |
| `select_transport_tool` | 同上 |
| `select_accommodation_tool` | 同上 |
| `select_food_tool` | 同上 |
| `generate_itinerary_tool` | 同上 |
| `summarize_budget_tool` | 同上 |
| `go_back_to_step` | 同上 |
| 7 个 `go_back_to_*` 快捷工具 | 委托 `go_back_to_step`，自动生效 |

`generate_order_tool` 不动——保留 `goto="__end__"` 终止语义。

**不动的文件**: `graph.py`、所有查询工具、`__init__.py`、`step_config.py`、测试。

## 前后对比

```
之前:
  select_destination → Command(update, goto="agent") → agent (跳过 guard)

之后:
  select_destination → Command(update) → 走边 → guard → agent ✓
```

## 测试

现有测试无需修改——工具行为不变（仍更新 state + 推进步骤），只是路由路径从 `goto="agent"` 变为走图边，最终仍到达 agent。运行全量测试确认无回归。

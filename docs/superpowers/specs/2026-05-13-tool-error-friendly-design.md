# 工具调用错误友好化设计

## 概述

当 LLM 调用状态转换工具参数不完整时，将 Pydantic 验证错误包装为引导性提示，让 LLM 自然向用户补充提问，而非暴露原始报错。

## 问题

```
LLM 调 record_requirement_tool(budget_level=None, destination="")
  → @tool Pydantic 校验失败
  → ToolMessage 含原始 "Error invoking tool..."  + Pydantic ValidationError
  → 用户看到 "抱歉，系统出错了"
```

## 改动

### 1. graph.py：ToolNode 加错误包装器

新增 `_wrap_tool_error` 函数，传入 `ToolNode(handle_tool_errors=...)`：

```python
def _wrap_tool_error(error: Exception) -> str:
    msg = str(error)
    if "Input should be" in msg or "validation error" in msg.lower():
        return (
            f"参数校验未通过：\n{msg}\n\n"
            f"请向用户逐一确认上述信息，补充完整后重新调用。"
        )
    return f"操作未能完成：{msg[:300]}。请向用户说明并询问如何处理。"
```

### 2. state.py：放宽 UserRequirement 字段约束

| 字段 | 当前类型 | 修复类型 |
|------|----------|----------|
| `budget_level` | `BudgetLevel` (必填) | `Optional[BudgetLevel]` |
| `travel_styles` | `List[TravelStyle]` (必填) | `Optional[List[TravelStyle]]` |

### 3. state_transition.py：record_requirement_tool 防御

- `travel_styles` 为 None/空时 → 设为 `[]`
- `special_needs` 为空字符串时 → 设为 `None`
- `destination` 为空时 → 不推进，返回提示 ToolMessage 让 LLM 询问用户

### 不改的

- 其他 7 个状态转换工具：参数校验失败同样被 `_wrap_tool_error` 包装
- 查询工具（返回 str 的）：不受影响

## 测试

- mock ToolNode 调用含无效参数的 tool，验证返回 friendly 消息
- 验证 budget_level=None 不再报错
- 验证 destination="" 时工具拒绝推进

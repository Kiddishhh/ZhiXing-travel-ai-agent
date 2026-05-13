# 用户确认门控 + 工具修复设计

## 问题

1. LLM 不等用户确认就自主连续调用状态过渡工具跳步
2. `query_destination_info` 的 `with_structured_output` 报 `json_object` 验证错误
3. `query_accommodation` MCP 连接断开

## 改动

### 1. step_config.py — 规则块 + 工具分层

8 个步骤 prompt 全部改造：

**顶部加硬性规则块**：
```
## ⚠️ 关键规则（必须遵守）
- 查询工具：随时可用，帮用户获取信息
- 🔒 确认工具：仅在用户明确说出"可以""好的""就这个""确认""行""没问题"后才能调用
- 如果用户还在提问或补充需求 → 只回复文字，不调确认工具
- 禁止在一条回复中连续调用多个确认工具
- 回退工具仅在用户主动提出修改时使用
```

**工具分三类展示**：查询工具 / 🔒确认工具 / ↩️回退工具

### 2. state_transition.py — ToolMessage 刹车信号

7 个过渡工具的 ToolMessage 从陈述句改为引导句，含"当前阶段 → 下一步。请向用户..."结构。

### 3. destination_router.py — method="function_calling"

```python
structured_llm = llm.with_structured_output(
    ClassificationResult, method="function_calling"
)
```

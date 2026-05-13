# 上下文压缩 v2 设计

## 概述

修复去掉 `goto="agent"` 后 guard 介入导致的消息顺序错乱问题，建立三层消息隔离架构，优化压缩参数。

## 当前问题

1. **消息顺序错误**：`context_summary` → `step prompt` → `conversation`，历史摘要压制步骤指令
2. **死代码**：guard 过滤 SystemMessage，但 state["messages"] 已无 SystemMessage
3. **参数偏小**：压缩阈值 8000 太低，摘要 500 字信息不足

## 改动范围

仅 `app/agents/handoffs/graph.py` 一个文件。

## 改动内容

### 1. 参数调整

| 参数 | 旧值 | 新值 |
|------|------|------|
| `COMPRESSION_MAX_TOKENS` | 8000 | 12000 |
| 摘要字数上限 (prompt 中) | 500 字 | 800 字 |

### 2. agent_node：三层消息隔离

```
第1层: SystemMessage(步骤 prompt)       ← 指令层，优先级最高
第2层: SystemMessage("[已收集的旅行信息]") ← 记忆层，仅事实数据
第3层: state["messages"]                ← 对话层，交互历史
```

步骤 prompt 放在最前面确保指令优先；摘要标签从 `[对话历史摘要]` 改为 `[已收集的旅行信息]`。

### 3. guard_node：移除 SystemMessage 分离

`state["messages"]` 已不含 SystemMessage（步骤 prompt 由 agent_node 临时注入），删除消息类型判断逻辑，直接按位置切片。

### 4. 压缩 Prompt 优化

在 `COMPRESSION_SYSTEM_PROMPT` 中新增约束：
- 禁止描述 AI 做过什么、问过什么
- 只提取事实数据（日期/目的地/预算/已选选项）和工具返回的具体数据

## 测试

更新 `test_context_compression.py`：
- `test_all_system_messages_no_compression` 删除（SystemMessage 不再出现在 messages 中）
- `test_preserves_system_messages` 删除（同上）
- 其余测试适配新的 keep_recent 逻辑

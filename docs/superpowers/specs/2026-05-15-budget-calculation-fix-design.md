# 预算计算优化设计方案

## 概述

修复 `budget_tools.py` 中两个预算计算缺陷：人均预算与总价口径不一致、多人多间房未计入。

## 问题 1：人均预算 vs 总预算口径不一致

**根因：** step_config 需求收集写 `元/人`，用户按人均输入（如 5000/人），但 `calculate_budget`
求出的 `grand_total` 是总价（如 2 人共 7400），直接对比 `budget_max=5000` → 误判超支。

**修复：** 预算上限乘以人数，统一为总价口径：

```python
total_budget_limit = budget_limit * traveler_count
```

超支判断改为对比 `grand_total > total_budget_limit`，输出文案展示计算过程。

## 问题 2：多人多间房未计入

**根因：** 住宿费只算 `price_per_night × nights`，默认 1 间房。多人出行可能需要多间。

**修复：** 给 `calculate_budget` 加可选参数 `rooms_needed: int = 1`：

```python
acc_total = price_per_night * nights * rooms_needed
```

LLM 在调用时根据上下文灵活判断房间数，不写死公式。

## 涉及文件

| 文件 | 动作 | 说明 |
|------|------|------|
| `app/tools/budget_tools.py` | 修改 | 两处计算逻辑修复 |
| `tests/tools/test_budget_and_order.py` | 修改 | 验证新的计算逻辑 |

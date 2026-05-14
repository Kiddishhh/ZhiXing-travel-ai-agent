# 目的地推荐反馈流失 + 预算计算修复 设计

## 目标

1. 修复目的地推荐阶段：RAG 结果返回后 LLM 跳过用户反馈直接推进状态
2. 修复预算计算：多方案时取平均值而非累加全部价格

---

## 一、query_destination_info 追加制动标记

### 1.1 问题

`query_destination_info` 返回 Router 的综合报告后，LLM 有时在同一轮直接调用 `select_destination_tool`，用户未看到 RAG 结果

### 1.2 修复

**文件:** `app/tools/router_query.py:41`

```python
# 旧
return result["final_report"]

# 新
brake = (
    "\n\n---\n"
    "⚠️ 请将以上目的地信息整理后用简洁的语言向用户展示（每个目的地 2-3 句话），"
    "列出推荐理由后等待用户选择。用户明确确认目的地之后，再调用 select_destination_tool。"
)
return result["final_report"] + brake
```

### 1.3 原理

制动标记直接追加在 ToolMessage content 末尾，是 LLM 在当前轮次最后读取的内容，命中率最高。不改图结构、不改 prompt、不改路由。

---

## 二、预算计算改为平均值

### 2.1 问题

`calculate_budget` 中交通/住宿/餐饮三项直接将所有方案价格累加。当 LLM 传入多方案时（如 3 个航班 ¥800/¥1200/¥1500），总计错误地变成 ¥3500

### 2.2 修复

**文件:** `app/tools/budget_tools.py:40-72`

每一项从"累加全部"改为"平均值"：

**交通费 (第 43-47 行):**
```python
# 旧
transport_total = 0.0
for t in transport_options:
    price = t.get("price", 0)
    transport_total += price

# 新
if transport_options:
    prices = [t.get("price", 0) for t in transport_options if t.get("price")]
    transport_total = round(sum(prices) / len(prices), 2) if prices else 0.0
else:
    transport_total = 0.0
```

**住宿费 (第 53-60 行):**
```python
# 旧
for a in accommodation_options:
    ...
    accommodation_total += acc_total

# 新
if accommodation_options:
    totals = []
    for a in accommodation_options:
        ...
        totals.append(acc_total)
    accommodation_total = round(sum(totals) / len(totals), 2)
else:
    accommodation_total = 0.0
```

**餐饮费 (第 66-72 行):**
```python
# 旧
for f in food_options:
    ...
    food_total += f_total

# 新
if food_options:
    totals = []
    for f in food_options:
        ...
        totals.append(f_total)
    food_total = round(sum(totals) / len(totals), 2)
else:
    food_total = 0.0
```

**输出标注:** 汇总行标注"（均价）"
```python
f"> 交通小计 (均价): ¥{transport_total}"
f"> 住宿小计 (均价): ¥{accommodation_total}"
f"> 餐饮小计 (均价): ¥{food_total}"
```

---

## 三、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/tools/router_query.py` | 修改第 41 行 | 追加制动标记 |
| `app/tools/budget_tools.py` | 修改第 40-72 行 + 第 96-108 行 | 累加→平均值 + 标注 |

---

## 四、测试验证

```bash
# 纯逻辑测试
python -m pytest tests/tools/ -v -s

# 交互式测试
python tests/interactive/interactive_destination.py
```

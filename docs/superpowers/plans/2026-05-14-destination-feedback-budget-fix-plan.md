# 目的地反馈制动 + 预算平均化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复两个 bug：目的地推荐后 LLM 跳过用户反馈直接推进状态，以及预算计算将多方案价格累加而非取平均值

**Architecture:** 2 文件修改 — `router_query.py` 在返回末尾追加制动标记，`budget_tools.py` 将交通/住宿/餐饮改为平均值并标注

**Tech Stack:** Python 3.11+, LangChain tools

---

### Task 1: 追加制动标记到 query_destination_info

**Files:**
- Modify: `app/tools/router_query.py`

- [ ] **Step 1: 修改 `query_destination_info` 的 return 语句**

Read the file, find line 41:
```python
    return result["final_report"]
```

Replace with:
```python
    brake = (
        "\n\n---\n"
        "⚠️ 请将以上目的地信息整理后用简洁的语言向用户展示（每个目的地 2-3 句话），"
        "列出推荐理由后等待用户选择。用户明确确认目的地之后，再调用 select_destination_tool。"
    )
    return result["final_report"] + brake
```

- [ ] **Step 2: 验证语法并运行测试**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/tools/router_query.py', encoding='utf-8').read()); print('[OK] 语法正确')"
python -m pytest tests/tools/ tests/agents/ -v
```

- [ ] **Step 3: Commit**

```bash
git add app/tools/router_query.py
git commit -m "fix(router): add brake marker to query_destination_info to prevent state skip"
```

---

### Task 2: 预算计算改为平均值

**Files:**
- Modify: `app/tools/budget_tools.py`

- [ ] **Step 1: 修改交通费计算（第 40-47 行）**

Read the file, find:
```python
    # ── 1. 交通费 ──
    transport_total = 0.0
    transport_detail = []
    transport_options = state.get("transport_options", []) or []
    for t in transport_options:
        price = t.get("price", 0)
        transport_total += price
        transport_detail.append(f"  {t.get('details', '未知交通')}: ¥{price}")
```

Replace with:
```python
    # ── 1. 交通费 ──
    transport_total = 0.0
    transport_detail = []
    transport_options = state.get("transport_options", []) or []
    for t in transport_options:
        price = t.get("price", 0)
        transport_detail.append(f"  {t.get('details', '未知交通')}: ¥{price}")
    if transport_options:
        prices = [t.get("price", 0) for t in transport_options if t.get("price")]
        transport_total = round(sum(prices) / len(prices), 2) if prices else 0.0
```

- [ ] **Step 2: 修改住宿费计算（第 49-60 行）**

Find:
```python
    # ── 2. 住宿费 ──
    accommodation_total = 0.0
    accommodation_detail = []
    accommodation_options = state.get("accommodation_options", []) or []
    nights = max(travel_days - 1, 1)
    for a in accommodation_options:
        price_per_night = a.get("price_per_night", 0)
        acc_total = price_per_night * nights
        accommodation_total += acc_total
        accommodation_detail.append(
            f"  {a.get('name', '未知住宿')}: ¥{price_per_night}/晚 × {nights}晚 = ¥{acc_total}"
        )
```

Replace with:
```python
    # ── 2. 住宿费 ──
    accommodation_total = 0.0
    accommodation_detail = []
    accommodation_options = state.get("accommodation_options", []) or []
    nights = max(travel_days - 1, 1)
    for a in accommodation_options:
        price_per_night = a.get("price_per_night", 0)
        acc_total = price_per_night * nights
        accommodation_detail.append(
            f"  {a.get('name', '未知住宿')}: ¥{price_per_night}/晚 × {nights}晚 = ¥{acc_total}"
        )
    if accommodation_options:
        totals = [
            a.get("price_per_night", 0) * nights
            for a in accommodation_options if a.get("price_per_night")
        ]
        accommodation_total = round(sum(totals) / len(totals), 2) if totals else 0.0
```

- [ ] **Step 3: 修改餐饮费计算（第 62-72 行）**

Find:
```python
    # ── 3. 餐饮费 ──
    food_total = 0.0
    food_detail = []
    food_options = state.get("food_options", []) or []
    for f in food_options:
        daily = f.get("estimated_daily_cost", 0)
        f_total = daily * travel_days
        food_total += f_total
        food_detail.append(
            f"  {f.get('type', '未知餐饮')}: ¥{daily}/天 × {travel_days}天 = ¥{f_total}"
        )
```

Replace with:
```python
    # ── 3. 餐饮费 ──
    food_total = 0.0
    food_detail = []
    food_options = state.get("food_options", []) or []
    for f in food_options:
        daily = f.get("estimated_daily_cost", 0)
        f_total = daily * travel_days
        food_detail.append(
            f"  {f.get('type', '未知餐饮')}: ¥{daily}/天 × {travel_days}天 = ¥{f_total}"
        )
    if food_options:
        totals = [
            f.get("estimated_daily_cost", 0) * travel_days
            for f in food_options if f.get("estimated_daily_cost")
        ]
        food_total = round(sum(totals) / len(totals), 2) if totals else 0.0
```

- [ ] **Step 4: 修改汇总行标注（第 96、100、104 行附近）**

Find the three summary lines and add "（均价）"：

```python
# 旧
f"> 交通小计: ¥{transport_total}"
f"> 住宿小计: ¥{accommodation_total}"
f"> 餐饮小计: ¥{food_total}"

# 新
f"> 交通小计 (均价): ¥{transport_total}"
f"> 住宿小计 (均价): ¥{accommodation_total}"
f"> 餐饮小计 (均价): ¥{food_total}"
```

- [ ] **Step 5: 验证语法并运行测试**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/tools/budget_tools.py', encoding='utf-8').read()); print('[OK] 语法正确')"
python -m pytest tests/tools/ -v
```

- [ ] **Step 6: Commit**

```bash
git add app/tools/budget_tools.py
git commit -m "fix(budget): calculate average instead of sum for multi-option transport/accommodation/food"
```

---

### Task 3: 全量验证

- [ ] **Step 1: 运行全部测试**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -m pytest tests/rag/ tests/tools/ tests/agents/ -v
```

预期：75 passed

- [ ] **Step 2: 语法全量检查**

```bash
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in __import__('pathlib').Path('.').rglob('*.py') if 'venv' not in str(p)]; print('[OK]')"
```

---

## Self-Review Results

1. **Spec coverage**: Task 1 → 制动标记，Task 2 → 预算平均值，Task 3 → 验证 ✅
2. **Placeholder scan**: No TBD/TODO ✅
3. **Type consistency**: N/A (no cross-task type dependencies) ✅

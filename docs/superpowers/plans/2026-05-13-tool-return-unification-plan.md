# 工具返回口径统一 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans.

**Goal:** 删除 `state_transition.py` 中 8 处 `, goto="agent"`，让所有状态转换工具走图边统一路由。

**Architecture:** `replace_all` 一次替换 `}, goto="agent")` → `})`，覆盖全部 8 处。`generate_order_tool` 的 `goto="__end__"` 不动。

---

### Task 1: 删除 goto="agent" + 测试验证 + Commit

**Files:**
- Modify: `app/tools/state_transition.py`

- [ ] **Step 1: 删除 8 处 `, goto="agent"`**

在 `app/tools/state_transition.py` 中，用 `replace_all` 批量替换：

```
老: }, goto="agent")
新: })
```

注意：`generate_order_tool` 用的是 `goto="__end__"`，不会被此替换影响。

- [ ] **Step 2: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/tools/state_transition.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: 验证 regenerate_order_tool 的 goto="__end__" 未受影响**

```bash
grep 'goto=' app/tools/state_transition.py
```

Expected: 仅剩一行 `goto="__end__"`（`generate_order_tool`）

- [ ] **Step 4: 运行全量测试**

```bash
python -m pytest tests/ -v --ignore=tests/test_api --ignore=scripts
```

Expected: 0 failures（`test_driving_route` 可能有预先存在的网络问题）

- [ ] **Step 5: Commit**

```bash
git add app/tools/state_transition.py docs/superpowers/specs/2026-05-13-tool-return-unification-design.md docs/superpowers/plans/2026-05-13-tool-return-unification-plan.md
git commit -m "$(cat <<'EOF'
fix: remove goto="agent" from state transition tools, let graph edges control routing

All tools now follow tools → guard → agent edge, ensuring context
compression guard runs after every tool call. Only generate_order_tool
retains goto="__end__" for graph termination.
EOF
)"
```

# Scripts & Tests 目录整理设计

## 目标

清理 `scripts/` 和 `tests/` 目录，消除过期和冗余文件，将初始化脚本与测试脚本分离，统一测试输出风格。

## 核心原则

1. `scripts/` 只放初始化脚本（init），不放任何测试
2. `tests/` 按模块分层：纯逻辑 pytest 测试 + 交互式脚本
3. 纯逻辑测试（mock）保持 pytest 风格，增强 `[N/M]` 阶段 print 输出
4. 涉及 LLM/外部 API 的测试做成交互式脚本，用 `input()` 让用户输入参数
5. 消除重复：合并功能重叠的文件

---

## 一、最终目录结构

### `scripts/` — 仅初始化（2 个文件）

```
scripts/
├── init_db.py          # 数据库初始化 (Checkpointer + Store 表)
└── init_rag.py         # RAG 系统初始化 (文档→切分→BM25+ChromaDB)
```

### `tests/` — 模块 + 交互式分层（19 个文件）

```
tests/
├── rag/                          # RAG 纯逻辑测试 (pytest + mock)
│   ├── test_query_optimizer.py   # 查询优化器
│   ├── test_text_splitter.py     # 文档切分
│   ├── test_reranker.py          # 重排序
│   ├── test_retriever.py         # 混合检索 (BM25+RRF)
│   └── test_pipeline.py          # RAG 管线流程
│
├── tools/                        # 工具纯逻辑测试 (pytest + mock)
│   ├── test_food_tools.py        # 餐饮工具函数单元测试 (geocode/POI/Tavily)
│   ├── test_budget_and_order.py  # 预算计算 + 订单生成
│   └── test_tools_validation.py  # 工具注册表/签名/State结构/导入链综合验证
│
├── agents/                       # Agent 纯逻辑测试 (pytest + mock)
│   └── test_context_compression.py  # 上下文压缩 guard 节点
│
└── interactive/                  # 交互式脚本 (需真实 LLM/MCP/API)
    ├── interactive_llm.py        # LLM 连接测试
    ├── interactive_rag.py        # 完整 RAG 管道 (合并3个重复脚本)
    ├── interactive_flow.py       # 主流程对话 (handoffs graph)
    ├── interactive_destination.py # 目的地 Router (RAG+天气)
    ├── interactive_mcp.py        # MCP 工具列表查看
    ├── interactive_weather.py    # 天气查询
    ├── interactive_search.py     # 搜索查询
    ├── interactive_transport.py  # 交通子代理 (航班/高铁/自驾)
    ├── interactive_accommodation.py # 住宿查询
    └── interactive_food.py       # 餐饮查询 (Amap+Tavily)
```

---

## 二、交互式脚本统一模板

所有 `tests/interactive/` 脚本遵循统一模式：

```python
"""
交互式 XXX 测试
运行: python tests/interactive/interactive_xxx.py
"""
import asyncio, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def print_stage(stage: str, total: int, current: int):
    """统一的分阶段打印"""
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  测试名称")
    print("=" * 60)

    # [1/N] 初始化
    print_stage("初始化服务", N, 1)
    try:
        # ... 初始化代码 ...
        print("[OK] 初始化完成")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {type(e).__name__}: {e}")
        return

    # [2/N] 用户输入
    print_stage("输入测试参数", N, 2)
    user_input = input("请输入查询内容: ").strip()
    print(f"[输入] 收到: {user_input}")

    # [3/N] 执行并打印结果
    print_stage("执行测试", N, 3)
    try:
        result = await some_service(user_input)
        print(f"[结果]\n{result}")
        print("[OK] 测试通过")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")

    print("\n测试结束")


if __name__ == "__main__":
    asyncio.run(main())
```

### 特征

- `[N/M]` 分阶段标注（与项目现有 `[1/4]` `[2/4]` 风格一致）
- 每个阶段 `try/except` 包裹，打印 `[ERROR]` + 完整异常类型和消息
- 成功步骤标记 `[OK]`
- 涉及 LLM/外部服务的参数使用 `input()` 交互
- 格式化分隔线（60 个 `=`）

---

## 三、文件变更清单

### 保留并增强（加 print 输出）

| 文件 | 变更 |
|------|------|
| `tests/rag/test_query_optimizer.py` | 增加 `[1/N]` 阶段标注 print，fail case 打印 `[ERROR]` |
| `tests/rag/test_text_splitter.py` | 已有完整 print ✅ |
| `tests/rag/test_reranker.py` | 增加阶段标注 print |
| `tests/rag/test_retriever.py` | 增加权重/RRF 融合过程 print |
| `tests/rag/test_pipeline.py` | 已有完整 print ✅ |
| `tests/tools/test_food_tools.py` | 增加阶段标注 print |
| `tests/tools/test_budget_and_order.py` | 增加阶段标注 print |
| `tests/tools/test_tools_validation.py` | 已有完整流程 ✅ |
| `tests/agents/test_context_compression.py` | 增加压缩前后对比 print |
| `scripts/init_db.py` | 不变 |
| `scripts/init_rag.py` | 不变 |

### 新增（交互式脚本）

| 文件 | 来源合并 |
|------|---------|
| `tests/interactive/interactive_llm.py` | `scripts/test_llm.py` |
| `tests/interactive/interactive_rag.py` | `scripts/test_rag.py` + `scripts/test_rag_pipeline.py` + `tests/test_rag/test_full_pipeline.py` |
| `tests/interactive/interactive_flow.py` | `tests/handoffs_flow_test.py` |
| `tests/interactive/interactive_destination.py` | `tests/test_agents/test_destination_router.py` |
| `tests/interactive/interactive_mcp.py` | `tests/test_mcp/test_client.py` |
| `tests/interactive/interactive_weather.py` | `tests/test_mcp/test_weather_server.py` |
| `tests/interactive/interactive_search.py` | `tests/test_mcp/test_search_mcp.py` |
| `tests/interactive/interactive_transport.py` | `tests/test_mcp/test_transport_subagents.py` |
| `tests/interactive/interactive_accommodation.py` | `tests/test_mcp/test_accommodation.py` |
| `tests/interactive/interactive_food.py` | `tests/test_mcp/test_food.py` + `tests/test_api/test_food_tool.py` |

### 删除

```
scripts/test_llm.py
scripts/test_rag.py
scripts/test_rag_pipeline.py
tests/handoffs_flow_test.py
tests/test_rag/test_full_pipeline.py
tests/test_agents/test_destination_router.py
tests/test_mcp/test_client.py
tests/test_mcp/test_weather_server.py
tests/test_mcp/test_search_mcp.py
tests/test_mcp/test_transport_subagents.py
tests/test_mcp/test_accommodation.py
tests/test_mcp/test_food.py
tests/test_api/test_food_tool.py
tests/test_api/__init__.py
```

### 目录变化

- 重命名：`tests/test_rag/` → `tests/rag/`
- 重命名：`tests/test_tools/` → `tests/tools/`
- 重命名：`tests/test_agents/` → `tests/agents/`
- 删除：`tests/test_api/`（空目录）
- 删除：`tests/test_mcp/`（均迁移到 `tests/interactive/`）
- 新增：`tests/interactive/`

---

## 四、交互式脚本关键交互点

| 脚本 | `input()` 提示 |
|------|---------------|
| `interactive_llm.py` | 输入测试消息（可多次输入） |
| `interactive_rag.py` | 输入查询词，打印检索+重排序全流程 |
| `interactive_flow.py` | 已有的持续对话式交互（保持） |
| `interactive_destination.py` | 输入目的地 + 查询类型 |
| `interactive_mcp.py` | 选择要连接的 MCP 服务 |
| `interactive_weather.py` | 输入城市 adcode |
| `interactive_search.py` | 输入搜索关键词 |
| `interactive_transport.py` | 输入出发地/目的地/日期 |
| `interactive_accommodation.py` | 输入目的地/入住日期/住宿类型 |
| `interactive_food.py` | 输入目的地/餐饮类型 |

---

## 五、运行方式

```bash
# 初始化
python scripts/init_db.py
python scripts/init_rag.py

# 纯逻辑测试（pytest + mock，无需外部服务）
python -m pytest tests/rag/ -v -s
python -m pytest tests/tools/ -v -s
python -m pytest tests/agents/ -v -s

# 交互式测试（需外部 LLM/MCP/API）
python tests/interactive/interactive_llm.py
python tests/interactive/interactive_rag.py
python tests/interactive/interactive_flow.py
python tests/interactive/interactive_destination.py
python tests/interactive/interactive_mcp.py
python tests/interactive/interactive_weather.py
python tests/interactive/interactive_search.py
python tests/interactive/interactive_transport.py
python tests/interactive/interactive_accommodation.py
python tests/interactive/interactive_food.py

# 全部单元测试
python -m pytest tests/rag/ tests/tools/ tests/agents/ -v -s
```

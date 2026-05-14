# RAG 管道集成到 Explore Agent + ChromaDB 精简 设计

## 目标

1. 将 `destination_router.py` 的 `_explore_agent` 从直连 ChromaDB 改为使用完整 RAG 管道
2. 精简 `chroma_client.py` 中的冗余接口

---

## 一、`_explore_agent` 改造

### 1.1 现状

`_explore_agent` 在每次调用时新建 `ChromaManager()`，直接调用 `similarity_search_with_score(query, k=5)`，返回原始片段。绕过了整个 RAG 管道。

### 1.2 改造方案

引入模块级懒加载单例 `_rag_pipeline`，首次访问时初始化，后续调用复用。

```python
# 新增导入
from typing import Optional
from app.rag.pipeline import RAGPipeline
from app.rag.query_optimizer import QueryOptimizer
from app.rag.retriever import HybridRetriever
from app.rag.reranker import LLMReranker
from app.rag.text_splitter import ParentDocumentSplitter
from app.rag.document_loader import DocumentManager
from app.core.ChromaDB.chroma_client import ChromaManager

_rag_pipeline: Optional[RAGPipeline] = None

def _get_rag_pipeline() -> RAGPipeline:
    """懒加载初始化 RAG 管线"""
    global _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline

    doc_manager = DocumentManager()
    documents = doc_manager.load_all_documents()
    splitter = ParentDocumentSplitter(
        parent_chunk_size=1000, parent_chunk_overlap=200,
        child_chunk_size=200, child_chunk_overlap=50,
    )
    _, child_docs = splitter.split_documents(documents)

    chroma_manager = ChromaManager()
    chroma_manager.delete_collection("travel_children")
    retriever = HybridRetriever(
        chroma_manager=chroma_manager,
        collection_name="travel_children",
    )
    retriever.initialize(child_docs)

    _rag_pipeline = RAGPipeline(
        optimizer=QueryOptimizer(),
        retriever=retriever,
        parent_splitter=splitter,
        reranker=LLMReranker(top_k=5),
    )
    return _rag_pipeline
```

### 1.3 改造后的 `_explore_agent`

```python
def _explore_agent(query: str) -> str:
    """探索 Agent：通过 RAG 管道检索景点攻略"""
    try:
        pipeline = _get_rag_pipeline()
        result = pipeline.run(query)

        if not result.final_docs:
            return f"未找到与「{query}」相关的攻略信息。"

        lines = [f"## 相关攻略 ({len(result.final_docs)} 条)\n"]
        for i, doc in enumerate(result.final_docs, 1):
            score = doc.metadata.get("relevance_score", "N/A")
            source = doc.metadata.get("source", "未知来源")
            snippet = doc.page_content[:200].replace("\n", " ")
            lines.append(f"{i}. [{source}] (相关度:{score}) {snippet}...")
        return "\n\n".join(lines)
    except Exception as e:
        app_logger.error(f"探索 Agent 检索失败: {e}")
        return f"攻略检索异常: {e}"
```

### 1.4 关键变化

| 维度 | 旧 | 新 |
|------|-----|-----|
| 检索方式 | ChromaDB cosine similarity | BM25 + Dense RRF 融合 |
| 查询处理 | 直接传入 | QueryOptimizer 优化 |
| 上下文 | 子文档片段（200字） | 父文档完整块 |
| 排序 | distance 排序 | LLM 重排序 (relevance_score) |
| LLM 依赖 | 无 | QueryOptimizer + LLMReranker（各有 LLM 调用） |
| 初始化 | 每次调用新建 ChromaManager | 首次懒加载，后续复用 |
| 函数签名 | 不变 | 不变 |

---

## 二、`chroma_client.py` 精简

### 2.1 删除

- `client` @property（第 36-39 行）— 无外部调用者
- `embedding_function` @property（第 42-53 行）— 改为私有方法 `_get_embedding_function`
- `_vectorstores: Dict[str, Chroma]` 缓存字典（第 33 行）— 仅一个集合 `travel_children`，缓存不必要

### 2.2 新增/改为私有

- `_get_embedding_function()` — 原 `embedding_function` property 逻辑，仅内部使用
- `_get_vectorstore(collection_name)` — 从 public 改为 private（原 `get_vectorstore`），直接创建返回，去掉缓存

### 2.3 保留（接口不变）

| 方法 | 签名 | 说明 |
|------|------|------|
| `add_documents` | `(documents, ids, collection_name)` | 内部调用 `_get_vectorstore` |
| `similarity_search_with_score` | `(query, k, collection_name)` | 内部调用 `_get_vectorstore` |
| `delete_collection` | `(collection_name)` | 直接操作 `self._client` |

### 2.4 影响面

- `HybridRetriever._build_vector_store` → 调用 `chroma_manager.add_documents()` — **不变**
- `HybridRetriever._dense_search` → 调用 `chroma_manager.similarity_search_with_score()` — **不变**
- `init_rag.py` → `ChromaManager()` 创建、`delete_collection()` — **不变**
- 不再被直接调用的 `get_vectorstore()` 改为私有，外部无调用者

---

## 三、文件变更清单

| 文件 | 操作 |
|------|------|
| `app/agents/routers/destination_router.py` | 修改：`_explore_agent` 改用 RAGPipeline；新增 `_get_rag_pipeline` 懒加载；增加 7 个导入 |
| `app/core/ChromaDB/chroma_client.py` | 修改：移除 `client`/`embedding_function` property；`get_vectorstore` → `_get_vectorstore`；去掉 `_vectorstores` 缓存 |

---

## 四、测试验证

```bash
# 纯逻辑测试（无外部依赖）
python -m pytest tests/rag/ tests/tools/ tests/agents/ -v -s

# 交互式测试（验证 RAG 管道在 explore agent 中的行为）
python tests/interactive/interactive_destination.py
```

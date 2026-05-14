# RAG 管道集成 + ChromaDB 精简 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `_explore_agent` 从直连 ChromaDB 改为使用完整 RAG 管道，同时精简 `chroma_client.py` 冗余接口

**Architecture:** 2 文件修改 — `chroma_client.py` 移除未使用的 property 和缓存，`destination_router.py` 新增模块级懒加载 RAGPipeline 单例并重写 `_explore_agent`

**Tech Stack:** Python 3.11+, langchain, ChromaDB, RAG pipeline (QueryOptimizer + HybridRetriever + LLMReranker)

---

### Task 1: 精简 `chroma_client.py`

**Files:**
- Modify: `app/core/ChromaDB/chroma_client.py`

- [ ] **Step 1: 删除 `client` property（第 36-39 行）**

找到并删除：
```python
    @property
    def client(self) -> PersistentClient:
        """获取 ChromaDB 持久化客户端"""
        return self._client
```

- [ ] **Step 2: 删除 `embedding_function` property（第 42-53 行），新增私有方法 `_get_embedding_function`**

删除：
```python
    @property
    def embedding_function(self) -> DashScopeEmbeddings:
        """延迟初始化的嵌入函数

        使用 DashScope text-embedding-v2 API 进行文本向量化。
        需要环境变量 DASHSCOPE_API_KEY 已配置。
        """
        if self._embedding_function is None:
            self._embedding_function = DashScopeEmbeddings(
                model="text-embedding-v2",
            )
            app_logger.info("DashScope Embedding 已初始化 (model=text-embedding-v2)")
        return self._embedding_function
```

在 `__init__` 之后新增：
```python
    def _get_embedding_function(self) -> DashScopeEmbeddings:
        """私有：延迟初始化嵌入模型"""
        if self._embedding_fn is None:
            self._embedding_fn = DashScopeEmbeddings(
                model="text-embedding-v2",
            )
            app_logger.info("DashScope Embedding 已初始化 (model=text-embedding-v2)")
        return self._embedding_fn
```

- [ ] **Step 3: 修改 `__init__` — 属性名从 `_embedding_function` 改为 `_embedding_fn`，删除 `_vectorstores` 缓存**

将第 29 行：
```python
        self._embedding_function: Optional[DashScopeEmbeddings] = None
```
改为：
```python
        self._embedding_fn: Optional[DashScopeEmbeddings] = None
```

删除第 33 行：
```python
        self._vectorstores: Dict[str, Chroma] = {}
```

- [ ] **Step 4: 将 `get_vectorstore` 改为 `_get_vectorstore`，去掉缓存逻辑**

删除原方法（第 55-65 行），替换为：
```python
    def _get_vectorstore(
        self, collection_name: str = "travel_children"
    ) -> Chroma:
        """私有：创建 Chroma 向量存储实例"""
        return Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self._get_embedding_function(),
        )
```

- [ ] **Step 5: 更新内部调用 — `add_documents` 和 `similarity_search_with_score` 中**

将 `self.get_vectorstore(collection_name)` 改为 `self._get_vectorstore(collection_name)`

- [ ] **Step 6: 清理不再使用的导入**

删除第 6 行（`Dict` 不再需要）：
```python
from typing import Dict, List, Optional, Tuple
```
改为：
```python
from typing import List, Optional, Tuple
```

- [ ] **Step 7: 验证语法并运行 RAG 测试**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/core/ChromaDB/chroma_client.py', encoding='utf-8').read()); print('[OK] 语法正确')"
python -m pytest tests/rag/ -v
```

- [ ] **Step 8: Commit**

```bash
git add app/core/ChromaDB/chroma_client.py
git commit -m "refactor(chroma): remove unused client/embedding_function properties, inline vectorstore cache"
```

---

### Task 2: 集成 RAG 管道到 `_explore_agent`

**Files:**
- Modify: `app/agents/routers/destination_router.py`

- [ ] **Step 1: 新增 7 个导入**

在第 9 行（`from pydantic import BaseModel, Field` 之后，空行前）插入：
```python
from typing import Optional

from app.rag.pipeline import RAGPipeline
from app.rag.query_optimizer import QueryOptimizer
from app.rag.retriever import HybridRetriever
from app.rag.reranker import LLMReranker
from app.rag.text_splitter import ParentDocumentSplitter
from app.rag.document_loader import DocumentManager
from app.core.ChromaDB.chroma_client import ChromaManager
```

- [ ] **Step 2: 在 `classifier_node` 之前新增 `_get_rag_pipeline` 懒加载函数**

在第 72 行（`def classifier_node` 之前）插入：
```python

# ── RAG 管线（懒加载单例）────────────────────────────────

_rag_pipeline: Optional[RAGPipeline] = None


def _get_rag_pipeline() -> RAGPipeline:
    """懒加载初始化 RAG 管线 — 首次调用时加载文档+构建索引"""
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
    app_logger.info("RAG 管线懒加载初始化完成")
    return _rag_pipeline
```

- [ ] **Step 3: 替换 `_explore_agent` 函数体**

删除旧实现（第 132-149 行）：
```python
def _explore_agent(query: str) -> str:
    """探索 Agent：从 RAG 检索景点攻略"""
    try:
        chroma_manager = ChromaManager()
        docs = chroma_manager.similarity_search_with_score(query, k=5)

        if not docs:
            return f"未找到与「{query}」相关的攻略信息。"

        lines = [f"## 相关攻略 ({len(docs)} 条)\n"]
        for i, (doc, score) in enumerate(docs, 1):
            snippet = doc.page_content[:200].replace("\n", " ")
            source = doc.metadata.get("source", "未知来源")
            lines.append(f"{i}. [{source}] {snippet}...")
        return "\n\n".join(lines)
    except Exception as e:
        app_logger.error(f"探索 Agent 检索失败: {e}")
        return f"攻略检索异常: {e}"
```

替换为：
```python
def _explore_agent(query: str) -> str:
    """探索 Agent：通过 RAG 管道检索景点攻略（查询优化→混合检索→父文档扩展→重排序）"""
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

- [ ] **Step 4: 验证语法**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; ast.parse(open('app/agents/routers/destination_router.py', encoding='utf-8').read()); print('[OK] 语法正确')"
```

- [ ] **Step 5: Commit**

```bash
git add app/agents/routers/destination_router.py
git commit -m "feat(router): integrate RAG pipeline into explore agent with lazy loading"
```

---

### Task 3: 验证全部测试通过

- [ ] **Step 1: 运行全部纯逻辑测试**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -m pytest tests/rag/ tests/tools/ tests/agents/ -v
```

预期：所有 75 个测试 PASSED

- [ ] **Step 2: 运行语法全量检查**

```bash
cd "D:\AI agent\知行智能旅游规划助手"
python -c "import ast; [ast.parse(open(p, encoding='utf-8').read()) for p in __import__('pathlib').Path('.').rglob('*.py') if 'venv' not in str(p)]; print('[OK] 所有 Python 文件语法正确')"
```

- [ ] **Step 3: Commit（如有必要）**

---

## Self-Review Results

1. **Spec coverage**: Task 1 covers chroma_client.py simplification (all 3 removals + private method conversion). Task 2 covers destination_router.py changes (_get_rag_pipeline + _explore_agent rewrite + 7 imports). Task 3 covers verification. ✅
2. **Placeholder scan**: No TBD/TODO. All code is complete with exact line references. ✅
3. **Type consistency**: `_get_rag_pipeline()` returns `RAGPipeline`, consumed by `_explore_agent`. `_get_vectorstore` and `_get_embedding_function` use consistent naming with `_` prefix for private. ✅

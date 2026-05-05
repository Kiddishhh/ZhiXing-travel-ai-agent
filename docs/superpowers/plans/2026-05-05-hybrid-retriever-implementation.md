# Hybrid Retriever (BM25 + Dense + RRF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a hybrid retriever that combines BM25 keyword search, Dense semantic search, and RRF fusion for the travel planner's RAG pipeline.

**Architecture:** Two-layer design separating data storage (ChromaDB manager in `app/core/ChromaDB/`) from application logic (HybridRetriever in `app/rag/`). ChromaManager wraps ChromaDB PersistentClient + HuggingFaceEmbeddings. HybridRetriever extends LangChain BaseRetriever with BM25Okapi + jieba and RRF fusion.

**Tech Stack:** langchain-chroma, sentence-transformers (bge-small-zh-v1.5), rank-bm25, jieba, numpy

---

### Task 1: Create ChromaDB data storage layer

**Files:**
- Create: `app/core/ChromaDB/chroma_client.py`
- Create: `app/core/ChromaDB/__init__.py`

- [ ] **Step 1.1: Create `app/core/ChromaDB/chroma_client.py`**

```python
"""
ChromaDB 管理器 - 数据存储层
封装 PersistentClient 和 Embedding 模型
"""
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.utils.logger import app_logger


class ChromaManager:
    """ChromaDB 管理器

    职责：封装 ChromaDB 持久化客户端和嵌入模型，
    提供向量存储的增删查能力。
    """

    def __init__(self, persist_directory: str = "data/chroma_db"):
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.persist_path = Path(project_root, persist_directory)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self._embedding_function: Optional[HuggingFaceEmbeddings] = None
        self.client: PersistentClient = chromadb.PersistentClient(
            path=str(self.persist_path)
        )
        app_logger.info(f"ChromaDB 客户端已初始化，持久化路径: {self.persist_path}")

    @property
    def embedding_function(self) -> HuggingFaceEmbeddings:
        """延迟初始化的嵌入函数"""
        if self._embedding_function is None:
            self._embedding_function = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            app_logger.info("嵌入模型 BAAI/bge-small-zh-v1.5 已加载")
        return self._embedding_function

    def get_vectorstore(
        self, collection_name: str = "travel"
    ) -> Chroma:
        """获取或创建 Chroma 向量存储实例"""
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        collection_name: str = "travel",
    ) -> None:
        """向集合添加文档"""
        if not documents:
            app_logger.warning(f"没有文档可添加到集合 '{collection_name}'")
            return

        vectorstore = self.get_vectorstore(collection_name)
        vectorstore.add_documents(documents, ids=ids)
        app_logger.info(
            f"已添加 {len(documents)} 篇文档到集合 '{collection_name}'"
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        collection_name: str = "travel",
    ) -> List[Tuple[Document, float]]:
        """语义相似度检索，返回 (Document, distance) 列表"""
        vectorstore = self.get_vectorstore(collection_name)
        results = vectorstore.similarity_search_with_score(query, k=k)
        return results

    def delete_collection(self, collection_name: str) -> None:
        """删除指定集合"""
        try:
            self.client.delete_collection(collection_name)
            app_logger.info(f"已删除集合 '{collection_name}'")
        except ValueError:
            app_logger.warning(f"集合 '{collection_name}' 不存在")
```

- [ ] **Step 1.2: Create `app/core/ChromaDB/__init__.py`**

```python
from .chroma_client import ChromaManager

__all__ = ["ChromaManager"]
```

- [ ] **Step 1.3: Verify ChromaDB module imports correctly**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -c "from app.core.ChromaDB import ChromaManager; print('ChromaManager imported OK')"
```

Expected output:
```
ChromaManager imported OK
```

- [ ] **Step 1.4: Commit**

```bash
git add app/core/ChromaDB/chroma_client.py app/core/ChromaDB/__init__.py
git commit -m "feat: add ChromaDB data storage layer (ChromaManager)"
```

---

### Task 2: Create HybridRetriever

**Files:**
- Create: `app/rag/retriever.py`
- Modify: `app/rag/__init__.py`

- [ ] **Step 2.1: Create `app/rag/retriever.py`**

```python
"""
混合检索器 - 应用服务层

实现 BM25 关键词检索 + Dense 语义检索 + RRF 倒数排名融合。
继承 LangChain BaseRetriever，与 LangChain/LangGraph 生态兼容。
"""
from typing import Dict, List, Optional, Tuple

import jieba
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from rank_bm25 import BM25Okapi

from app.core.ChromaDB import ChromaManager
from app.utils.logger import app_logger


class HybridRetriever(BaseRetriever):
    """混合检索器

    组合 BM25（关键词）和 Dense（语义）两种互补的检索方式，
    通过 RRF（Reciprocal Rank Fusion）进行结果融合。
    """

    # Pydantic 可配置字段
    chroma_manager: ChromaManager
    bm25_top_k: int = 10
    dense_top_k: int = 10
    rrf_k: int = 60
    final_top_k: int = 10
    collection_name: str = "travel"

    # 私有属性（非序列化）
    _bm25: Optional[BM25Okapi] = PrivateAttr(default=None)
    _documents: List[Document] = PrivateAttr(default_factory=list)
    _doc_id_map: Dict[str, Document] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)

    def initialize(self, documents: List[Document]) -> None:
        """构建 BM25 索引和向量存储

        Args:
            documents: 已切分的子文档列表
        """
        self._documents = documents
        self._doc_id_map = {}

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            doc.metadata["doc_id"] = doc_id
            self._doc_id_map[doc_id] = doc

        self._build_bm25_index(documents)
        self._build_vector_store(documents)
        self._is_initialized = True
        app_logger.info(
            f"混合检索器初始化完成: {len(documents)} 篇文档"
        )

    # ── 索引构建 ──────────────────────────────────────

    def _build_bm25_index(self, documents: List[Document]) -> None:
        """使用 jieba 分词构建 BM25Okapi 索引"""
        tokenized_docs = []
        empty_count = 0

        for doc in documents:
            tokens = list(jieba.cut(doc.page_content))
            if tokens:
                tokenized_docs.append(tokens)
            else:
                empty_count += 1

        self._bm25 = BM25Okapi(tokenized_docs)
        app_logger.info(
            f"BM25 索引构建完成: {len(tokenized_docs)} 篇文档"
            f"{' (' + str(empty_count) + ' 篇为空被跳过)' if empty_count else ''}"
        )

    def _build_vector_store(self, documents: List[Document]) -> None:
        """将文档添加到 ChromaDB 向量存储"""
        ids = [doc.metadata["doc_id"] for doc in documents]
        self.chroma_manager.add_documents(
            documents, ids=ids, collection_name=self.collection_name
        )
        app_logger.info(
            f"向量库构建完成: {len(documents)} 篇文档添加到 '{self.collection_name}'"
        )

    # ── 检索方法 ──────────────────────────────────────

    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """BM25 关键词检索"""
        tokenized_query = list(jieba.cut(query))
        scores = np.array(self._bm25.get_scores(tokenized_query))

        # 过滤零分文档，按分数降序取 top_k
        valid = np.where(scores > 0)[0]
        if len(valid) == 0:
            return []

        sorted_idx = valid[np.argsort(scores[valid])[::-1]][:self.bm25_top_k]
        return [(f"doc_{idx}", float(scores[idx])) for idx in sorted_idx]

    def _dense_search(self, query: str) -> List[Tuple[str, float]]:
        """Dense 语义检索

        使用 similarity_search_with_score，返回 (doc_id, distance)，
        distance 越小表示越相似。
        """
        results = self.chroma_manager.similarity_search_with_score(
            query, k=self.dense_top_k, collection_name=self.collection_name
        )
        # results: List[Tuple[Document, float]]
        dense_list = []
        for doc, distance in results:
            doc_id = doc.metadata.get("doc_id", "")
            if doc_id:
                dense_list.append((doc_id, distance))
        return dense_list

    # ── RRF 融合 ──────────────────────────────────────

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """RRF（Reciprocal Rank Fusion）倒数排名融合

        对每篇出现在任一检索结果中的文档，按排名计算：
            RRF_score = Σ 1 / (rrf_k + rank)

        Args:
            bm25_results: [(doc_id, score), ...]
            dense_results: [(doc_id, distance), ...]

        Returns:
            [(doc_id, rrf_score), ...] 按 rrf_score 降序
        """
        rrf_scores: Dict[str, float] = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            )

        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            )

        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:self.final_top_k]

    # ── LangChain BaseRetriever 接口 ─────────────────

    def _get_relevant_documents(
        self,
        query: str,
        **kwargs,
    ) -> List[Document]:
        """混合检索入口（实现 BaseRetriever 抽象方法）"""
        if not query or not query.strip():
            app_logger.error("拒绝空查询")
            return []

        if not self._is_initialized:
            app_logger.error("检索器尚未初始化，请先调用 initialize()")
            return []

        if self._bm25 is None:
            app_logger.error("BM25 索引为空")
            return []

        app_logger.info(f"混合检索开始: query='{query[:50]}'")

        # 1. BM25 关键词检索
        bm25_results = self._bm25_search(query)
        app_logger.debug(f"  BM25 检索到 {len(bm25_results)} 条结果")

        # 2. Dense 语义检索
        dense_results = self._dense_search(query)
        app_logger.debug(f"  Dense 检索到 {len(dense_results)} 条结果")

        # 3. RRF 融合
        fused = self._rrf_fusion(bm25_results, dense_results)

        # 4. 组装返回
        result_docs = []
        for doc_id, rrf_score in fused:
            doc = self._doc_id_map.get(doc_id)
            if doc is not None:
                doc.metadata["rrf_score"] = round(rrf_score, 4)
                result_docs.append(doc)

        app_logger.info(
            f"混合检索完成: 返回 {len(result_docs)} 条结果"
        )
        return result_docs
```

- [ ] **Step 2.2: Update `app/rag/__init__.py`**

```python
from .retriever import HybridRetriever

__all__ = ["HybridRetriever"]
```

- [ ] **Step 2.3: Verify HybridRetriever imports correctly**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -c "from app.rag.retriever import HybridRetriever; print('HybridRetriever imported OK')"
```

Expected output:
```
HybridRetriever imported OK
```

- [ ] **Step 2.4: Commit**

```bash
git add app/rag/retriever.py app/rag/__init__.py
git commit -m "feat: implement HybridRetriever with BM25+Dense+RRF"
```

---

### Task 3: End-to-end verification

- [ ] **Step 3.1: Run full import smoke test**

Run:
```bash
cd "d:/AI agent/知行智能旅游规划助手"
python -c "
from app.core.ChromaDB import ChromaManager
from app.rag.retriever import HybridRetriever
print('All modules imported successfully')

cm = ChromaManager()
print(f'ChromaManager OK: persist_path={cm.persist_path}')
print(f'Embedding model: {cm.embedding_function.model_name}')
"
```

Expected: No import errors, ChromaManager initializes with correct persist path.

- [ ] **Step 3.2: Verify file structure**

```bash
ls -la app/core/ChromaDB/
ls -la app/rag/
```

Expected:
```
app/core/ChromaDB/:
chroma_client.py  __init__.py

app/rag/:
__init__.py  document_loader.py  retriever.py  text_splitter.py
```

- [ ] **Step 3.3: Commit final verification (if any fixes needed)**

If any fixes were needed during verification:
```bash
git add -A
git commit -m "fix: address verification issues"
```

---

## Design Decisions

1. **doc_id 方案**: 使用 `doc_{i}` 整数索引作为文档唯一标识，同时写入 metadata 和 ChromaDB id，确保 BM25 索引位置和 ChromaDB ID 解耦。

2. **RRF rank 从 1 开始**: `enumerate(..., start=1)` 确保排名从 1 开始（RRF 标准），避免 rank=0 时 `1/(k+0)` 权重过大。

3. **ChromaManager 路径计算**: 使用 `Path(__file__).resolve()` 定位项目根目录，避免工作目录依赖（与 `app/config.py:BASE_DIR` 风格一致）。

4. **延迟加载 Embedding**: `embedding_function` 属性延迟初始化，避免 `ChromaManager` 构造时加载模型导致启动变慢。

5. **BM25 零分过滤**: `_bm25_search` 中过滤 scores=0 的文档，减少 RRF 计算噪声。

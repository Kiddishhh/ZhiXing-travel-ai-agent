# RAG 管道组装设计

日期: 2026-05-14

## 概述

创建 `RAGPipeline` 类将 QueryOptimizer → HybridRetriever → expand_context → LLMReranker 串联为一条完整检索管道，并编写集成测试脚本验证端到端效果。

---

## 1. RAGPipelineResult 数据类

**文件**: `app/rag/pipeline.py`

```python
@dataclass
class RAGPipelineResult:
    original_query: str
    strategy: str                          # 使用的优化策略
    optimized_queries: list[str]           # 优化后的查询列表
    child_docs: list[Document]             # 检索到的子文档（合并去重后）
    parent_docs: list[Document]            # expand_context 后的父文档
    final_docs: list[Document]             # rerank + reorder 后的最终文档
    child_count: int                       # 子文档数（去重前）
    parent_count: int                      # 父文档数（去重后）
```

---

## 2. RAGPipeline 类

**文件**: `app/rag/pipeline.py`

### 构造器

```python
class RAGPipeline:
    def __init__(
        self,
        optimizer: QueryOptimizer,
        retriever: HybridRetriever,        # 已调用 initialize() 的检索器
        chroma_manager: ChromaManager,
        reranker: LLMReranker,
        parent_collection_name: str = "travel_parents",
    ):
```

### run() 方法

```python
def run(self, query: str) -> RAGPipelineResult:
```

数据流:
1. `QueryOptimizer.optimize(query)` → 获取 optimized_queries
2. 对每个 optimized_query 调 `HybridRetriever.invoke()`，合并去重（按 doc_id，保留最高 rrf_score）
3. `ParentDocumentSplitter.expand_context(child_docs, parent_collection)` → 父文档
4. fallback: expand 后为空则用 child_docs
5. `LLMReranker.rerank(original_query, parent_docs)` → 最终结果

### 错误处理

- 空查询 → 返回空 RAGPipelineResult
- 检索无结果 → 跳过 expand/rerank，返回空结果
- expand_context 失败 → fallback 用 child_docs 继续 rerank
- 任何步骤异常 → 记录日志，返回当前已收集的结果

---

## 3. 集成测试脚本

**文件**: `scripts/test_rag_pipeline.py`

自包含脚本：
1. 加载文档（DocumentManager）
2. 切分文档（ParentDocumentSplitter）
3. 清理旧 collection，初始化 ChromaDB
4. 构建 HybridRetriever + 索引子文档到 travel_children
5. 索引父文档到 travel_parents
6. 创建 RAGPipeline
7. 用 3-5 个真实旅游查询测试，打印每个查询的结果摘要

测试查询示例:
- "推荐北京三天亲子游行程"
- "有什么适合情侣的浪漫旅行目的地"
- "重庆火锅哪家好吃"

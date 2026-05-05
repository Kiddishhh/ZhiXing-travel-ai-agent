# 混合检索器（BM25 + Dense + RRF）设计文档

## 背景

知行智能旅游规划助手是一个基于 LangGraph 的旅游规划应用，需要从旅游文档（目的地、美食、住宿）中高效检索相关信息。单一的检索方式（纯关键词或纯语义）无法同时满足精确匹配和语义理解的需求。本方案实现 BM25 关键词检索 + Dense 语义检索 + RRF 融合的混合检索器。

## 架构概览

```
┌─────────────────────────────────────────────────┐
│              应用服务层 (app/rag/)                │
│  ┌──────────────────────────────────────────────┐│
│  │         HybridRetriever(BaseRetriever)        ││
│  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ││
│  │  │ BM25     │  │ Dense    │  │ RRF       │  ││
│  │  │ Search   │  │ Search   │  │ Fusion    │  ││
│  │  │ (jieba)  │  │(Chroma)  │  │(rank merge)│ ││
│  │  └──────────┘  └──────────┘  └───────────┘  ││
│  └──────────────────────────────────────────────┘│
├─────────────────────────────────────────────────┤
│              数据存储层 (app/core/ChromaDB/)      │
│  ┌──────────────────────────────────────────────┐│
│  │            ChromaManager                      ││
│  │  ┌──────────────┐  ┌──────────────────────┐  ││
│  │  │PersistentClient│ │HuggingFaceEmbeddings │  ││
│  │  │(chroma_db/)   │ │(bge-small-zh-v1.5)   │  ││
│  │  └──────────────┘  └──────────────────────┘  ││
│  └──────────────────────────────────────────────┘│
├─────────────────────────────────────────────────┤
│               数据源 (data/)                      │
│  ┌──────────┐  ┌──────┐  ┌─────────────┐       │
│  │destination│  │ food │  │accommodation│       │
│  │   .md     │  │ .md  │  │    .md      │       │
│  └──────────┘  └──────┘  └─────────────┘       │
└─────────────────────────────────────────────────┘
```

## 详细设计

### 1. 数据存储层 — `app/core/ChromaDB/chroma_client.py`

**类：`ChromaManager`**

职责：封装 ChromaDB PersistentClient 和 Embedding 模型，提供向量存储的增删查能力。

```
ChromaManager
├── __init__(persist_directory: str = "data/chroma_db")
│   └── 初始化 PersistentClient
│   └── 延迟初始化 HuggingFaceEmbeddings
├── embedding_function (property)
│   └── HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", normalize_embeddings=True)
├── get_or_create_collection(name: str) → Collection
├── add_documents(documents: List[Document], collection_name: str)
│   └── 通过 Chroma.from_documents 批量添加
├── similarity_search_with_score(query: str, k: int, collection_name: str)
│   └── → List[Tuple[Document, float]]  (distance, 越小越相似)
└── delete_collection(collection_name: str)
```

### 2. 应用服务层 — `app/rag/retriever.py`

**类：`HybridRetriever(BaseRetriever)`**

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chroma_manager` | `ChromaManager` | (必填) | 向量存储管理器 |
| `bm25_top_k` | `int` | 10 | BM25 检索返回数 |
| `dense_top_k` | `int` | 10 | Dense 检索返回数 |
| `rrf_k` | `int` | 60 | RRF 融合常数 |
| `final_top_k` | `int` | 10 | 最终返回结果数 |
| `collection_name` | `str` | "travel" | ChromaDB 集合名 |

#### 内部状态 (PrivateAttr)

| 属性 | 类型 | 说明 |
|------|------|------|
| `_bm25` | `BM25Okapi` | BM25 索引 |
| `_documents` | `List[Document]` | 所有索引的子文档 |
| `_doc_id_map` | `Dict[str, Document]` | doc_id → Document 映射 |
| `_is_initialized` | `bool` | 索引是否已构建 |

#### 方法

**`initialize(documents: List[Document])`**
- 为每个文档分配 doc_id（`doc_0`, `doc_1`, ...）
- 构建 `_doc_id_map`
- 调用 `_build_bm25_index(documents)`
- 调用 `_build_vector_store(documents)`

**`_build_bm25_index(documents)`**
- 对每篇文档使用 `jieba.cut()` 分词
- 构建 `BM25Okapi` 实例
- 日志记录索引规模

**`_bm25_search(query) → List[Tuple[int, float]]`**
- `jieba.cut()` 分词查询
- `BM25Okapi.get_scores()` 计算分数
- `np.argsort` 取 top_k，返回 `[(doc_index, score), ...]`

**`_dense_search(query) → List[Tuple[str, float]]`**
- `chroma_manager.similarity_search_with_score(query, k=dense_top_k)`
- ChromaDB 返回 `(Document, distance)`，distance 越小越相似
- 转换为 `[(doc_id, rank+1), ...]`（rank 从 1 开始）

**`_rrf_fusion(bm25_results, dense_results) → List[Tuple[str, float]]`**
- 对 BM25 结果：按排名 index 为每篇文档计算 `1 / (rrf_k + rank)`，rank 从 1 开始
- 对 Dense 结果：同理
- 合并 score 到 dict，按 RRF 总值降序排列
- 取 `final_top_k`

**`_get_relevant_documents(query) → List[Document]`**
- LangChain `BaseRetriever` 抽象方法入口
- 依次调用 `_bm25_search` → `_dense_search` → `_rrf_fusion`
- 根据 doc_id 从 `_doc_id_map` 取出 Document，注入 `metadata["rrf_score"]`
- 日志记录查询和结果

### 3. RRF 融合算法

```
对于每篇在任一检索器结果中出现的文档 d：
    RRF_score(d) = Σ (1 / (rrf_k + rank_i(d)))
        其中 rank_i(d) 是文档 d 在检索器 i 中的排名（从 1 开始）

最终结果按 RRF_score 降序排列，取 final_top_k 条
```

特点：
- 不依赖具体的分数值（BM25 分数和 Dense 距离不在同一量纲），只依赖排名
- `rrf_k` 控制排名权重衰减速度，越大则各排名之间差异越小（更平滑）

## 与项目现有模块的集成

### 数据流

```
DocumentManager.load_*()  →  原始 Document 列表
         ↓
ParentDocumentSplitter.split_documents()  →  parent_docs + child_docs
         ↓
HybridRetriever.initialize(child_docs)
    ├── _build_bm25_index()  →  BM25Okapi 索引
    └── _build_vector_store()  →  ChromaDB 持久化
         ↓
HybridRetriever.invoke(query)  →  List[Document] (含 rrf_score)
```

### 日志集成

所有重要操作均使用 `app_logger` 记录：
- `[INFO] BM25 索引构建完成，共 N 篇文档`
- `[INFO] 向量库构建完成，共 N 篇文档`
- `[INFO] 混合检索开始: query='...'`
- `[INFO] 混合检索完成: 返回 N 条结果 (耗时 X.XXs)`

## 边界情况与错误处理

| 场景 | 处理方式 |
|------|----------|
| BM25 索引为空 | 记录警告，返回空列表 |
| ChromaDB 集合为空 | 记录警告，仅返回 BM25 结果 |
| 查询为空字符串 | 记录错误，返回空列表 |
| 单篇文档分词后为空 | 跳过该文档，不影响整体索引 |
| RRF 融合无交集结果 | 正常按各自排名融合，不会报错 |

## 依赖项

（以下已存在于 `pyproject.toml` 中，无需额外安装）
- `rank-bm25==0.2.2` — BM25Okapi
- `jieba==0.42.1` — 中文分词
- `sentence-transformers==3.3.0` — 嵌入模型
- `langchain-chroma>=0.1.4` — ChromaDB 集成
- `langchain-core~=1.2.7` — BaseRetriever

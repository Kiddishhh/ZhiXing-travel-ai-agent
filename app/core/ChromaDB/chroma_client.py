"""
ChromaDB 管理器 - 数据存储层
封装 PersistentClient 和 Embedding 模型
"""
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
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

        self._embedding_fn: Optional[DashScopeEmbeddings] = None
        self._client: PersistentClient = chromadb.PersistentClient(
            path=str(self.persist_path)
        )
        app_logger.info(f"ChromaDB 客户端已初始化，持久化路径: {self.persist_path}")

    def _get_embedding_function(self) -> DashScopeEmbeddings:
        """私有：延迟初始化嵌入模型"""
        if self._embedding_fn is None:
            self._embedding_fn = DashScopeEmbeddings(
                model="text-embedding-v2",
            )
            app_logger.info("DashScope Embedding 已初始化 (model=text-embedding-v2)")
        return self._embedding_fn

    def _get_vectorstore(
        self, collection_name: str = "travel_children"
    ) -> Chroma:
        """私有：创建 Chroma 向量存储实例"""
        return Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self._get_embedding_function(),
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        collection_name: str = "travel_children",
    ) -> None:
        """向集合添加文档"""
        if not documents:
            app_logger.warning(f"没有文档可添加到集合 '{collection_name}'")
            return

        vectorstore = self._get_vectorstore(collection_name)
        vectorstore.add_documents(documents, ids=ids)
        app_logger.info(
            f"已添加 {len(documents)} 篇文档到集合 '{collection_name}'"
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        collection_name: str = "travel_children",
    ) -> List[Tuple[Document, float]]:
        """语义相似度检索，返回 (Document, distance) 列表

        distance 越小表示越相似（cosine distance）。
        """
        vectorstore = self._get_vectorstore(collection_name)
        results = vectorstore.similarity_search_with_score(query, k=k)
        return results

    def delete_collection(self, collection_name: str) -> None:
        """删除指定集合"""
        try:
            self._client.delete_collection(collection_name)
            app_logger.info(f"已删除集合 '{collection_name}'")
        except (ValueError, chromadb.errors.NotFoundError):
            app_logger.warning(f"集合 '{collection_name}' 不存在")

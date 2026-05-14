"""
文本切分：父文档 + 子文档，维护父子映射表
"""
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import app_logger


class ParentDocumentSplitter:
    """父文档切分器 — 维护父子映射表，支持检索后上下文扩展"""

    def __init__(
        self,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self._parent_map: Dict[str, Document] = {}

    def split_documents(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """切分文档并构建父子映射表"""
        parent_docs: List[Document] = []
        child_docs: List[Document] = []
        self._parent_map = {}
        parent_idx = 0

        for doc in documents:
            parent_chunks = self.parent_splitter.split_documents([doc])
            for parent_chunk in parent_chunks:
                parent_id = f"parent_{parent_idx}"
                parent_idx += 1
                parent_chunk.metadata["parent_id"] = parent_id
                parent_docs.append(parent_chunk)
                self._parent_map[parent_id] = parent_chunk

                child_chunks = self.child_splitter.split_documents([parent_chunk])
                for child_chunk in child_chunks:
                    child_chunk.metadata["parent_id"] = parent_id
                    child_docs.append(child_chunk)

        app_logger.info(
            f"切分文档为 {len(parent_docs)} 个父文档, "
            f"{len(child_docs)} 个子文档"
        )
        return parent_docs, child_docs

    def get_parent_context(self, child_docs: List[Document]) -> List[Document]:
        """通过子文档的 parent_id 查父文档，去重返回

        Args:
            child_docs: 检索到的子文档列表

        Returns:
            去重后的父文档列表（保持首次出现顺序）
        """
        if not self._parent_map:
            app_logger.warning("父文档映射表为空，请先调用 split_documents()")
            return []

        seen: set = set()
        parents: List[Document] = []
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if not pid or pid in seen:
                continue
            parent = self._parent_map.get(pid)
            if parent is not None:
                seen.add(pid)
                parents.append(parent)

        app_logger.info(
            f"上下文扩展: {len(child_docs)} 子文档 → {len(parents)} 父文档"
        )
        return parents

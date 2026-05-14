"""
文本切分：父文档 + 子文档
"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import app_logger

class ParentDocumentSplitter:
    """父文档切分器"""
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
            separators=["\n\n", "\n", " ", ""]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """切分文档"""
        parent_docs = []
        child_docs = []

        for doc in documents:
            parent_chunks = self.parent_splitter.split_documents([doc])
            for i, parent_chunk in enumerate(parent_chunks):
                parent_id = f"{doc.metadata.get('source','unknown')}_{i}"
                parent_chunk.metadata["parent_id"] = parent_id
                parent_docs.append(parent_chunk)

                child_chunks = self.child_splitter.split_documents([parent_chunk])

                for child_chunk in child_chunks:
                    child_chunk.metadata["parent_id"] = parent_id
                    child_docs.append(child_chunk)
                
        app_logger.info(
            f"切分文档为 {len(parent_docs)} 个父文档, "
            f"{len(child_docs)} 个子文档"
        )

        return parent_docs, child_docs

    @staticmethod
    def expand_context(
        child_docs: List[Document],
        parent_collection,
    ) -> List[Document]:
        """通过子文档 parent_id 查父文档，去重返回父文档列表

        Args:
            child_docs: 检索到的子文档列表（metadata 中需含 parent_id）
            parent_collection: ChromaDB collection（需支持 get(ids=[...]) 接口）

        Returns:
            去重后的父文档列表（保持首次出现顺序）
        """
        seen: set = set()
        parents: List[Document] = []

        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            try:
                result = parent_collection.get(ids=[pid])
                if result and result.get("documents") and result["documents"]:
                    parents.append(Document(
                        page_content=result["documents"][0],
                        metadata=result.get("metadatas", [{}])[0] or {},
                    ))
            except Exception:
                app_logger.warning(f"查找父文档失败: parent_id={pid}")
                continue

        app_logger.info(
            f"上下文扩展: {len(child_docs)} 子文档 → {len(parents)} 父文档"
        )
        return parents
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
                parent_id = f"{doc.metadata.get('source','unkown')}_{i}"
                parent_chunk.metadata["parent_id"] = parent_id
                parent_docs.append(parent_chunk)

                child_chunks = self.child_splitter.split_documents(parent_chunks)

                for child_chunk in child_chunks:
                    child_chunk.metadata["parent_id"] = parent_id
                    child_docs.append(child_chunk)
                
            app_logger.info(
                f"切分文档为 {len(parent_docs)} 个父文档"
                f"{len(child_docs)} 个子文档"
            )

            return parent_docs, child_docs
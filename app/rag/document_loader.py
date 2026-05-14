"""
文档加载器
"""
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from app.utils.logger import app_logger


class DocumentManager:
    """文档管理器 — 加载 data/documents/ 下的目的地、美食、住宿文档"""

    _CATEGORIES = [
        ("destination", "destination_guide", "destinations", "目的地"),
        ("food", "food_guide", "food", "美食"),
        ("accommodation", "accommodation_guide", "accommodations", "住宿"),
    ]

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.base_dir = Path(project_root, "data", "documents")
        else:
            self.base_dir = Path(base_dir)

    def _load_from_dir(
        self, dir_name: str, source_tag: str, category_tag: str, label: str
    ) -> List[Document]:
        """从子目录加载文档并设置元数据"""
        target_dir = self.base_dir / dir_name
        if not target_dir.exists():
            app_logger.error(f"{label}目录不存在: {target_dir}")
            return []

        loader = DirectoryLoader(
            str(target_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = source_tag
            doc.metadata["category"] = category_tag

        app_logger.info(f"加载{label}文档: {len(documents)} 个")
        return documents

    def load_destination_documents(self) -> List[Document]:
        """加载目的地文档"""
        return self._load_from_dir("destination", "destination_guide", "destinations", "目的地")

    def load_food_documents(self) -> List[Document]:
        """加载美食文档"""
        return self._load_from_dir("food", "food_guide", "food", "美食")

    def load_accommodation_documents(self) -> List[Document]:
        """加载住宿文档"""
        return self._load_from_dir("accommodation", "accommodation_guide", "accommodations", "住宿")

    def load_all_documents(self) -> List[Document]:
        """一次加载全部三类文档"""
        documents: List[Document] = []
        for dir_name, source_tag, category_tag, label in self._CATEGORIES:
            documents.extend(self._load_from_dir(dir_name, source_tag, category_tag, label))
        return documents

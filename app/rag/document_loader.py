"""
文档加载器
"""
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from app.utils.logger import app_logger

class DocumentManager: 
    """文档管理器"""
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.base_dir = Path(project_root, "data" , "documents")
        else:
            self.base_dir = Path(base_dir)

    def load_destination_documents(self) -> list[Document]:
        """加载目的地文档"""
        destination_dir = self.base_dir / "destination"
        if not destination_dir.exists():
            app_logger.error(f"目的地目录不存在: {destination_dir}")
            return []

        loader = DirectoryLoader(
            str(destination_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        documents = loader.load()
        app_logger.info(f"加载目的地文档: {len(documents)} 个")

        for doc in documents:
            doc.metadata["source"] = "destination_guide"
            doc.metadata["category"] = "destinations"
        
        return documents
    
    def load_food_documents(self) -> list[Document]:
        """加载美食文档"""
        food_dir = self.base_dir / "food"
        if not food_dir.exists():
            app_logger.error(f"美食目录不存在: {food_dir}")
            return []

        loader = DirectoryLoader(
            str(food_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        documents = loader.load()
        app_logger.info(f"加载美食文档: {len(documents)} 个")

        for doc in documents:
            doc.metadata["source"] = "food_guide"
            doc.metadata["category"] = "food"

        return documents

    def load_accommodation_documents(self) -> list[Document]:
        """加载住宿文档"""
        accommodation_dir = self.base_dir / "accommodation"
        if not accommodation_dir.exists():
            app_logger.error(f"住宿目录不存在: {accommodation_dir}")
            return []

        loader = DirectoryLoader(
            str(accommodation_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        documents = loader.load()
        app_logger.info(f"加载住宿文档: {len(documents)} 个")

        for doc in documents:
            doc.metadata["source"] = "accommodation_guide"
            doc.metadata["category"] = "accommodations"

        return documents
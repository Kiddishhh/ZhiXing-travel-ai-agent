"""
RAG 系统初始化脚本

职责:
  1. 加载 data/documents/ 下的目的地、美食、住宿文档
  2. 切分为父/子文档块
  3. 构建 BM25 索引 + ChromaDB 向量库
"""
import asyncio
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.ChromaDB.chroma_client import ChromaManager
from app.rag.document_loader import DocumentManager
from app.rag.retriever import HybridRetriever
from app.rag.text_splitter import ParentDocumentSplitter
from app.utils.logger import app_logger


def _check_env():
    """检查 .env 文件是否存在"""
    env_file = _root / ".env"
    if not env_file.exists():
        app_logger.error(f".env 文件不存在: {env_file}")
        app_logger.error("请复制 .env.example 为 .env 并填入配置后重试")
        return False
    return True


def _load_all_documents(doc_manager):
    """加载所有分类文档"""
    documents = []
    for load_fn, label in [
        (doc_manager.load_destination_documents, "目的地"),
        (doc_manager.load_food_documents, "美食"),
        (doc_manager.load_accommodation_documents, "住宿"),
    ]:
        docs = load_fn()
        documents.extend(docs)
        app_logger.info(f"  {label}: {len(docs)} 篇")
    return documents


async def init_rag_system():
    """初始化 RAG 系统（加载 → 切分 → 索引）"""
    app_logger.info("=" * 60)
    app_logger.info("RAG 系统初始化开始")
    app_logger.info("=" * 60)

    if not _check_env():
        return False

    # 1. 加载文档
    app_logger.info("加载文档...")
    doc_manager = DocumentManager()
    documents = _load_all_documents(doc_manager)
    if not documents:
        app_logger.warning("未加载到任何文档，请确认 data/documents/ 目录下有 .md 文件")
        return False
    app_logger.info(f"共加载 {len(documents)} 篇文档")

    # 2. 切分文档
    app_logger.info("切分文档...")
    splitter = ParentDocumentSplitter(
        parent_chunk_size=1000, parent_chunk_overlap=200,
        child_chunk_size=200, child_chunk_overlap=50,
    )
    parent_docs, child_docs = splitter.split_documents(documents)
    app_logger.info(f"父文档: {len(parent_docs)} 个, 子文档: {len(child_docs)} 个")

    # 3. 初始化 ChromaDB
    app_logger.info("初始化 ChromaDB...")
    chroma_manager = ChromaManager()
    chroma_manager.delete_collection("travel")
    app_logger.info("ChromaDB 客户端已就绪")

    # 4. 构建混合检索器
    app_logger.info("构建混合检索器 (BM25 + ChromaDB + RRF)...")
    retriever = HybridRetriever(chroma_manager=chroma_manager)
    retriever.initialize(child_docs)
    app_logger.info("检索器初始化完成")

    # 汇总
    app_logger.info("=" * 60)
    app_logger.info("RAG 系统初始化完成！")
    app_logger.info(f"  文档数: {len(documents)}")
    app_logger.info(f"  父文档块: {len(parent_docs)}")
    app_logger.info(f"  子文档块: {len(child_docs)}")
    app_logger.info(f"  向量库: data/chroma_db/travel")
    app_logger.info("=" * 60)
    return True


async def main():
    return await init_rag_system()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

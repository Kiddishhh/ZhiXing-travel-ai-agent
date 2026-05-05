"""
ChromaDB 向量数据库模块
提供基于 ChromaDB 的文档向量化、存储与语义检索能力。
"""
from app.core.ChromaDB.chroma_client import ChromaManager

__all__ = ["ChromaManager"]

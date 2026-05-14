"""文本切分器单元测试"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.rag.text_splitter import ParentDocumentSplitter


class TestParentDocumentSplitter:
    """ParentDocumentSplitter 测试"""

    def test_split_documents_basic(self):
        splitter = ParentDocumentSplitter(
            parent_chunk_size=500,
            parent_chunk_overlap=50,
            child_chunk_size=100,
            child_chunk_overlap=20,
        )
        doc = Document(
            page_content="北京故宫\n\n故宫又称紫禁城，是中国明清两代的皇家宫殿。\n\n位于北京中轴线的中心。",
            metadata={"source": "beijing.md", "category": "destination"},
        )
        parent_docs, child_docs = splitter.split_documents([doc])
        assert len(parent_docs) >= 1
        assert len(child_docs) >= len(parent_docs)
        for child in child_docs:
            assert "parent_id" in child.metadata
            assert child.metadata["parent_id"].startswith("beijing.md_")

    def test_split_documents_parent_id_uniqueness(self):
        splitter = ParentDocumentSplitter(
            parent_chunk_size=800, parent_chunk_overlap=100,
            child_chunk_size=100, child_chunk_overlap=20,
        )
        docs = [
            Document(page_content="A\n" * 100, metadata={"source": "doc_a.md"}),
            Document(page_content="B\n" * 100, metadata={"source": "doc_b.md"}),
        ]
        parent_docs, child_docs = splitter.split_documents(docs)
        parent_ids = set(p.metadata["parent_id"] for p in parent_docs)
        for child in child_docs:
            assert child.metadata["parent_id"] in parent_ids

    def test_expand_context_basic(self):
        child_docs = [
            Document(
                page_content="故宫是明清皇家宫殿",
                metadata={"parent_id": "beijing.md_0"},
            ),
            Document(
                page_content="故宫位于北京中轴线中心",
                metadata={"parent_id": "beijing.md_0"},
            ),
            Document(
                page_content="长城是古代军事防御工程",
                metadata={"parent_id": "beijing.md_1"},
            ),
        ]

        mock_collection = MagicMock()
        mock_collection.get.side_effect = lambda ids: {
            "documents": [f"父文档内容: {ids[0]}"],
            "metadatas": [{"parent_id": ids[0]}],
        }

        result = ParentDocumentSplitter.expand_context(child_docs, mock_collection)
        assert len(result) == 2
        assert result[0].page_content == "父文档内容: beijing.md_0"
        assert result[1].page_content == "父文档内容: beijing.md_1"

    def test_expand_context_skips_missing_parent_id(self):
        child_docs = [
            Document(page_content="内容", metadata={}),
        ]
        mock_collection = MagicMock()
        result = ParentDocumentSplitter.expand_context(child_docs, mock_collection)
        assert len(result) == 0
        mock_collection.get.assert_not_called()

    def test_expand_context_empty_list(self):
        mock_collection = MagicMock()
        result = ParentDocumentSplitter.expand_context([], mock_collection)
        assert len(result) == 0

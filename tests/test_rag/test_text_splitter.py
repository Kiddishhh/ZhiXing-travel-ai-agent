"""文本切分器单元测试"""
import pytest
from langchain_core.documents import Document
from app.rag.text_splitter import ParentDocumentSplitter


class TestParentDocumentSplitter:
    """ParentDocumentSplitter 测试"""

    def test_split_documents_basic(self):
        print("\n=== 测试: 文档切分 + 映射表构建 ===")
        splitter = ParentDocumentSplitter(
            parent_chunk_size=500, parent_chunk_overlap=50,
            child_chunk_size=100, child_chunk_overlap=20,
        )
        doc = Document(
            page_content="北京故宫\n\n故宫又称紫禁城，是中国明清两代的皇家宫殿。\n\n位于北京中轴线的中心。",
            metadata={"source": "beijing.md", "category": "destination"},
        )
        print(f"[输入] 1 篇文档, {len(doc.page_content)} 字符")
        parent_docs, child_docs = splitter.split_documents([doc])
        print(f"[结果] 父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")

        assert len(parent_docs) >= 1
        assert len(child_docs) >= len(parent_docs)
        for child in child_docs:
            assert "parent_id" in child.metadata
        print("[OK] parent_id 已传递到所有子文档\n")

    def test_split_documents_parent_id_unique(self):
        print("\n=== 测试: 跨文档 parent_id 唯一性 ===")
        splitter = ParentDocumentSplitter(
            parent_chunk_size=800, parent_chunk_overlap=100,
            child_chunk_size=100, child_chunk_overlap=20,
        )
        docs = [
            Document(page_content="A\n" * 100, metadata={"source": "doc_a.md"}),
            Document(page_content="B\n" * 100, metadata={"source": "doc_b.md"}),
        ]
        print(f"[输入] 2 篇文档")
        parent_docs, child_docs = splitter.split_documents(docs)
        print(f"[结果] 父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")

        parent_ids = set(p.metadata["parent_id"] for p in parent_docs)
        for child in child_docs:
            assert child.metadata["parent_id"] in parent_ids
        # 验证无重复
        assert len(parent_ids) == len(parent_docs)
        print(f"[OK] {len(parent_ids)} 个唯一 parent_id, 无重复\n")

    def test_get_parent_context(self):
        print("\n=== 测试: get_parent_context 子→父映射 ===")
        splitter = ParentDocumentSplitter(
            parent_chunk_size=500, parent_chunk_overlap=50,
            child_chunk_size=100, child_chunk_overlap=20,
        )
        doc = Document(
            page_content="故宫是明清两代的皇家宫殿，位于北京中轴线中心，"
                         "又称紫禁城。始建于明永乐四年，"
                         "是中国古代宫廷建筑的精华。",
            metadata={"source": "beijing.md"},
        )
        parent_docs, child_docs = splitter.split_documents([doc])
        print(f"[切分] 父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")

        # 取前 3 个子文档，模拟检索结果
        sample_children = child_docs[:3]
        print(f"[输入] {len(sample_children)} 个子文档 (含相同 parent_id)")

        result = splitter.get_parent_context(sample_children)
        print(f"[结果] 去重后 {len(result)} 个父文档")

        assert len(result) >= 1
        assert len(result) <= len(sample_children)
        for r in result:
            assert "parent_id" in r.metadata
        print("[OK] 正确去重并返回父文档\n")

    def test_get_parent_context_no_map(self):
        print("\n=== 测试: 未切分时 get_parent_context 返回空 ===")
        splitter = ParentDocumentSplitter()
        child_docs = [
            Document(page_content="内容", metadata={"parent_id": "parent_0"}),
        ]
        print("[输入] 未调用 split_documents, 直接调 get_parent_context")
        result = splitter.get_parent_context(child_docs)
        assert result == []
        print("[OK] 映射表为空时返回 []\n")

    def test_get_parent_context_missing_id(self):
        print("\n=== 测试: 子文档缺少 parent_id → 跳过 ===")
        splitter = ParentDocumentSplitter(
            parent_chunk_size=500, child_chunk_size=100,
        )
        doc = Document(page_content="测试内容 " * 20)
        splitter.split_documents([doc])

        child_docs = [
            Document(page_content="无parent_id", metadata={}),
        ]
        print("[输入] 子文档无 parent_id")
        result = splitter.get_parent_context(child_docs)
        assert result == []
        print("[OK] 正确跳过\n")

    def test_get_parent_context_empty_list(self):
        print("\n=== 测试: 空列表 → 返回空 ===")
        splitter = ParentDocumentSplitter()
        result = splitter.get_parent_context([])
        assert result == []
        print("[OK]\n")

"""
交互式 RAG 完整管道测试
运行: python tests/interactive/interactive_rag.py

功能:
  1. 加载文档 → 切分 → 构建 BM25 + ChromaDB 索引
  2. 用户输入查询词
  3. 执行: 查询优化 → 混合检索 → 父文档扩展 → LLM 重排序
  4. 打印全流程结果和耗时
"""
import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.ChromaDB.chroma_client import ChromaManager
from app.rag.document_loader import DocumentManager
from app.rag.text_splitter import ParentDocumentSplitter
from app.rag.retriever import HybridRetriever
from app.rag.query_optimizer import QueryOptimizer
from app.rag.reranker import LLMReranker
from app.rag.pipeline import RAGPipeline


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  RAG 完整管道交互式测试")
    print("=" * 60)

    # [1/5] 加载文档
    print_stage("加载文档", 5, 1)
    try:
        doc_manager = DocumentManager()
        documents = doc_manager.load_all_documents()
        if not documents:
            print("[ERROR] 未加载到任何文档，请先运行 python scripts/init_rag.py")
            return
        print(f"[OK] 加载 {len(documents)} 篇文档")
    except Exception as e:
        print(f"[ERROR] 文档加载失败: {type(e).__name__}: {e}")
        return

    # [2/5] 切分文档
    print_stage("文档切分", 5, 2)
    try:
        splitter = ParentDocumentSplitter(
            parent_chunk_size=1000, parent_chunk_overlap=200,
            child_chunk_size=200, child_chunk_overlap=50,
        )
        parent_docs, child_docs = splitter.split_documents(documents)
        print(f"[OK] 父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")
    except Exception as e:
        print(f"[ERROR] 文档切分失败: {type(e).__name__}: {e}")
        return

    # [3/5] 构建索引
    print_stage("构建 BM25 + ChromaDB 索引", 5, 3)
    try:
        chroma_manager = ChromaManager()
        chroma_manager.delete_collection("travel_children")
        retriever = HybridRetriever(
            chroma_manager=chroma_manager,
            collection_name="travel_children",
        )
        retriever.initialize(child_docs)
        print("[OK] 索引构建完成")
    except Exception as e:
        print(f"[ERROR] 索引构建失败: {type(e).__name__}: {e}")
        return

    # [4/5] 创建管线
    print_stage("创建 RAG 管线", 5, 4)
    try:
        pipeline = RAGPipeline(
            optimizer=QueryOptimizer(),
            retriever=retriever,
            parent_splitter=splitter,
            reranker=LLMReranker(top_k=5),
        )
        print("[OK] RAG 管线就绪 (QueryOptimizer + HybridRetriever + LLMReranker)")
    except Exception as e:
        print(f"[ERROR] 管线创建失败: {type(e).__name__}: {e}")
        return

    # [5/5] 交互式查询
    print_stage("交互式检索", 5, 5)
    print("输入查询词 (输入 'quit' 退出)")
    print()

    while True:
        try:
            query = input("🔍 查询: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            print("[OK] 测试结束")
            break

        print(f"\n[输入] '{query}'")
        try:
            start = time.time()
            result = pipeline.run(query)
            elapsed = time.time() - start

            print(f"  策略: {result.strategy}")
            print(f"  优化查询: {result.optimized_queries}")
            print(f"  子文档: {len(result.child_docs)}, 父文档: {len(result.parent_docs)}, "
                  f"最终: {len(result.final_docs)}")
            print(f"  耗时: {elapsed:.2f}s")

            for j, doc in enumerate(result.final_docs, 1):
                score = doc.metadata.get("relevance_score", "N/A")
                source = doc.metadata.get("source", "unknown")
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"  [{j}] score={score} source={source}")
                print(f"      {preview}...")

            if not result.final_docs:
                print("  (无结果)")
            else:
                print(f"\n[OK] 检索完成, 返回 {len(result.final_docs)} 条结果")

        except Exception as e:
            print(f"[ERROR] 检索失败: {type(e).__name__}: {e}")

    print("\n测试结束")


if __name__ == "__main__":
    asyncio.run(main())

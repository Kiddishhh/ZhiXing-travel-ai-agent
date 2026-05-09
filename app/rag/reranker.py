"""
LLM 重排序器

使用 Qwen-turbo 对检索结果进行逐点评分重排序。
"""
import re
from typing import List

from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document

from app.config import settings
from app.utils.logger import app_logger


class LLMReranker:
    """LLM 重排序器

    使用大模型对候选文档进行逐点评分（Pointwise），
    按相关性分数降序重新排序。
    """

    def __init__(
        self,
        model_name: str = "qwen-turbo",
        temperature: float = 0.0,
        top_k: int = 5,
        score_threshold: float = 0.0,
        max_chars: int = 2000,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_chars = max_chars

        self._llm = ChatTongyi(
            model=model_name,
            temperature=temperature,
            api_key=settings.dashscope_api_key,
        )

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """对候选文档进行重排序

        Args:
            query: 用户查询
            documents: 候选文档列表

        Returns:
            按相关性分数降序排列的文档列表
        """
        if not query or not query.strip():
            app_logger.warning("拒绝空查询")
            return []

        if not documents:
            app_logger.warning("文档列表为空")
            return []

        app_logger.info(
            f"LLM 重排序开始: query='{query[:50]}', 文档数={len(documents)}"
        )

        scored_docs: List[tuple[float, Document]] = []
        for doc in documents:
            score = self._score_document(query, doc)
            doc.metadata["relevance_score"] = score
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        filtered = [
            doc for score, doc in scored_docs
            if score >= self.score_threshold
        ]

        result = filtered[:self.top_k]

        app_logger.info(
            f"LLM 重排序完成: 返回 {len(result)}/{len(documents)} 条"
        )
        return result

    # ── 内部方法 ──────────────────────────────────────

    def _score_document(self, query: str, document: Document) -> float:
        """对单个文档进行评分"""
        content = document.page_content[:self.max_chars]
        prompt = self._build_prompt(query, content)

        try:
            response = self._llm.invoke(prompt)
            return self._parse_score(response.content)
        except Exception as e:
            app_logger.error(f"文档评分失败（将赋分 0）: {e}")
            return 0.0

    @staticmethod
    def _build_prompt(query: str, content: str) -> str:
        """构建评分 prompt"""
        return (
            f"你是一个旅游信息检索评估助手。请评估以下文档与用户查询的相关性。\n\n"
            f"用户查询：{query}\n\n"
            f"文档内容：\n{content}\n\n"
            f"请分析文档是否包含用户所需信息，输出一个 0-10 的整数分数：\n"
            f"- 0 = 完全不相关\n"
            f"- 10 = 非常相关，直接回答了查询问题\n\n"
            f"只输出分数："
        )

    @staticmethod
    def _parse_score(response: str) -> float:
        """解析 LLM 返回的分数"""
        match = re.search(r"(\d+)", response.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        app_logger.warning(f"无法解析 LLM 评分响应（将赋分 0）: {response}")
        return 0.0

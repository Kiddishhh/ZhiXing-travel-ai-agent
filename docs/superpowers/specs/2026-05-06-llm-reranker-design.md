# LLMReranker 重排序器设计文档

## 背景

RAG 流水线中，检索阶段（BM25 + Dense + RRF）产出的结果可能包含与用户查询相关性不高的文档。需要引入基于 LLM 的重排序器对候选文档逐点评分，提升最终送入 LLM 的上下文质量。

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 评分方式 | Pointwise 逐点评分 | 实现简单，容错性好 |
| 评分模型 | Qwen-turbo (ChatTongyi) | 复用项目已有依赖和 API 配置 |
| 类设计 | 独立工具类（非 BaseRetriever） | 重排序是检索后处理步骤，无需检索器接口 |
| API 集成 | langchain_community.chat_models.ChatTongyi | 与项目 LangChain 生态一致 |

## 架构

```
输入: query + [doc1, doc2, ...]
  → 逐文档:
      prompt = 评估查询与文档相关性（0-10分）
      ChatTongyi.invoke(prompt)
      解析分数 → metadata["relevance_score"]
  → 按分数降序排序 → threshold过滤 → top_k截取
  → 输出重排后文档列表
```

## 接口

### LLMReranker

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_name | "qwen-turbo" | 通义千问模型 |
| temperature | 0.0 | 生成温度 |
| top_k | 5 | 最终返回文档数 |
| score_threshold | 0.0 | 最低分数阈值 |
| max_chars | 2000 | 文档截断长度 |

**方法：**

- `rerank(query, documents) -> List[Document]` — 同步入口
- `arerank(query, documents) -> List[Document]` — 异步入口

## 错误处理

| 场景 | 行为 |
|------|------|
| query 为空 | 返回空列表 |
| documents 为空 | 返回空列表 |
| API 调用失败 | 赋分 0，继续处理后续文档 |
| 响应无法解析 | 赋分 0，记录警告 |
| 文档过长 | 截断前 max_chars 字符 |
| 全部低于阈值 | 返回空列表 |

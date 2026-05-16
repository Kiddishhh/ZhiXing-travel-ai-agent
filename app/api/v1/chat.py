"""对话路由：SSE 流式对话 + 历史消息"""
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.schemas.chat import ChatStreamRequest
from app.api.v1.deps import get_db
from app.core.state import create_initial_state
from app.core.checkpointer import get_checkpointer
from app.core.memory_store import get_memory_store_manager
from app.agents.handoffs.graph import create_travel_planner
from app.utils.logger import app_logger

router = APIRouter(prefix="/chat", tags=["对话"])


async def _save_message(pool, conv_id: str, role: str, content: str,
                        content_type: str = "text", token_count: int = 0,
                        is_error: bool = False):
    """保存消息到 messages 表"""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, content_type, token_count, is_error)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            uuid4(), conv_id, role, content, content_type, token_count, is_error,
        )


@router.post("/stream")
async def chat_stream(body: ChatStreamRequest, pool_user: tuple = Depends(get_db)):
    """
    SSE 流式对话

    事件类型:
    - message: AI 文本回复
    - tool_call: 工具调用开始
    - tool_result: 工具返回结果
    - step: 步骤切换
    - done: 对话完成
    - error: 出错
    """
    pool, user_id = pool_user

    # 1. 验证 conversation 归属
    async with pool.connection() as conn:
        conv = await conn.fetchrow(
            "SELECT * FROM conversations WHERE id = $1 AND status != 'deleted'",
            body.conversation_id,
        )
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    conv_data = dict(conv)
    if conv_data["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

    thread_id = body.conversation_id

    # 2. 保存用户消息
    await _save_message(pool, body.conversation_id, "user", body.message)

    async def event_generator():
        try:
            # 3. 初始化 graph
            checkpointer = await get_checkpointer()
            memory_mgr = await get_memory_store_manager()
            store = memory_mgr.get_store()
            graph = await create_travel_planner(checkpointer=checkpointer, store=store)

            # 4. 构建初始状态
            initial_state = create_initial_state(user_id=user_id, session_id=thread_id)
            initial_state["messages"].append(HumanMessage(content=body.message))

            config = {"configurable": {"thread_id": thread_id}}

            # 5. 流式执行
            current_step = None
            async for event in graph.astream_events(initial_state, config, version="v2"):
                kind = event.get("event")

                # 步骤切换
                step = event.get("metadata", {}).get("langgraph_node", "")
                if step and step != current_step:
                    current_step = step
                    yield f"event: step\ndata: {json.dumps({'step': step})}\n\n"

                # LLM 流式输出
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"event: message\ndata: {json.dumps({'content': chunk.content})}\n\n"

                # 工具调用开始
                if kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    yield f"event: tool_call\ndata: {json.dumps({'tool': tool_name, 'args': tool_input})}\n\n"

                # 工具调用结束
                if kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    preview = str(output)[:500] if output else ""
                    yield f"event: tool_result\ndata: {json.dumps({'tool': tool_name, 'preview': preview})}\n\n"

                # 保存 AI 完整消息
                if kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output and hasattr(output, "content") and output.content:
                        await _save_message(pool, body.conversation_id, "assistant", output.content)

            # 6. 更新会话
            async with pool.connection() as conn:
                await conn.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
                    body.conversation_id,
                )

            yield f"event: done\ndata: {json.dumps({'conversation_id': body.conversation_id})}\n\n"

        except Exception as e:
            app_logger.error(f"流式对话异常: {e}")
            yield f"event: error\ndata: {json.dumps({'code': 'STREAM_ERROR', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{conv_id}/messages", response_model=dict)
async def get_messages(
    conv_id: str,
    pool_user: tuple = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
):
    """获取会话历史消息"""
    pool, user_id = pool_user

    # 归属校验
    async with pool.connection() as conn:
        conv = await conn.fetchrow("SELECT user_id FROM conversations WHERE id = $1", conv_id)
        if conv is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        if dict(conv)["user_id"] != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权访问")

        rows = await conn.fetch(
            """
            SELECT * FROM messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            conv_id, limit, offset,
        )

    return {
        "conversation_id": conv_id,
        "messages": [dict(r) for r in rows],
    }

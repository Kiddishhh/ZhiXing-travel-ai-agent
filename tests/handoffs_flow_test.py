"""
Handoffs 主流程交互式测试脚本

用法:
    python tests/handoffs_flow_test.py

功能:
    1. 生成唯一会话 ID (UUID), 通过 Checkpointer 持久化会话历史
    2. 获取用户输入 (CLI 交互)
    3. 构建输入 State (create_initial_state + HumanMessage)
    4. stream_mode="values" 流式输出完整 TravelState
"""
import asyncio
import sys
import uuid
from pathlib import Path

# 项目根路径
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.messages import HumanMessage

from app.core.state import create_initial_state
from app.core.checkpointer import get_checkpointer
from app.agents.handoffs.graph import create_travel_planner
from app.utils.logger import app_logger


def _format_event(event: dict) -> str:
    """格式化 stream 输出的单条事件"""
    lines = []
    step = event.get("current_step", "?")
    messages = event.get("messages", [])

    lines.append(f"\n{'─'*50}")
    lines.append(f"[步骤: {step}]")

    if messages:
        last_msg = messages[-1]
        msg_type = type(last_msg).__name__
        content = getattr(last_msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                for c in content
            )
        # 截断过长内容
        content_str = str(content)
        if len(content_str) > 300:
            content_str = content_str[:300] + "..."
        lines.append(f"[{msg_type}] {content_str}")

    return "\n".join(lines)


async def main():
    """交互式持续对话循环"""
    # ── 1. 生成唯一会话 ID ──
    session_id = str(uuid.uuid4())
    user_id = "test_user"

    print("=" * 60)
    print("  知行智能旅游规划助手 — Handoffs Flow 测试")
    print("=" * 60)
    print(f"会话 ID: {session_id}")
    print(f"用户 ID: {user_id}")
    print("输入 'quit' 或 'exit' 退出")
    print()

    checkpointer = None
    try:
        # ── 2. 初始化 Checkpointer ──
        print("正在连接 PostgreSQL Checkpointer...")
        checkpointer = await get_checkpointer()
        print("Checkpointer 已就绪")

        # ── 3. 编译 Graph ──
        print("正在编译 Travel Planner Graph...")
        graph = await create_travel_planner(checkpointer=checkpointer)
        print("Graph 编译完成\n")

        config = {"configurable": {"thread_id": session_id}}

        # ── 4. 首轮对话 ──
        print("请输入您的第一条消息（旅行需求）:")
        first_input = input("> ").strip()
        if first_input.lower() in ("quit", "exit"):
            return

        initial_state = create_initial_state(user_id, session_id)
        initial_state["messages"].append(HumanMessage(content=first_input))

        print("\n开始流式处理...")
        async for event in graph.astream(
            initial_state, config, stream_mode="values"
        ):
            print(_format_event(event))

        # ── 5. 持续对话循环 ──
        while True:
            print(f"\n{'─'*50}")
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            # 后续调用只需传入新消息, checkpointer 自动合并历史
            update = {"messages": [HumanMessage(content=user_input)]}
            async for event in graph.astream(
                update, config, stream_mode="values"
            ):
                print(_format_event(event))

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        app_logger.error(f"测试脚本异常: {e}")
        print(f"\n错误: {e}")
    finally:
        # ── 6. 关闭 Checkpointer ──
        if checkpointer:
            manager = await _get_manager_for_close()
            if manager:
                await manager.close()

        print(f"\n会话 {session_id} 已结束")


async def _get_manager_for_close():
    """获取 CheckpointerManager 实例用于关闭"""
    from app.core.checkpointer import CheckpointerManager
    return await CheckpointerManager.get_instance()


if __name__ == "__main__":
    asyncio.run(main())

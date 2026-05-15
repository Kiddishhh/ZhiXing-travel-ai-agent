"""
交互式 Handoffs 主流程测试
运行: python tests/interactive/interactive_flow.py

功能:
  1. 生成唯一会话 ID, 通过 Checkpointer 持久化
  2. CLI 交互式对话
  3. stream_mode="values" 流式输出完整 TravelState
"""
import asyncio
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.messages import HumanMessage

from app.core.state import create_initial_state
from app.core.checkpointer import get_checkpointer
from app.core.memory_store import get_memory_store_manager
from app.agents.handoffs.graph import create_travel_planner
from app.utils.logger import app_logger


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


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
        content_str = str(content)
        if len(content_str) > 300:
            content_str = content_str[:300] + "..."
        lines.append(f"[{msg_type}] {content_str}")

    return "\n".join(lines)


async def main():
    print("=" * 60)
    print("  知行智能旅游规划助手 — Handoffs Flow 测试")
    print("=" * 60)

    session_id = str(uuid.uuid4())
    user_id = "test_user"

    print(f"会话 ID: {session_id}")
    print(f"用户 ID: {user_id}")
    print("输入 'quit' 或 'exit' 退出")
    print()

    # [1/2] 初始化
    print_stage("初始化 Graph + Checkpointer", 2, 1)
    checkpointer = None
    try:
        print("正在连接 PostgreSQL Checkpointer...")
        checkpointer = await get_checkpointer()
        print("[OK] Checkpointer 已就绪")

        print("正在编译 Travel Planner Graph...")
        memory_mgr = await get_memory_store_manager()
        store = memory_mgr.get_store()
        graph = await create_travel_planner(checkpointer=checkpointer, store=store)
        print("[OK] Graph 编译完成")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {type(e).__name__}: {e}")
        if checkpointer:
            from app.core.checkpointer import CheckpointerManager
            manager = await CheckpointerManager.get_instance()
            await manager.close()
        return

    config = {"configurable": {"thread_id": session_id}}

    # [2/2] 交互式对话
    print_stage("开始对话", 2, 2)
    print("请输入您的第一条消息（旅行需求）:")

    try:
        first_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[OK] 用户退出")
        return

    if first_input.lower() in ("quit", "exit"):
        print("[OK] 用户退出")
        return

    initial_state = create_initial_state(user_id, session_id)
    initial_state["messages"].append(HumanMessage(content=first_input))

    print("\n开始流式处理...")
    try:
        async for event in graph.astream(
            initial_state, config, stream_mode="values"
        ):
            print(_format_event(event))
    except Exception as e:
        print(f"[ERROR] 流式处理失败: {type(e).__name__}: {e}")

    # 持续对话循环
    while True:
        print(f"\n{'─'*50}")
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户中断")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("[OK] 对话结束")
            break

        update = {"messages": [HumanMessage(content=user_input)]}
        try:
            async for event in graph.astream(
                update, config, stream_mode="values"
            ):
                print(_format_event(event))
        except Exception as e:
            print(f"[ERROR] 流式处理失败: {type(e).__name__}: {e}")

    # 关闭
    if checkpointer:
        try:
            from app.core.checkpointer import CheckpointerManager
            manager = await CheckpointerManager.get_instance()
            await manager.close()
            print("[OK] Checkpointer 已关闭")
        except Exception as e:
            print(f"[WARN] Checkpointer 关闭失败: {e}")

    print(f"\n会话 {session_id} 已结束")


if __name__ == "__main__":
    asyncio.run(main())

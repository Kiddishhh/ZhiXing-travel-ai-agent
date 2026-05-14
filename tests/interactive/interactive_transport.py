"""
交互式交通规划测试
运行: python tests/interactive/interactive_transport.py

功能: 测试交通 Coordinator (航班/高铁/自驾)，用户输入行程参数。
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.agents.subagents.transport_coordinator import create_transport_coordinator


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  交通规划交互式测试")
    print("=" * 60)

    # [1/3] 初始化 Coordinator
    print_stage("初始化 Transport Coordinator", 3, 1)
    try:
        coordinator = await create_transport_coordinator()
        print("[OK] Transport Coordinator 创建完成")
    except Exception as e:
        print(f"[ERROR] Coordinator 创建失败: {type(e).__name__}: {e}")
        return

    # [2/3] 用户输入
    print_stage("输入行程参数", 3, 2)
    print("支持的交通方式: 航班(flight) / 高铁(train) / 自驾(driving)")
    print("输入 'quit' 退出")
    print()

    while True:
        try:
            transport_type = input("🚗 交通方式 (flight/train/driving): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if transport_type in ("quit", "exit"):
            break
        if transport_type not in ("flight", "train", "driving"):
            print("[WARN] 请输入 flight / train / driving")
            continue

        try:
            origin = input("📍 出发城市: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if origin.lower() in ("quit", "exit"):
            break
        if not origin:
            continue

        try:
            destination = input("📍 到达城市: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if destination.lower() in ("quit", "exit"):
            break

        try:
            date = input("📅 出发日期 (如 2026-06-01, 留空=明天): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if date.lower() in ("quit", "exit"):
            break

        type_labels = {"flight": "航班", "train": "高铁", "driving": "自驾"}
        mode_map = {
            "flight": f"我想从{origin}飞到{destination}",
            "train": f"北京到{origin}，坐高铁去{destination}",
            "driving": f"我打算自驾从{origin}到{destination}",
        }
        query = mode_map.get(transport_type, f"从{origin}到{destination}")
        if date:
            query += f"，{date}出发"
        query += "，请帮我查询"

        print(f"\n[输入] 方式={type_labels[transport_type]}, 出发={origin}, 到达={destination}"
              f"{', 日期=' + date if date else ''}")

        # [3/3] 执行查询
        print_stage("执行交通查询", 3, 3)
        try:
            response = await coordinator.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })
            content = response["messages"][-1].content
            print(f"\n[结果]\n{content}")
            print("\n[OK] 查询完成")
        except Exception as e:
            print(f"[ERROR] 交通查询失败: {type(e).__name__}: {e}")

        print()

    print("测试结束")


if __name__ == "__main__":
    asyncio.run(main())

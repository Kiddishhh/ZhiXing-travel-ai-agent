"""
交互式目的地 Router 测试
运行: python tests/interactive/interactive_destination.py

功能: 测试目的地 Router (分类器 → 探索Agent + 天气Agent)
用户输入目的地和查询类型进行测试。
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.agents.routers.destination_router import create_destination_router


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  目的地 Router 交互式测试")
    print("=" * 60)

    # [1/3] 初始化
    print_stage("初始化 Destination Router", 3, 1)
    try:
        router = create_destination_router()
        print("[OK] Router 创建完成")
    except Exception as e:
        print(f"[ERROR] Router 创建失败: {type(e).__name__}: {e}")
        return

    # [2/3] 用户输入
    print_stage("输入测试参数", 3, 2)
    print("输入 'quit' 退出")
    print()

    while True:
        try:
            destination = input("📍 目的地 (如 西安): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if destination.lower() in ("quit", "exit"):
            break
        if not destination:
            continue

        try:
            query = input("❓ 查询内容 (如 西安有什么好玩的): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("quit", "exit"):
            break
        if not query:
            query = f"{destination}旅游推荐"

        print(f"\n[输入] destination='{destination}', query='{query}'")

        # [3/3] 执行
        print_stage("执行 Router", 3, 3)
        try:
            result = await router.ainvoke({
                "original_query": query,
                "destination": destination,
            })

            print(f"\n分类结果: {result['classifications']}")
            for c in result["classifications"]:
                print(f"  → {c['agent']} Agent")
            print(f"\n最终报告:\n{result['final_report']}")
            print("\n[OK] 测试通过")
        except Exception as e:
            print(f"[ERROR] Router 执行失败: {type(e).__name__}: {e}")

        print()

    print("测试结束")


if __name__ == "__main__":
    asyncio.run(main())

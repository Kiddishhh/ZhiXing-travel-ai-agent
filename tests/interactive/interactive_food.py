"""
交互式餐饮查询测试
运行: python tests/interactive/interactive_food.py

功能: 测试餐饮查询工具 (Amap POI + Tavily 搜索)，用户输入目的地/餐饮类型。
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.tools.food_tools import query_food


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  餐饮查询交互式测试")
    print("=" * 60)
    print("餐饮类型: restaurant(餐厅) / local_snack(小吃) / specialty(特色菜) / 留空=全部")
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
            food_type = input("🍜 餐饮类型 (restaurant/local_snack/specialty, 留空=全部): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        params = {"destination": destination}
        if food_type:
            params["food_type"] = food_type

        print(f"\n[输入] destination='{destination}'" + (f", food_type='{food_type}'" if food_type else ""))

        print_stage("查询餐饮 (Amap + Tavily)", 1, 1)
        print("正在调用 Amap POI + Tavily 搜索...")
        try:
            result = await query_food.ainvoke(params)
            print(f"\n[结果]\n{result}")
            print("\n[OK] 查询完成")
        except Exception as e:
            print(f"[ERROR] 餐饮查询失败: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())

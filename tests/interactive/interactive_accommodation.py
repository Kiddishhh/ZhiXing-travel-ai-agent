"""
交互式住宿查询测试
运行: python tests/interactive/interactive_accommodation.py

功能: 测试住宿查询工具 (aigohotel-mcp)，用户输入目的地/日期/类型。
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.tools.accommodation_tools import query_accommodation


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  住宿查询交互式测试")
    print("=" * 60)
    print("输入 'quit' 退出")
    print()

    while True:
        try:
            destination = input("📍 目的地 (如 北京): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if destination.lower() in ("quit", "exit"):
            break
        if not destination:
            continue

        try:
            check_in = input("📅 入住日期 (如 2026-06-01, 留空=2026-06-01): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not check_in:
            check_in = "2026-06-01"

        try:
            nights_str = input("🌙 住宿天数 (留空=2): ").strip()
            nights = int(nights_str) if nights_str else 2
        except ValueError:
            print("[WARN] 无效数字, 使用默认值 2")
            nights = 2

        try:
            acc_type = input("🏨 住宿类型 (hotel/hostel/guesthouse, 留空=全部): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        params = {"destination": destination, "check_in_date": check_in, "stay_nights": nights}
        if acc_type:
            params["accommodation_type"] = acc_type

        print(f"\n[输入] destination='{destination}', check_in='{check_in}', "
              f"nights={nights}" + (f", type='{acc_type}'" if acc_type else ""))

        print_stage("查询住宿", 1, 1)
        try:
            result = await query_accommodation.ainvoke(params)
            print(f"\n[结果]\n{result}")
            print("\n[OK] 查询完成")
        except Exception as e:
            print(f"[ERROR] 住宿查询失败: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())

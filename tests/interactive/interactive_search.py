"""
交互式搜索查询测试
运行: python tests/interactive/interactive_search.py

功能: 测试搜索 MCP Server，用户输入关键词进行旅行信息搜索。
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.mcp_core.servers.search_server import search_travel_info


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  搜索查询交互式测试")
    print("=" * 60)
    print("输入 'quit' 退出")
    print()

    while True:
        try:
            keyword = input("🔍 搜索关键词: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if keyword.lower() in ("quit", "exit"):
            print("[OK] 测试结束")
            break
        if not keyword:
            continue

        try:
            num_str = input("📊 结果数量 (默认 5): ").strip()
            num = int(num_str) if num_str else 5
        except ValueError:
            print("[WARN] 无效数字, 使用默认值 5")
            num = 5

        print(f"\n[输入] keyword='{keyword}', num={num}")

        print_stage("搜索中", 1, 1)
        try:
            result = await search_travel_info.fn(keyword, num)
            print(f"\n[结果]\n{result}")
            print("\n[OK] 搜索完成")
        except Exception as e:
            print(f"[ERROR] 搜索失败: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())

"""
交互式天气查询测试
运行: python tests/interactive/interactive_weather.py

功能: 测试天气 MCP Server，用户输入城市 adcode 获取天气预报。
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.mcp_core.servers.weather_server import get_weather_forecast


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  天气查询交互式测试")
    print("=" * 60)
    print()
    print("常用 adcode: 北京=110000, 上海=310000, 西安=610100")
    print("           成都=510100, 杭州=330100, 广州=440100")
    print("输入 'quit' 退出")
    print()

    while True:
        try:
            adcode = input("🏙️  输入城市 adcode: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if adcode.lower() in ("quit", "exit"):
            print("[OK] 测试结束")
            break
        if not adcode:
            continue

        print(f"\n[输入] adcode='{adcode}'")

        print_stage("查询天气", 1, 1)
        try:
            result = await get_weather_forecast.fn(adcode)
            print(f"\n[结果]\n{result}")
            print("\n[OK] 查询成功")
        except Exception as e:
            print(f"[ERROR] 天气查询失败: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    asyncio.run(main())

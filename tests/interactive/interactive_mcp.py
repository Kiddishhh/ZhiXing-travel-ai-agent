"""
交互式 MCP 客户端工具列表测试
运行: python tests/interactive/interactive_mcp.py

功能: 初始化 MCPClientManager，打印所有已注册工具。
"""
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.mcp_core.client import MCPClientManager


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


async def main():
    print("=" * 60)
    print("  MCP 客户端工具列表测试")
    print("=" * 60)

    # [1/2] 初始化
    print_stage("初始化 MCP 客户端管理器", 2, 1)
    print("可用 MCP 服务: weather, search, amap, 12306-mcp, VariFlight-Aviation, aigohotel-mcp")
    choice = input("输入要连接的服务 (用逗号分隔, 留空则全部连接): ").strip()

    if choice:
        servers = [s.strip() for s in choice.split(",")]
    else:
        servers = ["weather", "search", "amap"]

    print(f"[配置] 连接服务: {servers}")

    try:
        manager = await MCPClientManager.get_instance(servers=servers)
        print("[OK] MCP 客户端管理器初始化完成")
    except Exception as e:
        print(f"[ERROR] MCP 初始化失败: {type(e).__name__}: {e}")
        return

    # [2/2] 获取工具列表
    print_stage("获取工具列表", 2, 2)
    try:
        tools = await manager.get_tools()
        print(f"\n[OK] 共发现 {len(tools)} 个工具\n")

        for i, tool in enumerate(tools, 1):
            print(f"── 工具 [{i}] ──")
            print(f"  名称: {tool.name}")
            desc = tool.description or "(无描述)"
            print(f"  描述: {desc[:100]}")
            try:
                args_str = json.dumps(tool.args, indent=4, ensure_ascii=False)
                if len(args_str) > 300:
                    args_str = args_str[:300] + "..."
                print(f"  参数: {args_str}")
            except Exception:
                print(f"  参数: {repr(tool.args)[:200]}")
            print()
    except Exception as e:
        print(f"[ERROR] 获取工具列表失败: {type(e).__name__}: {e}")
    finally:
        try:
            await manager.close()
            print("[OK] MCP 连接已关闭")
        except Exception as e:
            print(f"[WARN] 关闭失败: {e}")

    print("\n测试结束")


if __name__ == "__main__":
    asyncio.run(main())

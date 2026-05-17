"""
交互式 LLM 连接测试
运行: python tests/interactive/interactive_llm.py

功能: 测试千问 LLM 连接，用户可自定义测试消息。
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def print_stage(stage: str, total: int, current: int):
    print(f"\n{'='*60}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'='*60}")


def main():
    print("=" * 60)
    print("  LLM 连接测试 — 千问 (DashScope)")
    print("=" * 60)

    # [1/3] 初始化模型
    print_stage("初始化 ChatOpenAI 模型", 3, 1)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] DASHSCOPE_API_KEY 未设置，请检查 .env 文件")
        return
    model_name = os.getenv("QWEN_MODEL_NAME", "qwen3.6-flash")
    print(f"[配置] model={model_name}, base_url=https://dashscope.aliyuncs.com/compatible-mode/v1")

    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
        )
        print("[OK] 模型初始化完成")
    except Exception as e:
        print(f"[ERROR] 模型初始化失败: {type(e).__name__}: {e}")
        return

    # [2/3] 用户输入
    print_stage("输入测试消息", 3, 2)
    print("输入 'quit' 退出对话")
    print()

    while True:
        try:
            user_msg = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[OK] 用户退出")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit"):
            print("[OK] 测试结束")
            break

        print(f"[输入] 收到消息 ({len(user_msg)} 字符)")

        # [3/3] 调用 LLM
        print_stage("LLM 推理", 3, 3)
        try:
            response = model.invoke([HumanMessage(content=user_msg)])
            content = response.content if hasattr(response, "content") else str(response)
            print(f"\n🤖 AI:\n{content}\n")
            print("[OK] 调用成功")
        except Exception as e:
            print(f"[ERROR] LLM 调用失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

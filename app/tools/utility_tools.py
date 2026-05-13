"""
通用工具
提供日期时间等基础功能
"""
from datetime import datetime
from langchain.tools import tool


@tool
def get_current_date() -> str:
    """
    获取当前日期和时间信息

    返回当前日期、时间、星期等信息，用于处理用户提到的相对日期
    （如"今天""明天""下周三""周末"等）。

    Returns:
        str: JSON 格式的日期信息，包含:
            - date: 当前日期 (YYYY-MM-DD)
            - datetime: 完整日期时间
            - weekday_cn: 中文星期
            - weekday: ISO 星期编号 (1=周一, 7=周日)
    """
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return (
        f"当前时间信息\n"
        f"- 日期: {now.strftime('%Y-%m-%d')}\n"
        f"- 时间: {now.strftime('%H:%M:%S')}\n"
        f"- 星期: {weekdays[now.weekday()]}\n"
        f"- ISO 日期: {now.strftime('%Y-%m-%d')} (星期{now.isoweekday()})"
    )

"""
订单生成工具集

后续实现时的注意事项:
- 汇总全部旅行计划生成最终订单
- 订单内容包括: 目的地、交通、住宿、餐饮、行程、预算
- 生成订单号并持久化 (PostgreSQL)
- 可选: 支付接口对接
"""
from langchain_core.tools import tool


# TODO: 实现订单生成逻辑
# 输入: 无 (从 State 中读取全部规划数据)
# 逻辑: 汇总  生成订单号  持久化到 PostgreSQL
# 返回: 订单号 + 完整订单摘要
# 工具签名: create_order() -> str
# 注册名: "create_order"
@tool
def create_order() -> str:
    """订单生成 (占位)"""
    return "订单生成功能待实现。请调用 generate_order_tool 完成流程。"

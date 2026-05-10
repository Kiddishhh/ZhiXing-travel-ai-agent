"""
预算计算工具集

后续实现时的注意事项:
- 汇总交通 + 住宿 + 餐饮 + 景点门票 + 杂费
- 与用户预算限额对比, 超支时给出警告
- 返回 BudgetBreakdown 结构
"""
from langchain_core.tools import tool


# TODO: 实现预算计算逻辑
# 输入: transport_cost, accommodation_cost, food_cost, attractions_cost, misc_cost
# 逻辑: 逐项累加  比较 budget_max  超支警告
# 返回: BudgetBreakdown (transport, accommodation, food, attractions, misc, total)
# 工具签名: calculate_budget(transport: float, accommodation: float, food: float,
#                            attractions: float, misc: float, budget_limit: float) -> dict
# 注册名: "calculate_budget"
@tool
def calculate_budget(transport: float, accommodation: float, food: float,
                     attractions: float, misc: float, budget_limit: float) -> str:
    """预算计算 (占位)"""
    total = transport + accommodation + food + attractions + misc
    if total > budget_limit:
        return (f"预算计算结果: 总计 {total} 元, 超出预算 {total - budget_limit} 元。"
                f"建议回退调整。")
    return f"预算计算结果: 总计 {total} 元, 在预算 {budget_limit} 元以内。"

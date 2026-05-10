"""
餐饮规划工具集

后续对接 API 时的注意事项:
- 大众点评/美团开放平台 API
- 或高德地图 POI 搜索 (keywords=美食)
- 需要获取目的地经纬度坐标
- 结果按美食类型过滤: 特色美食 / 连锁快餐 / 本地小吃
"""
from langchain_core.tools import tool


# TODO: 接入餐厅查询 API
# 接口: 大众点评/美团开放平台 或 高德 POI 搜索 (keywords=美食)
# 参数: destination(目的地), food_type(美食类型), count(返回数量)
# 返回: 餐厅列表 (名称、类型、人均消费、评分、位置)
# 工具签名: query_restaurants(destination: str, food_type: str = "") -> str
# 注册名: "query_restaurants"
@tool
async def query_restaurants(destination: str, food_type: str = "") -> str:
    """餐厅查询 (占位)"""
    type_hint = f" ({food_type})" if food_type else ""
    return f"餐厅查询功能待实现 (目的地: {destination}{type_hint})"


# TODO: 接入本地小吃查询 API
# 接口: 大众点评/美团 或 高德 POI 搜索 (keywords=小吃)
# 参数: destination, count
# 返回: 本地小吃列表 (名称、价格区间、推荐指数)
# 工具签名: query_local_food(destination: str) -> str
# 注册名: "query_local_food"
@tool
async def query_local_food(destination: str) -> str:
    """本地小吃查询 (占位)"""
    return f"本地小吃查询功能待实现 (目的地: {destination})"

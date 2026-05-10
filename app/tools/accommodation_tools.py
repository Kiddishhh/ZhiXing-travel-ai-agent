"""
住宿规划工具集

后续对接 API 时的注意事项:
- 携程/飞猪开放平台 API
- 或高德地图 POI 搜索 (酒店/民宿)
- 需要获取目的地经纬度坐标
- 结果包含: 名称、类型、位置、价格、评分、设施
"""
from langchain_core.tools import tool


# TODO: 接入酒店查询 API
# 接口: 携程/飞猪开放平台 或 高德 POI 搜索 (keywords=酒店)
# 参数: destination(目的地), check_in(入住日期), check_out(离店日期),
#       price_min, price_max, rating_min
# 返回: 酒店列表 (名称、星级、价格、评分、位置)
# 工具签名: query_hotels(destination: str, check_in: str, check_out: str) -> str
# 注册名: "query_hotels"
@tool
async def query_hotels(destination: str, check_in: str, check_out: str) -> str:
    """酒店查询 (占位)"""
    return f"酒店查询功能待实现 (目的地: {destination}, {check_in} ~ {check_out})"


# TODO: 接入民宿查询 API
# 接口: 途家/爱彼迎开放平台 或 高德 POI 搜索 (keywords=民宿)
# 参数: destination, check_in, check_out, price_range
# 返回: 民宿列表 (名称、价格、评分、位置、设施)
# 工具签名: query_hostels(destination: str, check_in: str, check_out: str) -> str
# 注册名: "query_hostels"
@tool
async def query_hostels(destination: str, check_in: str, check_out: str) -> str:
    """民宿查询 (占位)"""
    return f"民宿查询功能待实现 (目的地: {destination}, {check_in} ~ {check_out})"

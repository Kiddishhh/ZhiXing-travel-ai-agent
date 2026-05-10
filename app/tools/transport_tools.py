"""
交通规划工具集

后续对接 API 时的注意事项:
- 高德地图 API: https://lbs.amap.com/api/webservice/summary
- 12306/航司 API 需额外申请
- 所有工具返回统一结构方便前端渲染
"""
from langchain_core.tools import tool


# TODO: 接入高德地图驾车路径规划 API
# 接口: GET https://restapi.amap.com/v3/direction/driving
# 参数: origin(lng,lat), destination(lng,lat), strategy(0-5)
# 返回: 路线距离(m)、预估时间(s)、费用(元)
# 前置: 需要先调用地理编码接口获取经纬度坐标
# 工具签名: query_driving_route(origin: str, destination: str) -> str
# 注册名: "query_driving_route"
@tool
async def query_driving_route(origin: str, destination: str) -> str:
    """自驾路线查询 (占位)"""
    return f"自驾路线查询功能待实现 (出发: {origin}, 到达: {destination})"


# TODO: 接入航班查询 API
# 接口: 飞猪/携程开放平台 或 航空数据聚合接口
# 参数: departure_city, destination, date (YYYY-MM-DD)
# 返回: 航班号、出发/到达时间、时长、票价
# 工具签名: query_flight(departure_city: str, destination: str, date: str) -> str
# 注册名: "query_flight"
@tool
async def query_flight(departure_city: str, destination: str, date: str) -> str:
    """航班查询 (占位)"""
    return f"航班查询功能待实现 ({departure_city}  {destination}, {date})"


# TODO: 接入高铁/火车查询 API
# 接口: 12306 官方 API 或第三方聚合接口
# 参数: departure_city, destination, date (YYYY-MM-DD)
# 返回: 车次、出发/到达时间、时长、票价
# 工具签名: query_train(departure_city: str, destination: str, date: str) -> str
# 注册名: "query_train"
@tool
async def query_train(departure_city: str, destination: str, date: str) -> str:
    """高铁/火车查询 (占位)"""
    return f"火车查询功能待实现 ({departure_city}  {destination}, {date})"

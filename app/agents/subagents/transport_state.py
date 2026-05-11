"""
交通规划 Subagent State 定义

包含三种交通方式的详细 TypedDict 和 TransportState（子图内部状态）。
"""
from operator import add
from typing import Annotated, NotRequired, TypedDict, Dict, List

from app.core.state import TransportType


class FlightOption(TypedDict):
    """单个航班选项"""
    flight_number: str        # 航班号
    airline: str              # 航空公司
    departure_airport: str    # 出发机场
    arrival_airport: str      # 到达机场
    departure_time: str       # 起飞时间
    arrival_time: str         # 降落时间
    duration: str             # 飞行时长
    price: float              # 价格
    cabin_class: str          # 舱位等级
    available_seats: int      # 剩余座位


class TrainOption(TypedDict):
    """单个车次选项"""
    train_number: str                 # 车次号
    departure_station: str            # 出发站
    arrival_station: str              # 到达站
    departure_time: str               # 发车时间
    arrival_time: str                 # 到站时间
    duration: str                     # 运行时长
    seat_types: List[str]             # 可选座位类型
    prices: Dict[str, float]          # 各座位类型价格
    available: bool                   # 是否有票


class DrivingRoute(TypedDict):
    """自驾路线"""
    route_name: str          # 路线名称
    distance: str            # 总距离
    duration: str            # 预计时长
    toll_fee: float          # 过路费
    fuel_cost: float         # 油费估算
    steps: List[str]         # 导航步骤
    waypoints: List[str]     # 途经城市


class TransportState(TypedDict):
    """交通规划子图内部状态"""

    # 用户选择
    selected_transport: NotRequired[TransportType]

    # 查询参数
    origin_city: NotRequired[str]
    destination_city: NotRequired[str]
    departure_date: NotRequired[str]
    passenger_count: NotRequired[int]

    # 查询结果（累加）
    flight_options: Annotated[List[FlightOption], add]
    train_options: Annotated[List[TrainOption], add]
    driving_routes: Annotated[List[DrivingRoute], add]

    # 最终选择
    selected_flight: NotRequired[FlightOption]
    selected_train: NotRequired[TrainOption]
    selected_route: NotRequired[DrivingRoute]

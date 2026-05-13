"""
旅行规划系统 State 定义

包含枚举类型、子结构 TypedDict、主 TravelState、初始化函数和回退辅助数据。
"""
from operator import add
from typing import Annotated, Literal, TypedDict, Optional, NotRequired, List

from langgraph.graph import MessagesState

# ── 枚举 ──

PlanningStep = Literal[
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
    "order_generation",
]

TravelStyle = Literal["relaxation", "culture", "adventure", "food"]
BudgetLevel = Literal["economy", "comfort", "luxury"]
TransportType = Literal["flight", "train", "driving"]
AccommodationType = Literal["star_hotel", "economy_hotel", "hostel", "youth_hostel"]
FoodType = Literal["specialty", "chain", "local"]

# ── 子结构 ──

class UserRequirement(TypedDict):
    departure_city: str
    destination: str
    departure_date: str
    travel_days: int
    adult_count: int
    children_count: int
    budget_min: Optional[float]
    budget_max: Optional[float]
    budget_level: Optional[BudgetLevel]
    travel_styles: Optional[List[TravelStyle]]
    special_needs: Optional[str]


class DestinationInfo(TypedDict):
    name: str
    description: str
    weather_info: Optional[str]
    attractions: List[str]
    estimated_cost: Optional[float]


class TransportInfo(TypedDict):
    transport_type: TransportType
    details: str
    departure_time: str
    arrival_time: str
    duration: str
    price: float


class AccommodationInfo(TypedDict):
    name: str
    type: AccommodationType
    location: str
    price_per_night: float
    rating: Optional[float]
    amenities: List[str]


class FoodInfo(TypedDict):
    type: FoodType
    recommendations: List[str]
    estimated_daily_cost: float


class ItineraryDay(TypedDict):
    day_number: int
    activities: List[str]
    meals: List[str]
    accommodation: str


class BudgetBreakdown(TypedDict):
    transport: float
    accommodation: float
    food: float
    attractions: float
    misc: float
    total: float


# ── 步骤顺序 ──

STEP_ORDER: List[PlanningStep] = [
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
    "order_generation",
]

# ── 回退辅助数据 ──

ALLOWED_BACK_STEPS: set[PlanningStep] = {
    "requirement_collection",
    "destination_recommendation",
    "transport_planning",
    "accommodation_planning",
    "food_planning",
    "itinerary_generation",
    "budget_summarization",
}

STEP_CLEANUP_MAP: dict[PlanningStep, list[str]] = {
    "requirement_collection": [
        "selected_destination", "selected_transport",
        "selected_accommodation_types", "selected_food_types",
        "destination_options", "transport_options",
        "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "destination_recommendation": [
        "selected_transport", "selected_accommodation_types",
        "selected_food_types", "transport_options",
        "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "transport_planning": [
        "selected_accommodation_types", "selected_food_types",
        "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "accommodation_planning": [
        "selected_food_types", "food_options",
        "itinerary", "budget", "report", "order_id",
    ],
    "food_planning": [
        "itinerary", "budget", "report", "order_id",
    ],
    "itinerary_generation": [
        "budget", "report", "order_id",
    ],
    "budget_summarization": [
        "report", "order_id",
    ],
}

# ── 主 State ──

class TravelState(MessagesState):
    """旅行规划系统主状态，继承 MessagesState 自动获得 messages 字段"""

    # ── 流程控制 ──
    current_step: NotRequired[PlanningStep]  # 当前步骤
    tool_call_history: NotRequired[
        Annotated[List[dict], add]           # 工具调用历史（累加）
    ]

    # ── 用户输入 ──
    user_requirement: NotRequired[UserRequirement]  # 用户需求

    # ── 用户选择 ──
    selected_destination: NotRequired[str]                          # 选中的目的地
    selected_transport: NotRequired[TransportType]                  # 选中的交通方式
    selected_accommodation_types: NotRequired[List[AccommodationType]]  # 选中的住宿类型（多选）
    selected_food_types: NotRequired[List[FoodType]]                 # 选中的餐饮类型（多选）

    # ── 查询结果 ──
    destination_options: NotRequired[List[DestinationInfo]]     # 目的地选项
    transport_options: NotRequired[List[TransportInfo]]         # 交通选项
    accommodation_options: NotRequired[List[AccommodationInfo]] # 住宿选项
    food_options: NotRequired[List[FoodInfo]]                   # 餐饮选项

    # ── 最终结果 ──
    itinerary: NotRequired[List[ItineraryDay]]   # 行程安排
    budget: NotRequired[BudgetBreakdown]         # 预算明细
    report: NotRequired[str]                     # 旅行报告（Markdown 格式）
    order_id: NotRequired[str]                   # 订单号

    # ── 审批状态 ──
    approval_pending: NotRequired[bool]    # 是否等待审批
    approval_reason: NotRequired[str]      # 审批原因

    # ── 上下文压缩 ──
    context_summary: NotRequired[str]  # 历史对话压缩摘要（由 guard 节点生成）

    # ── 元数据 ──
    user_id: NotRequired[str]        # 用户 ID
    session_id: NotRequired[str]     # 会话 ID
    created_at: NotRequired[float]   # 创建时间（Unix 时间戳）
    updated_at: NotRequired[float]   # 更新时间


def create_initial_state(user_id: str, session_id: str) -> dict:
    """创建初始状态，返回符合 TravelState 的 dict"""
    import time
    return {
        "messages": [],
        "current_step": "requirement_collection",
        "destination_options": [],
        "transport_options": [],
        "accommodation_options": [],
        "food_options": [],
        "approval_pending": False,
        "user_id": user_id,
        "session_id": session_id,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

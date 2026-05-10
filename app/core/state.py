"""
旅行规划系统 State 定义

包含枚举类型、子结构 TypedDict、主 TravelState、初始化函数和回退辅助数据。
"""
from typing import Literal, TypedDict, Optional, NotRequired, List

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
    budget_level: BudgetLevel
    travel_styles: List[TravelStyle]
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
    current_step: NotRequired[PlanningStep]
    user_requirement: NotRequired[UserRequirement]
    selected_destination: NotRequired[str]
    selected_transport: NotRequired[TransportType]
    selected_accommodation_types: NotRequired[List[AccommodationType]]
    selected_food_types: NotRequired[List[FoodType]]
    destination_options: NotRequired[List[DestinationInfo]]
    transport_options: NotRequired[List[TransportInfo]]
    accommodation_options: NotRequired[List[AccommodationInfo]]
    food_options: NotRequired[List[FoodInfo]]
    itinerary: NotRequired[List[ItineraryDay]]
    budget: NotRequired[BudgetBreakdown]
    report: NotRequired[str]
    order_id: NotRequired[str]
    approval_pending: NotRequired[bool]
    approval_reason: NotRequired[str]
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    created_at: NotRequired[float]
    updated_at: NotRequired[float]


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

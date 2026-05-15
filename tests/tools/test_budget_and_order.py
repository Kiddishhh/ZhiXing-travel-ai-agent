"""预算计算和订单生成测试

注意: calculate_budget 和 create_order 使用 ToolRuntime 注入模式,
runtime 参数由 ToolNode 自动注入。测试中通过 .func() 直接调用底层函数。
"""
import pytest
from unittest.mock import MagicMock
from app.tools.budget_tools import calculate_budget
from app.tools.order_tools import create_order


def _print_stage(stage: str, total: int, current: int):
    print(f"\n{'─'*50}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'─'*50}")


def make_mock_runtime(state_overrides=None):
    """构造模拟的 ToolRuntime"""
    state = {
        "current_step": "budget_summarization",
        "user_requirement": {
            "departure_city": "北京",
            "destination": "西安",
            "departure_date": "2026-06-01",
            "travel_days": 3,
            "adult_count": 2,
            "children_count": 0,
            "budget_min": 2000,
            "budget_max": 5000,
            "budget_level": "comfort",
            "travel_styles": ["culture"],
            "special_needs": None,
        },
        "selected_destination": "西安",
        "selected_transport": "train",
        "selected_accommodation_types": ["star_hotel"],
        "selected_food_types": ["specialty", "local"],
        "transport_options": [
            {"transport_type": "train", "details": "G651", "departure_time": "08:00", "arrival_time": "12:30", "duration": "4h30m", "price": 550.0}
        ],
        "accommodation_options": [
            {"name": "西安钟楼酒店", "type": "star_hotel", "location": "钟楼附近", "price_per_night": 400.0, "rating": 4.5, "amenities": ["WiFi", "早餐"]}
        ],
        "food_options": [
            {"type": "specialty", "recommendations": ["羊肉泡馍", "肉夹馍"], "estimated_daily_cost": 150.0}
        ],
        "itinerary": [
            {"day_number": 1, "date": "2026-06-01", "activities": ["兵马俑"], "meals": ["午餐: 回民街"], "accommodation": "西安钟楼酒店", "transport": "步行"},
            {"day_number": 2, "date": "2026-06-02", "activities": ["大雁塔"], "meals": ["晚餐: 饺子宴"], "accommodation": "西安钟楼酒店", "transport": "公交"},
            {"day_number": 3, "date": "2026-06-03", "activities": ["城墙"], "meals": ["午餐: 凉皮"], "accommodation": "无", "transport": "高铁返回"},
        ],
    }
    if state_overrides:
        state.update(state_overrides)

    mock_runtime = MagicMock()
    mock_runtime.state = state
    mock_runtime.tool_call_id = "test-001"
    return mock_runtime


class TestCalculateBudget:
    """测试预算计算工具"""

    def test_calculate_budget_normal(self):
        """测试正常预算计算"""
        _print_stage("calculate_budget", 3, 1)
        print("[注入] runtime state: 西安3日游, comfort档")
        runtime = make_mock_runtime()
        # 通过 .func() 直接调用底层函数, 绕过 @tool 的 schema 验证
        result = calculate_budget.func(runtime=runtime)

        assert isinstance(result, str)
        assert "交通" in result
        assert "住宿" in result
        assert "餐饮" in result
        assert "总计" in result

    def test_calculate_budget_over_limit(self):
        """测试超预算场景"""
        _print_stage("calculate_budget", 3, 2)
        print("[注入] budget_max=500 (人均), 2人 → 总预算上限=1000, grand_total≈3740 超支")
        runtime = make_mock_runtime()
        runtime.state["user_requirement"]["budget_max"] = 500
        result = calculate_budget.func(runtime=runtime)

        assert isinstance(result, str)
        assert "超支" in result or "超出" in result
        assert "人均" in result

    def test_calculate_budget_with_rooms(self):
        """测试多房间预算计算"""
        _print_stage("calculate_budget", 3, 3)
        print("[注入] rooms_needed=2, 4成人")
        runtime = make_mock_runtime()
        runtime.state["user_requirement"]["adult_count"] = 4
        result = calculate_budget.func(rooms_needed=2, runtime=runtime)

        assert isinstance(result, str)
        assert "2间" in result
        print(f"[OK] 住宿费中已包含 2 间房计算")

    def test_calculate_budget_missing_data(self):
        """测试缺少数据时的处理"""
        _print_stage("calculate_budget", 3, 3)
        print("[注入] transport/accommodation/food 全部为空")
        runtime = make_mock_runtime({"transport_options": [], "accommodation_options": [], "food_options": []})
        result = calculate_budget.func(runtime=runtime)

        assert isinstance(result, str)
        assert "总计" in result


class TestCreateOrder:
    """测试订单生成工具"""

    def test_create_order(self):
        """测试生成完整订单"""
        _print_stage("create_order", 1, 1)
        print("[注入] runtime state: 西安3日完整行程")
        runtime = make_mock_runtime()
        result = create_order.func(runtime=runtime)

        assert isinstance(result, str)
        assert "西安" in result
        assert "2026-06-01" in result
        assert "兵马俑" in result

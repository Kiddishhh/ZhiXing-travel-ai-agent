"""用户长期记忆工具测试"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.tools.memory_tools import save_user_preference, auto_save_from_state


def _print_stage(stage: str, total: int, current: int):
    print(f"\n{'─'*50}")
    print(f"  [{current}/{total}] {stage}")
    print(f"{'─'*50}")


def make_mock_runtime(state_overrides=None):
    """构造模拟 ToolRuntime"""
    state = {
        "current_step": "order_generation",
        "user_id": "test-user-001",
        "session_id": "test-session-001",
        "user_requirement": {
            "departure_city": "成都",
            "destination": "重庆",
            "departure_date": "2026-06-15",
            "travel_days": 2,
            "adult_count": 2,
            "children_count": 0,
            "budget_min": 1000,
            "budget_max": 3000,
            "budget_level": "comfort",
            "travel_styles": ["food", "culture"],
            "special_needs": None,
        },
        "selected_destination": "重庆",
        "selected_transport": "train",
        "selected_accommodation_types": ["star_hotel"],
        "selected_food_types": ["specialty"],
        "transport_options": [
            {"transport_type": "train", "details": "G1234", "price": 150.0}
        ],
        "accommodation_options": [
            {"name": "重庆解放碑酒店", "price_per_night": 350.0}
        ],
        "food_options": [
            {"type": "specialty", "estimated_daily_cost": 120.0}
        ],
    }
    if state_overrides:
        state.update(state_overrides)

    mock_runtime = MagicMock()
    mock_runtime.state = state
    mock_runtime.tool_call_id = "test-memory-001"
    return mock_runtime


class TestSaveUserPreference:
    """测试显式偏好保存工具"""

    @pytest.mark.asyncio
    async def test_save_transport_preference(self):
        """测试保存交通偏好"""
        _print_stage("save_user_preference — transport", 4, 1)
        print("[注入] preference_type='transport', value='高铁'")
        runtime = make_mock_runtime()

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.upsert_profile = AsyncMock(return_value={
                "user_id": "test-user-001",
                "preferred_transport": "高铁",
                "total_trips": 0,
            })
            mock_get_mgr.return_value = mock_mgr

            result = await save_user_preference.coroutine(
                preference_type="transport",
                value="高铁",
                runtime=runtime,
            )

        assert isinstance(result, str)
        assert "高铁" in result
        print(f"[OK] 返回: {result}")

    @pytest.mark.asyncio
    async def test_save_food_preference(self):
        """测试保存饮食偏好（数组字段）"""
        _print_stage("save_user_preference — food", 4, 2)
        print("[注入] preference_type='food', value='川菜'")
        runtime = make_mock_runtime()

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.upsert_profile = AsyncMock(return_value={
                "user_id": "test-user-001",
                "dietary_preferences": ["川菜"],
            })
            mock_get_mgr.return_value = mock_mgr

            result = await save_user_preference.coroutine(
                preference_type="food",
                value="川菜",
                runtime=runtime,
            )

        assert "川菜" in result
        print(f"[OK] 返回: {result}")

    @pytest.mark.asyncio
    async def test_save_invalid_preference_type(self):
        """测试无效偏好类型"""
        _print_stage("save_user_preference — invalid type", 4, 3)
        print("[注入] preference_type='invalid', value='xxx'")
        runtime = make_mock_runtime()

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_get_mgr.return_value = mock_mgr

            result = await save_user_preference.coroutine(
                preference_type="invalid",
                value="xxx",
                runtime=runtime,
            )

        assert "未知" in result
        print(f"[OK] 返回: {result}")


class TestAutoSaveFromState:
    """测试自动保存工具"""

    @pytest.mark.asyncio
    async def test_auto_save_full_state(self):
        """测试从完整 state 自动提取并保存画像"""
        _print_stage("auto_save_from_state", 2, 1)
        runtime = make_mock_runtime()
        print("[注入] 2人重庆2日游 comfort 档，train + specialty")

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            new_callable=AsyncMock,
        ) as mock_get_mgr:
            mock_mgr = AsyncMock()
            mock_mgr.upsert_profile = AsyncMock(return_value={
                "user_id": "test-user-001",
                "total_trips": 5,
                "last_destination": "重庆",
                "last_travel_date": "2026-06-15",
            })
            mock_get_mgr.return_value = mock_mgr

            result = await auto_save_from_state.coroutine(runtime=runtime)

        assert isinstance(result, str)
        assert "5" in result or "重庆" in result
        print(f"[OK] 返回: {result}")

    @pytest.mark.asyncio
    async def test_auto_save_db_failure_graceful(self):
        """测试数据库不可用时降级不崩溃"""
        _print_stage("auto_save_from_state — DB failure", 2, 2)
        runtime = make_mock_runtime()
        print("[注入] get_memory_store_manager 抛出异常")

        with patch(
            "app.tools.memory_tools.get_memory_store_manager",
            side_effect=Exception("数据库连接失败"),
        ):
            result = await auto_save_from_state.coroutine(runtime=runtime)

        # 不抛异常，返回错误提示
        assert isinstance(result, str)
        print(f"[OK] 降级处理: {result}")

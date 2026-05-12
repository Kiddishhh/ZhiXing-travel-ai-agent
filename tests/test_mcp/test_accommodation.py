"""住宿工具集成测试"""
import pytest
from app.tools.accommodation_tools import query_accommodation


@pytest.mark.asyncio
async def test_query_accommodation_hotel():
    """测试查询酒店"""
    result = await query_accommodation.ainvoke({
        "destination": "北京",
        "check_in_date": "2026-06-01",
        "stay_nights": 2,
        "accommodation_type": "hotel",
    })
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)


@pytest.mark.asyncio
async def test_query_accommodation_all_types():
    """测试查询所有住宿类型（不指定 type）"""
    result = await query_accommodation.ainvoke({
        "destination": "上海",
        "check_in_date": "2026-07-15",
        "stay_nights": 3,
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)

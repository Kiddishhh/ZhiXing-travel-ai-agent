"""餐饮工具集成测试"""
import pytest
from app.tools.food_tools import query_food


@pytest.mark.asyncio
async def test_query_food_restaurant():
    """测试查询餐厅"""
    result = await query_food.ainvoke({
        "destination": "西安",
        "food_type": "restaurant",
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)


@pytest.mark.asyncio
async def test_query_food_local_snack():
    """测试查询本地小吃"""
    result = await query_food.ainvoke({
        "destination": "成都",
        "food_type": "local_snack",
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)


@pytest.mark.asyncio
async def test_query_food_all():
    """测试查询所有餐饮类型"""
    result = await query_food.ainvoke({
        "destination": "重庆",
    })
    assert isinstance(result, str)
    assert len(result) > 0
    print(result)

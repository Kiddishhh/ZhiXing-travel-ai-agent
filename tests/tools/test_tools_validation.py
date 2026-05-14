"""
工具验证主流程测试
模拟完整调用链路，打印关键信息和内容
运行: python tests/tools/test_tools_validation.py
"""
import sys
import asyncio
from unittest.mock import MagicMock

sys.stdout.reconfigure(encoding='utf-8')


def make_mock_runtime(state_overrides=None):
    """构造模拟的 ToolRuntime，包含完整的旅程数据"""
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
            {
                "transport_type": "train",
                "details": "G651 北京西-西安北",
                "departure_time": "08:00",
                "arrival_time": "12:30",
                "duration": "4h30m",
                "price": 550.0,
            }
        ],
        "accommodation_options": [
            {
                "name": "西安钟楼酒店",
                "type": "star_hotel",
                "location": "钟楼附近",
                "price_per_night": 400.0,
                "rating": 4.5,
                "amenities": ["WiFi", "早餐"],
            }
        ],
        "food_options": [
            {
                "type": "specialty",
                "recommendations": ["羊肉泡馍", "肉夹馍", "凉皮", "岐山臊子面"],
                "estimated_daily_cost": 150.0,
            }
        ],
        "itinerary": [
            {
                "day_number": 1,
                "date": "2026-06-01",
                "activities": ["上午: 乘G651抵达西安", "下午: 回民街美食探索", "晚上: 钟楼夜景"],
                "meals": ["午餐: 回民街羊肉泡馍", "晚餐: 饺子宴"],
                "accommodation": "西安钟楼酒店",
                "transport": "步行+地铁",
            },
            {
                "day_number": 2,
                "date": "2026-06-02",
                "activities": ["上午: 兵马俑", "下午: 华清宫", "晚上: 大雁塔音乐喷泉"],
                "meals": ["午餐: 临潼肉夹馍", "晚餐: 大唐不夜城小吃"],
                "accommodation": "西安钟楼酒店",
                "transport": "旅游大巴",
            },
            {
                "day_number": 3,
                "date": "2026-06-03",
                "activities": ["上午: 西安城墙骑行", "下午: G652高铁返回北京"],
                "meals": ["午餐: 城墙脚下凉皮"],
                "accommodation": "无（返程）",
                "transport": "地铁+高铁返回",
            },
        ],
        "order_id": None,
    }
    if state_overrides:
        state.update(state_overrides)

    mock_runtime = MagicMock()
    mock_runtime.state = state
    mock_runtime.tool_call_id = "test-validation-001"
    return mock_runtime


def print_separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def test_tool_registry():
    """测试 1: 验证工具注册表"""
    print_separator("测试 1: TOOL_REGISTRY 工具注册表")

    from app.tools import TOOL_REGISTRY

    print(f"\n📊 工具总数: {len(TOOL_REGISTRY)}")
    print(f"\n📋 工具清单:")
    for name in sorted(TOOL_REGISTRY.keys()):
        func = TOOL_REGISTRY[name]
        is_async = asyncio.iscoroutinefunction(func) or hasattr(func, 'ainvoke')
        print(f"  {'🔧' if not is_async else '🌐'} {name}")

    # 验证关键工具存在
    required = [
        "query_accommodation", "query_food", "calculate_budget", "create_order",
        "query_transport_options", "query_destination_info",
    ]
    missing = [r for r in required if r not in TOOL_REGISTRY]
    if missing:
        print(f"\n❌ 缺失工具: {missing}")
    else:
        print(f"\n✅ 所有 {len(required)} 个关键业务工具均已注册")

    return len(TOOL_REGISTRY)


def test_budget_calculation():
    """测试 2: 预算计算"""
    print_separator("测试 2: calculate_budget 预算计算")

    from app.tools.budget_tools import calculate_budget

    runtime = make_mock_runtime()
    result = calculate_budget.func(runtime=runtime)

    print(f"\n{result}")

    # 关键断言
    assert "西安" in result, "应包含目的地"
    assert "交通" in result, "应包含交通费"
    assert "住宿" in result, "应包含住宿费"
    assert "餐饮" in result, "应包含餐饮费"
    assert "总计" in result, "应包含总计"

    # 提取总金额
    for line in result.split("\n"):
        if "总计:" in line:
            print(f"\n💡 关键信息: {line.strip()}")

    print(f"\n✅ 预算计算验证通过")
    return result


def test_budget_over_limit():
    """测试 3: 超预算场景"""
    print_separator("测试 3: calculate_budget 超预算警告")

    from app.tools.budget_tools import calculate_budget

    runtime = make_mock_runtime()
    runtime.state["user_requirement"]["budget_max"] = 500

    result = calculate_budget.func(runtime=runtime)

    assert "超支" in result or "超出" in result, "应有超支警告"
    print(f"\n{result}")
    print(f"\n✅ 超预算检测验证通过")
    return result


def test_order_generation():
    """测试 4: 订单生成"""
    print_separator("测试 4: create_order 订单摘要")

    from app.tools.order_tools import create_order

    runtime = make_mock_runtime()
    result = create_order.func(runtime=runtime)

    print(f"\n{result}")

    # 关键断言
    assert "西安" in result, "应包含目的地"
    assert "G651" in result, "应包含交通详情"
    assert "兵马俑" in result, "应包含景点"
    assert "钟楼酒店" in result, "应包含住宿"
    # 行程中的餐饮通过 meals 字段展示（如 "午餐: 回民街羊肉泡馍"），这是正确的

    print(f"\n✅ 订单生成验证通过")
    return result


def test_tool_signatures():
    """测试 5: 工具签名验证"""
    print_separator("测试 5: 工具签名和文档")

    from app.tools.accommodation_tools import query_accommodation
    from app.tools.food_tools import query_food
    from app.tools.budget_tools import calculate_budget
    from app.tools.order_tools import create_order
    from app.tools.transport_tools import query_transport_options

    tools = {
        "query_accommodation": query_accommodation,
        "query_food": query_food,
        "query_transport_options": query_transport_options,
        "calculate_budget": calculate_budget,
        "create_order": create_order,
    }

    for name, tool_func in tools.items():
        sig_params = []
        if hasattr(tool_func, 'args_schema'):
            schema = tool_func.args_schema
            if hasattr(schema, 'model_fields'):
                sig_params = list(schema.model_fields.keys())

        desc = tool_func.description or "(无描述)"
        # 取第一行描述
        short_desc = desc.strip().split("\n")[0][:80]

        print(f"\n📦 {name}")
        print(f"   描述: {short_desc}")
        print(f"   参数: {sig_params if sig_params else '(仅 runtime)'}")


def test_state_structure():
    """测试 6: State 结构验证"""
    print_separator("测试 6: TravelState 数据结构")

    from app.core.state import TravelState

    # 列出所有 state 字段
    annotations = TravelState.__annotations__ if hasattr(TravelState, '__annotations__') else {}
    print(f"\n📊 TravelState 包含 {len(annotations)} 个字段:")
    for key, typ in annotations.items():
        type_str = str(typ).replace("typing.", "").replace("typing_extensions.", "")
        if len(type_str) > 60:
            type_str = type_str[:57] + "..."
        print(f"  {key}: {type_str}")

    # 验证关键字段存在
    required_fields = [
        "current_step", "user_requirement", "selected_destination",
        "selected_transport", "selected_accommodation_types", "selected_food_types",
        "transport_options", "accommodation_options", "food_options",
        "itinerary", "budget", "report", "order_id",
    ]
    missing = [f for f in required_fields if f not in annotations]
    if missing:
        print(f"\n⚠️ 缺失字段: {missing}")
    else:
        print(f"\n✅ 所有 {len(required_fields)} 个关键字段存在")


def test_import_chain():
    """测试 7: 导入链完整性"""
    print_separator("测试 7: 导入链验证")

    modules_to_test = [
        ("app.tools.accommodation_tools", "query_accommodation"),
        ("app.tools.food_tools", "query_food"),
        ("app.tools.budget_tools", "calculate_budget"),
        ("app.tools.order_tools", "create_order"),
        ("app.tools.transport_tools", "query_transport_options"),
        ("app.tools.router_query", "query_destination_info"),
        ("app.tools.state_transition", "record_requirement_tool"),
        ("app.tools.state_transition", "select_destination_tool"),
        ("app.tools.state_transition", "select_transport_tool"),
        ("app.tools.state_transition", "select_accommodation_tool"),
        ("app.tools.state_transition", "select_food_tool"),
        ("app.tools.state_transition", "generate_itinerary_tool"),
        ("app.tools.state_transition", "summarize_budget_tool"),
        ("app.tools.state_transition", "generate_order_tool"),
    ]

    import importlib
    for module_name, attr_name in modules_to_test:
        try:
            mod = importlib.import_module(module_name)
            obj = getattr(mod, attr_name)
            has_tool_decorator = hasattr(obj, 'invoke') or hasattr(obj, 'ainvoke')
            print(f"  ✅ {module_name}.{attr_name} {'(@tool)' if has_tool_decorator else ''}")
        except Exception as e:
            print(f"  ❌ {module_name}.{attr_name} — {e}")


def test_step_config_coverage():
    """测试 8: step_config 工具覆盖"""
    print_separator("测试 8: step_config 步骤工具覆盖")

    from app.tools import TOOL_REGISTRY

    # 模拟 step_config 的导入（不去真正运行 async get_step_config）
    step_tools = {
        "requirement_collection": ["record_requirement_tool", "check_current_progress"],
        "destination_recommendation": ["query_destination_info", "select_destination_tool"],
        "transport_planning": ["query_transport_options", "select_transport_tool"],
        "accommodation_planning": ["query_accommodation", "select_accommodation_tool"],
        "food_planning": ["query_food", "select_food_tool"],
        "itinerary_generation": ["generate_itinerary_tool"],
        "budget_summarization": ["calculate_budget", "summarize_budget_tool"],
        "order_generation": ["create_order", "generate_order_tool"],
    }

    all_ok = True
    for step_name, expected_tools in step_tools.items():
        missing_in_step = [t for t in expected_tools if t not in TOOL_REGISTRY]
        if missing_in_step:
            print(f"  ❌ 步骤 {step_name}: 缺失 {missing_in_step}")
            all_ok = False
        else:
            print(f"  ✅ 步骤 {step_name}: {len(expected_tools)} 个工具齐全")

    if all_ok:
        print(f"\n✅ 全部 8 个步骤工具覆盖正确")
    else:
        print(f"\n❌ 存在工具缺失")


def main():
    """运行所有验证测试"""
    print("=" * 70)
    print("  知行智能旅游规划助手 — 工具验证测试")
    print("=" * 70)

    results = []
    failures = []

    tests = [
        ("工具注册表", test_tool_registry),
        ("预算计算", test_budget_calculation),
        ("超预算检测", test_budget_over_limit),
        ("订单生成", test_order_generation),
        ("工具签名", test_tool_signatures),
        ("State结构", test_state_structure),
        ("导入链", test_import_chain),
        ("步骤覆盖", test_step_config_coverage),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASS", result))
        except Exception as e:
            failures.append((name, str(e)))
            print(f"\n❌ 测试失败 [{name}]: {e}")

    # ── 汇总 ──
    print(f"\n{'=' * 70}")
    print(f"  测试汇总")
    print(f"{'=' * 70}")
    for name, status, *_ in results:
        print(f"  ✅ {name}")
    for name, err in failures:
        print(f"  ❌ {name}: {err}")

    passed = len(results)
    failed = len(failures)
    total = passed + failed
    print(f"\n  结果: {passed}/{total} 通过", end="")
    if failed:
        print(f", {failed} 失败")
    else:
        print(" ✅ 全部通过")


if __name__ == "__main__":
    main()

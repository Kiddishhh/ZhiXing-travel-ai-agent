"""
步骤配置中心

集中管理 8 个步骤的 prompt、工具列表和前置依赖。
get_step_config() 返回的字典由 AgentMiddleware 在每次 LLM 调用前查询。
"""
from app.tools.state_transition import (
    record_requirement_tool,
    select_destination_tool,
    select_transport_tool,
    select_accommodation_tool,
    select_food_tool,
    generate_itinerary_tool,
    summarize_budget_tool,
    generate_order_tool,
    go_back_to_step,
    go_back_to_requirement,
    go_back_to_destination,
    go_back_to_transport,
    go_back_to_accommodation,
    go_back_to_food,
    go_back_to_itinerary,
    go_back_to_budget,
    check_current_progress,
)
from app.tools.router_query import query_destination_info
from app.tools.transport_tools import query_transport_options
from app.tools.accommodation_tools import query_accommodation
from app.tools.food_tools import query_food
from app.tools.budget_tools import calculate_budget
from app.tools.order_tools import create_order
from app.tools.utility_tools import get_current_date


async def get_step_config() -> dict:
    """
    获取步骤配置字典。

    每步结构:
    - prompt: system prompt (使用 {field_name} 占位符, 由 middleware 渲染)
    - tools: 该步可用的工具函数列表
    - requires: 前置 State 字段 (未就绪时 middleware 报错拦截)
    """
    return {
        # ========== 步骤 1: 需求收集 ==========
        "requirement_collection": {
            "prompt": """你是专业的旅行规划顾问, 负责收集用户的旅行需求。

**当前阶段**: 需求收集 (第 1 步, 共 8 步)

**需要收集的信息**:
- 🏠 出发地点
- 📅 出发日期
- 🗓️ 出行天数
- 👥 人数 (成人/儿童)
- 💰 预算范围 (元/人)
- 🎨 旅行风格: relaxation/culture/adventure/food (可多选)
- 📝 特殊需求 (可选)

**操作指南**:
- 一次只问 1-2 个问题, 保持对话自然
- 信息完整后 → 调用 `record_requirement_tool` 进入下一步
- 这是第一步, 无回退选项
""",
            "tools": [
                record_requirement_tool,
                check_current_progress,
                get_current_date,
            ],
            "requires": []
        },

        # ========== 步骤 2: 目的地推荐 ==========
        "destination_recommendation": {
            "prompt": """你是目的地推荐专家。

**当前阶段**: 目的地推荐 (第 2 步, 共 8 步)

**用户需求**:
- 出发日期: {user_requirement}
- 预算: {user_requirement}
- 旅行风格: {user_requirement}

**任务**:
1. 调用 `query_destination_info` 获取目的地信息
2. 根据需求推荐 3 个目的地, 说明特色和适合理由
3. 用户确认后 → 调用 `select_destination_tool`

**回退选项**:
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")` 或 `go_back_to_step("requirement_collection", reason="...")`
""",
            "tools": [
                query_destination_info,
                select_destination_tool,
                go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement"]
        },

        # ========== 步骤 3: 交通规划 ==========
        "transport_planning": {
            "prompt": """你是交通规划专家。

**当前阶段**: 交通规划 (第 3 步, 共 8 步)

**已确定信息**:
- 目的地: {selected_destination}

**可用的交通查询工具**:
- `query_transport_options` — 统一交通查询 (支持航班/高铁/自驾)

**任务**:
1. 推荐交通方式: flight(航班) / train(高铁) / driving(自驾)
2. 调用 `query_transport_options` 查询具体信息
3. 用户确认后 → 调用 `select_transport_tool`

**回退选项**:
- 换目的地 → `go_back_to_destination(reason="...")` 或 `go_back_to_step("destination_recommendation", reason="...")`
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")`
""",
            "tools": [
                query_transport_options,
                select_transport_tool,
                go_back_to_destination, go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement", "selected_destination"]
        },

        # ========== 步骤 4: 住宿规划 ==========
        "accommodation_planning": {
            "prompt": """你是住宿规划专家。

**当前阶段**: 住宿规划 (第 4 步, 共 8 步)

**已确定信息**:
- 目的地: {selected_destination}
- 交通: {selected_transport}

**可用的住宿查询工具**:
- `query_accommodation` — 住宿查询 (支持酒店/民宿/青旅)

**任务**:
1. 推荐住宿类型: 🏨 星级酒店 / 🏠 民宿 / 🛏️ 青旅 (可多选)
2. 调用查询工具获取具体选项
3. 用户确认后 → 调用 `select_accommodation_tool`

**回退选项**:
- 换交通 → `go_back_to_transport(reason="...")`
- 换目的地 → `go_back_to_destination(reason="...")`
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")`
""",
            "tools": [
                query_accommodation,
                select_accommodation_tool,
                go_back_to_transport, go_back_to_destination, go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement", "selected_destination", "selected_transport"]
        },

        # ========== 步骤 5: 餐饮规划 ==========
        "food_planning": {
            "prompt": """你是餐饮规划专家。

**当前阶段**: 餐饮规划 (第 5 步, 共 8 步)

**已确定信息**:
- 目的地: {selected_destination}
- 住宿类型: {selected_accommodation_types}

**可用的餐饮查询工具**:
- `query_food` — 餐饮查询 (支持特色美食/连锁快餐/本地小吃)

**任务**:
1. 推荐餐饮类型: 🍜 特色美食 / 🍔 连锁快餐 / 🍘 本地小吃 (可多选)
2. 调用查询工具获取具体选项
3. 用户确认后 → 调用 `select_food_tool`

**回退选项**:
- 换住宿 → `go_back_to_accommodation(reason="...")`
- 换交通 → `go_back_to_transport(reason="...")`
- 换目的地 → `go_back_to_destination(reason="...")`
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")`
""",
            "tools": [
                query_food,
                select_food_tool,
                go_back_to_accommodation, go_back_to_transport,
                go_back_to_destination, go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement", "selected_destination", "selected_transport", "selected_accommodation_types"]
        },

        # ========== 步骤 6: 行程生成 ==========
        "itinerary_generation": {
            "prompt": """你是行程规划专家。

**当前阶段**: 行程生成 (第 6 步, 共 8 步)

**已收集信息**:
- 目的地: {selected_destination}
- 交通: {selected_transport}
- 住宿: {selected_accommodation_types}
- 餐饮: {selected_food_types}

**任务**:
1. 根据用户需求生成每日详细行程
2. 包含景点、餐饮、住宿安排
3. 用户确认后 → 调用 `generate_itinerary_tool`

**回退选项**:
- 改餐饮 → `go_back_to_food(reason="...")`
- 改住宿 → `go_back_to_accommodation(reason="...")`
- 改交通 → `go_back_to_transport(reason="...")`
- 换目的地 → `go_back_to_destination(reason="...")`
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")`
""",
            "tools": [
                generate_itinerary_tool,
                go_back_to_food, go_back_to_accommodation,
                go_back_to_transport, go_back_to_destination,
                go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement", "selected_destination", "selected_transport", "selected_accommodation_types", "selected_food_types"]
        },

        # ========== 步骤 7: 预算汇总 ==========
        "budget_summarization": {
            "prompt": """你是预算分析专家。

**当前阶段**: 预算汇总 (第 7 步, 共 8 步)

**任务**:
1. 调用 `calculate_budget` 计算费用明细
2. 展示: 交通 + 住宿 + 餐饮 + 门票 + 杂费
3. 如超预算, 建议回退调整
4. 用户确认后 → 调用 `summarize_budget_tool`

**回退选项**:
- 改行程 → `go_back_to_itinerary(reason="...")`
- 改餐饮 → `go_back_to_food(reason="...")`
- 改住宿 → `go_back_to_accommodation(reason="...")`
- 改交通 → `go_back_to_transport(reason="...")`
- 换目的地 → `go_back_to_destination(reason="...")`
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")`
- 回到任意步骤 → `go_back_to_step("<step_name>", reason="...")`
""",
            "tools": [
                calculate_budget,
                summarize_budget_tool,
                go_back_to_itinerary, go_back_to_food,
                go_back_to_accommodation, go_back_to_transport,
                go_back_to_destination, go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement", "itinerary"]
        },

        # ========== 步骤 8: 订单生成 ==========
        "order_generation": {
            "prompt": """你是订单处理专家。

**当前阶段**: 订单生成 (第 8 步, 共 8 步)

**任务**:
1. 调用 `create_order` 生成最终订单摘要供用户确认
2. 展示完整旅行计划摘要
3. 用户确认后 → 调用 `generate_order_tool` (自动结束流程)

**回退选项** (最后修改机会):
- 看预算 → `go_back_to_budget(reason="...")`
- 改行程 → `go_back_to_itinerary(reason="...")`
- 改餐饮 → `go_back_to_food(reason="...")`
- 改住宿 → `go_back_to_accommodation(reason="...")`
- 改交通 → `go_back_to_transport(reason="...")`
- 换目的地 → `go_back_to_destination(reason="...")`
- 重新规划整个旅行 → `go_back_to_requirement(reason="...")`
- 回到任意步骤 → `go_back_to_step("<step_name>", reason="...")`
""",
            "tools": [
                create_order,
                generate_order_tool,
                go_back_to_budget, go_back_to_itinerary, go_back_to_food,
                go_back_to_accommodation, go_back_to_transport,
                go_back_to_destination, go_back_to_requirement,
                go_back_to_step,
                check_current_progress,
                get_current_date,
            ],
            "requires": ["user_requirement", "itinerary", "budget"]
        },
    }

"""
步骤配置中间件

StepConfigResolver 在每次 LLM 调用前:
1. 读取 current_step
2. 查 step_config 获取对应 prompt + tools
3. 验证前置依赖
4. 返回渲染后的 prompt 和工具列表
"""
from app.utils.logger import app_logger
from app.core.memory_store import get_memory_store_manager


class StepConfigResolver:
    """步骤配置解析器 - 根据 current_step 返回对应的 prompt 和 tools"""

    def __init__(self, step_config: dict):
        self._step_config = step_config


    #状态注入current_step,返回对应步骤的system_prompt and tools
    async def resolve(self, state: dict) -> tuple:
        """
        根据 current_step 解析步骤配置。

        参数:
        - state: 当前 TravelState 字典

        返回:
        - (system_prompt: str, tools: list)

        抛出:
        - ValueError: 未知步骤或前置依赖缺失
        """
        current_step = state.get("current_step", "requirement_collection")

        app_logger.info(f"当前步骤: {current_step}")

        if current_step not in self._step_config:
            app_logger.error(f"未知步骤: {current_step}")
            raise ValueError(f"未知步骤: {current_step}")

        step_config = self._step_config[current_step]

        # ── 验证前置依赖 ──
        for required_field in step_config["requires"]:
            val = state.get(required_field)
            if val is None:
                error_msg = (
                    f"步骤 {current_step} 需要 '{required_field}' 字段, "
                    f"但当前未设置"
                )
                app_logger.error(f"前置依赖缺失: {error_msg}")
                raise ValueError(error_msg)
            app_logger.debug(f"前置依赖满足: {required_field}")

        # ── 渲染 prompt ──
        try:
            system_prompt = step_config["prompt"].format(**state)
        except KeyError as e:
            app_logger.warning(f"prompt 占位符无法渲染: {e}, 使用原始模板")
            system_prompt = step_config["prompt"]

        # ── 注入用户长期记忆 ──
        user_id = state.get("user_id")
        if user_id:
            try:
                manager = await get_memory_store_manager()
                profile = await manager.get_profile(user_id)
                if profile:
                    profile_text = _format_profile_for_prompt(profile)
                    system_prompt += f"\n\n{profile_text}"
                    app_logger.info(f"已注入用户画像 (user_id={user_id})")
            except Exception as e:
                app_logger.warning(f"画像注入失败，跳过: {e}")

        app_logger.info(f"已解析步骤配置: {len(step_config['tools'])} 个工具")
        return system_prompt, step_config["tools"]


def _format_profile_for_prompt(profile: dict) -> str:
    """将 user_profiles 行格式化为 prompt 可用的画像文本"""
    lines = ["[用户长期画像]"]

    transport = profile.get("preferred_transport")
    if transport:
        lines.append(f"- 交通偏好: {transport}")

    budget = profile.get("budget_level")
    if budget:
        lines.append(f"- 预算档位: {budget}")

    styles = profile.get("travel_styles") or []
    if styles:
        lines.append(f"- 旅行风格: {', '.join(styles)}")

    dests = profile.get("favorite_destinations") or []
    if dests:
        lines.append(f"- 偏好目的地: {', '.join(dests)}")

    diets = profile.get("dietary_preferences") or []
    if diets:
        lines.append(f"- 饮食偏好: {', '.join(diets)}")

    total = profile.get("total_trips", 0)
    if total:
        last_dest = profile.get("last_destination", "") or ""
        last_date = profile.get("last_travel_date", "") or ""
        parts = [f"共{total}次"]
        if last_date:
            parts.append(f"最近一次{last_date}")
        if last_dest:
            parts.append(f"去{last_dest}")
        lines.append(f"- 历史出行: {'，'.join(parts)}")

    extensions = profile.get("extensions") or {}
    for k, v in extensions.items():
        if v:
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


async def create_step_config_resolver() -> StepConfigResolver:
    """
    工厂函数: 创建预加载配置的 StepConfigResolver 实例
    """
    from app.agents.handoffs.step_config import get_step_config

    step_config = await get_step_config()
    app_logger.info("StepConfigResolver 创建完成")
    return StepConfigResolver(step_config)

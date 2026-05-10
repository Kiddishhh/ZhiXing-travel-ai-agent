"""
步骤配置中间件

StepConfigResolver 在每次 LLM 调用前:
1. 读取 current_step
2. 查 step_config 获取对应 prompt + tools
3. 验证前置依赖
4. 返回渲染后的 prompt 和工具列表
"""
from app.utils.logger import app_logger


class StepConfigResolver:
    """步骤配置解析器 - 根据 current_step 返回对应的 prompt 和 tools"""

    def __init__(self, step_config: dict):
        self._step_config = step_config

    def resolve(self, state: dict) -> tuple:
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

        app_logger.info(f"已解析步骤配置: {len(step_config['tools'])} 个工具")
        return system_prompt, step_config["tools"]


async def create_step_config_resolver() -> StepConfigResolver:
    """
    工厂函数: 创建预加载配置的 StepConfigResolver 实例
    """
    from app.agents.handoffs.step_config import get_step_config

    step_config = await get_step_config()
    app_logger.info("StepConfigResolver 创建完成")
    return StepConfigResolver(step_config)

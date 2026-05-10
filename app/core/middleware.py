"""
步骤配置中间件

AgentMiddleware 实现 awrap_model_call 钩子, 在每次 LLM 调用前:
1. 读取 current_step
2. 查 step_config 获取对应 prompt + tools
3. 验证前置依赖
4. 注入配置到模型请求
"""
from typing import Callable, Any

from langgraph.config import ModelRequest, ModelResponse
from app.core.state import TravelState
from app.utils.logger import app_logger


class AgentMiddleware:
    """步骤配置中间件 - 根据 current_step 动态配置 Agent"""

    def __init__(self, step_config: dict):
        self._step_config = step_config

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """
        根据 current_step 动态注入 prompt 和 tools
        """
        state: TravelState = request.state
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

        # ── 注入 prompt + tools ──
        try:
            system_prompt = step_config["prompt"].format(**state)
        except KeyError as e:
            app_logger.warning(f"prompt 占位符无法渲染: {e}, 使用原始模板")
            system_prompt = step_config["prompt"]

        modified_request = request.override(
            system_prompt=system_prompt,
            tools=step_config["tools"]
        )

        app_logger.info(
            f"已注入步骤配置: {len(step_config['tools'])} 个工具"
        )
        return await handler(modified_request)


async def create_step_config_middleware() -> AgentMiddleware:
    """
    工厂函数: 创建预加载配置的 AgentMiddleware 实例
    """
    from app.agents.handoffs.step_config import get_step_config

    step_config = await get_step_config()
    app_logger.info("AgentMiddleware 创建完成")
    return AgentMiddleware(step_config)

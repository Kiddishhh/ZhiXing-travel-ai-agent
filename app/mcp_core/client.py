"""
MCP 客户端管理器
统一管理所有 MCP 服务连接
"""
import asyncio
import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.utils.logger import app_logger

load_dotenv()


class MCPClientManager:
    """
    MCP 客户端管理器（单例模式）
    """
    _instance: Optional['MCPClientManager'] = None
    _client: Optional[MultiServerMCPClient] = None
    _tools: Optional[list] = None
    _lock = asyncio.Lock()

    # 项目根目录（用于 stdio 服务）
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # 环境变量（追加 PYTHONPATH）
    ENV_VARS = os.environ.copy()
    ENV_VARS["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + ENV_VARS.get("PYTHONPATH", "")

    # 服务器配置
    SERVER_CONFIGS = {
        # ========== 自建服务 (stdio) ==========
        "weather": {
            "command": "python",
            "args": ["-m", "app.mcp_core.servers.weather_server"],
            "transport": "stdio",
            "env": ENV_VARS,
        },
        "search": {
            "command": "python",
            "args": ["-m", "app.mcp_core.servers.search_server"],
            "transport": "stdio",
            "env": ENV_VARS,
        },

        # ========== 外部服务 (HTTP) ==========
        "amap": {
            "url": f"https://mcp.amap.com/mcp?key={os.getenv('AMAP_API_KEY', '')}",
            "transport": "http",
        },
        "12306-mcp": {
            "url": "https://mcp.api-inference.modelscope.net/215d3cfb299e47/mcp",
            "transport": "streamable_http",
        },
        "VariFlight-Aviation": {
            "url": f"https://ai.variflight.com/servers/aviation/mcp/?api_key={os.getenv('VARIFLIGHT_API_KEY', '')}",
            "transport": "streamable_http",
        },
        "aigohotel-mcp": {
            "url": "https://mcp.aigohotel.com/mcp",
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {os.getenv('AIGOHOTEL_MCP_API', '')}",
                "Content-Type": "application/json"
            }
        },
    }

    @classmethod
    async def get_instance(cls, servers: Optional[List[str]] = None) -> 'MCPClientManager':
        """获取单例实例"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    try:
                        await instance.initialize(servers=servers)
                        cls._instance = instance
                    except Exception:
                        raise
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试）"""
        if cls._instance is not None:
            cls._instance._client = None
            cls._instance._tools = None
        cls._instance = None

    async def initialize(self, servers: Optional[List[str]] = None):
        """
        初始化 MCP 客户端

        Args:
            servers: 要启用的服务列表，默认启用所有
        """
        if self._client is not None:
            app_logger.warning("MCP 客户端已初始化，跳过")
            return

        # 过滤出请求的服务
        servers = servers or list(self.SERVER_CONFIGS.keys())
        configs = {k: v for k, v in self.SERVER_CONFIGS.items() if k in servers}

        app_logger.info(f"初始化 MCP 服务: {list(configs.keys())}")

        # 创建客户端
        self._client = MultiServerMCPClient(configs)

        # 预加载工具
        try:
            self._tools = await self._client.get_tools()
            app_logger.info(f"已加载 {len(self._tools)} 个 MCP 工具")
        except Exception as e:
            app_logger.warning(f"预加载工具失败: {e}")
            self._tools = []

    async def close(self):
        """关闭客户端"""
        if self._client:
            self._client = None
            self._tools = None
            app_logger.info("MCP 客户端已关闭")

    async def get_tools(self) -> List:
        """
        获取所有 MCP 工具

        Returns:
            LangChain 工具列表
        """
        if self._client is None:
            raise RuntimeError("MCP 客户端未初始化，请先调用 initialize()")

        if self._tools:
            return self._tools

        self._tools = await self._client.get_tools()
        return self._tools


async def get_mcp_client(servers: Optional[List[str]] = None) -> MCPClientManager:
    """获取 MCP 客户端管理器实例"""
    return await MCPClientManager.get_instance(servers)

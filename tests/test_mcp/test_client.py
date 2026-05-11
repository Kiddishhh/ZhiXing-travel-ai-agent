"""MCP 客户端管理器测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.mcp_core.client import MCPClientManager, get_mcp_client


@pytest.fixture(autouse=True)
def reset_manager():
    """每个测试前后重置单例"""
    MCPClientManager.reset_instance()
    yield
    MCPClientManager.reset_instance()


@pytest.fixture
def mock_mcp_client():
    """模拟 MultiServerMCPClient"""
    with patch("app.mcp_core.client.MultiServerMCPClient") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_tools = AsyncMock(return_value=["tool1", "tool2"])
        mock_cls.return_value = mock_instance
        yield mock_cls


class TestMCPClientManagerSingleton:
    """单例模式测试"""

    @pytest.mark.asyncio
    async def test_get_instance_returns_same_object(self, mock_mcp_client):
        """两次调用 get_instance 返回同一实例"""
        instance1 = await MCPClientManager.get_instance()
        instance2 = await MCPClientManager.get_instance()
        assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_get_instance_initializes_client(self, mock_mcp_client):
        """首次获取实例会初始化 MultiServerMCPClient"""
        await MCPClientManager.get_instance()
        mock_mcp_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_call_skips_initialize(self, mock_mcp_client):
        """第二次调用不再创建 MultiServerMCPClient"""
        await MCPClientManager.get_instance()
        call_count = mock_mcp_client.call_count
        await MCPClientManager.get_instance()
        assert mock_mcp_client.call_count == call_count


class TestServerFiltering:
    """服务过滤测试"""

    @pytest.mark.asyncio
    async def test_specific_servers(self, mock_mcp_client):
        """指定 servers 参数只初始化选定服务"""
        await MCPClientManager.get_instance(servers=["weather"])
        configs_passed = mock_mcp_client.call_args[0][0]
        assert "weather" in configs_passed
        assert "search" not in configs_passed

    @pytest.mark.asyncio
    async def test_unknown_server_skipped(self, mock_mcp_client):
        """传入未知服务名不会崩溃，configs 为空"""
        await MCPClientManager.get_instance(servers=["nonexistent"])
        configs_passed = mock_mcp_client.call_args[0][0]
        assert configs_passed == {}

    @pytest.mark.asyncio
    async def test_default_all_servers(self, mock_mcp_client):
        """不传 servers 参数默认启用所有服务"""
        await MCPClientManager.get_instance()
        configs_passed = mock_mcp_client.call_args[0][0]
        assert len(configs_passed) == len(MCPClientManager.SERVER_CONFIGS)


class TestGetTools:
    """get_tools 方法测试"""

    @pytest.mark.asyncio
    async def test_get_tools_without_initialize_raises(self):
        """未初始化直接调用 get_tools 抛出 RuntimeError"""
        manager = MCPClientManager()
        with pytest.raises(RuntimeError, match="未初始化"):
            await manager.get_tools()

    @pytest.mark.asyncio
    async def test_get_tools_returns_cached(self, mock_mcp_client):
        """get_tools 返回缓存工具列表"""
        manager = await MCPClientManager.get_instance()
        tools = await manager.get_tools()
        assert tools == ["tool1", "tool2"]
        mock_instance = mock_mcp_client.return_value
        assert mock_instance.get_tools.call_count == 1


class TestResetInstance:
    """reset_instance 测试"""

    @pytest.mark.asyncio
    async def test_reset_allows_new_instance(self, mock_mcp_client):
        """reset_instance 后可以获取新实例"""
        instance1 = await MCPClientManager.get_instance()
        MCPClientManager.reset_instance()
        instance2 = await MCPClientManager.get_instance()
        assert instance1 is not instance2

    @pytest.mark.asyncio
    async def test_reset_before_any_instance(self):
        """未初始化时调用 reset 不会报错"""
        MCPClientManager.reset_instance()
        assert MCPClientManager._instance is None


class TestClose:
    """close 方法测试"""

    @pytest.mark.asyncio
    async def test_close_clears_state(self, mock_mcp_client):
        """close 后 client 和 tools 被清空"""
        manager = await MCPClientManager.get_instance()
        await manager.close()
        assert manager._client is None
        assert manager._tools is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_mcp_client):
        """重复 close 不报错"""
        manager = await MCPClientManager.get_instance()
        await manager.close()
        await manager.close()


class TestServerConfigs:
    """配置完整性测试（不需要 mock）"""

    def test_config_has_six_servers(self):
        """SERVER_CONFIGS 包含 6 个服务"""
        configs = MCPClientManager.SERVER_CONFIGS
        assert len(configs) == 6
        assert "weather" in configs
        assert "search" in configs
        assert "amap" in configs
        assert "12306-mcp" in configs
        assert "VariFlight-Aviation" in configs
        assert "aigohotel-mcp" in configs

    def test_stdio_servers_have_correct_args(self):
        """自建服务的 args 指向正确的模块路径"""
        weather_args = MCPClientManager.SERVER_CONFIGS["weather"]["args"]
        search_args = MCPClientManager.SERVER_CONFIGS["search"]["args"]
        assert weather_args == ["-m", "app.mcp_core.servers.weather_server"]
        assert search_args == ["-m", "app.mcp_core.servers.search_server"]

    def test_stdio_servers_have_env(self):
        """自建服务包含 ENV_VARS"""
        assert "env" in MCPClientManager.SERVER_CONFIGS["weather"]
        assert "env" in MCPClientManager.SERVER_CONFIGS["search"]

    def test_external_services_have_url(self):
        """外部服务配置包含 url"""
        assert "url" in MCPClientManager.SERVER_CONFIGS["amap"]
        assert "url" in MCPClientManager.SERVER_CONFIGS["12306-mcp"]
        assert "url" in MCPClientManager.SERVER_CONFIGS["VariFlight-Aviation"]
        assert "url" in MCPClientManager.SERVER_CONFIGS["aigohotel-mcp"]


class TestGetMCPClient:
    """便捷函数测试"""

    @pytest.mark.asyncio
    async def test_get_mcp_client_returns_manager(self, mock_mcp_client):
        """get_mcp_client 返回 MCPClientManager 实例"""
        client = await get_mcp_client()
        assert isinstance(client, MCPClientManager)

    @pytest.mark.asyncio
    async def test_get_mcp_client_passes_servers(self, mock_mcp_client):
        """get_mcp_client 透传 servers 参数"""
        client = await get_mcp_client(servers=["weather"])
        assert isinstance(client, MCPClientManager)
        configs_passed = mock_mcp_client.call_args[0][0]
        assert list(configs_passed.keys()) == ["weather"]

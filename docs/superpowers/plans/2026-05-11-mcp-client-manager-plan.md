# MCP 客户端管理器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建 `MCPClientManager` 单例管理器，统一管理 6 个 MCP 服务连接。

**Architecture:** 在 `app/mcp_core/client.py` 中实现单例模式的 `MCPClientManager`，封装 `MultiServerMCPClient`，通过 `get_mcp_client()` 便捷函数对外暴露。配置走 `os.getenv`，新增 `variflight_api_key` 到 `Settings`。

**Tech Stack:** Python 3.13, `langchain-mcp-adapters==0.2.1`, `fastmcp==2.13.1`, pytest + unittest.mock

**File structure:**
- Modify: `app/config.py` — 新增 1 个字段
- Create: `app/mcp_core/client.py` — MCPClientManager + get_mcp_client
- Create: `tests/test_mcp/test_client.py` — 6 个单测

---

### Task 1: 在 Settings 中新增 variflight_api_key

**Files:**
- Modify: `app/config.py:60`

- [ ] **Step 1: 添加字段**

在 `app/config.py` 的 MCP 配置段（第 60 行 `tavily_api_key` 之后）插入：

```python
variflight_api_key: str = Field(default="", alias="VARIFLIGHT_API_KEY")
```

修改后的 MCP 配置段：

```python
# ============== MCP 服务配置 ==============
amap_api_key: str = Field(default="", alias="AMAP_API_KEY")
tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
variflight_api_key: str = Field(default="", alias="VARIFLIGHT_API_KEY")
```

- [ ] **Step 2: 验证语法**

```bash
python -c "from app.config import Settings; s = Settings(); print(s.variflight_api_key)"
```
Expected: 输出 `sk-wvB3gIb2DoTh36Cw00JlIsd78q_CjVD7UE85ax9sHjQ`（从 .env 读取）

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat: add variflight_api_key to Settings"
```

---

### Task 2: 创建 MCPClientManager

**Files:**
- Create: `app/mcp_core/client.py`

- [ ] **Step 1: 检查父目录存在**

```bash
ls "D:/AI agent/知行智能旅游规划助手/app/mcp_core/__init__.py"
```
Expected: 文件存在

- [ ] **Step 2: 写入 client.py**

```python
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
    _tools: Optional[List] = None
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
    async def get_instance(cls, servers: List[str] = None) -> 'MCPClientManager':
        """获取单例实例"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize(servers=servers)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例（用于测试）"""
        cls._instance = None

    async def initialize(self, servers: List[str] = None):
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


async def get_mcp_client(servers: List[str] = None) -> MCPClientManager:
    """获取 MCP 客户端管理器实例"""
    return await MCPClientManager.get_instance(servers)
```

- [ ] **Step 3: 验证语法**

```bash
python -c "import ast; ast.parse(open('app/mcp_core/client.py', encoding='utf-8').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: 验证模块可导入**

```bash
python -c "from app.mcp_core.client import MCPClientManager, get_mcp_client; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 5: Commit**

```bash
git add app/mcp_core/client.py
git commit -m "feat: add MCPClientManager with 6 server configs"
```

---

### Task 3: 编写测试

**Files:**
- Create: `tests/test_mcp/test_client.py`

- [ ] **Step 1: 检查测试目录存在**

```bash
ls "D:/AI agent/知行智能旅游规划助手/tests/test_mcp/__init__.py"
```
Expected: 文件存在

- [ ] **Step 2: 写入 test_client.py**

```python
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

    def test_get_tools_without_initialize_raises(self):
        """未初始化直接调用 get_tools 抛出 RuntimeError"""
        manager = MCPClientManager()
        with pytest.raises(RuntimeError, match="未初始化"):
            # 注意：get_tools 是 async 方法，需要用 asyncio.run
            import asyncio
            asyncio.run(manager.get_tools())

    @pytest.mark.asyncio
    async def test_get_tools_returns_cached(self, mock_mcp_client):
        """get_tools 返回缓存工具列表"""
        manager = await MCPClientManager.get_instance()
        tools = await manager.get_tools()
        assert tools == ["tool1", "tool2"]
        # Mock 的 get_tools 只被底层调用 1 次（initialize 里），缓存后不再调用
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
        await manager.close()  # 不应抛异常


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
```

- [ ] **Step 3: 运行全部测试**

```bash
python -m pytest tests/test_mcp/test_client.py -v
```
Expected: 全部 PASS（17 个测试）

- [ ] **Step 4: Commit**

```bash
git add tests/test_mcp/test_client.py
git commit -m "test: add MCP client manager unit tests"
```

---

### Task 4: 最终验证

- [ ] **Step 1: 运行完整 test_mcp 套件确保无回归**

```bash
python -m pytest tests/test_mcp/ -v
```
Expected: 所有已有测试 + 新测试全部通过

- [ ] **Step 2: 运行全局语法检查**

```bash
python -c "import ast, sys, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').rglob('*.py')]"
```
Expected: 无错误输出

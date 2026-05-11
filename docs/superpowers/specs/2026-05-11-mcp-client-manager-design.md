# MCP 客户端管理器设计

## 概述

在 `app/mcp_core/client.py` 中创建 `MCPClientManager`，统一管理所有 MCP 服务连接（自建 stdio 服务 + 外部 HTTP 服务），提供单例 + 延迟初始化 + 容错加载。

## 服务范围

| 服务 | 传输方式 | API Key | 状态 |
|------|---------|---------|------|
| weather | stdio | 无（内部用 AMAP_API_KEY） | 启用 |
| search | stdio | 无（内部用 TAVILY_API_KEY） | 启用 |
| amap | http | AMAP_API_KEY | 启用 |
| 12306-mcp | streamable_http | 无 | 启用 |
| VariFlight-Aviation | streamable_http | VARIFLIGHT_API_KEY | 启用 |
| aigohotel-mcp | streamable_http | 占位符 | 占位 |

## 文件布局

```
app/mcp_core/
├── __init__.py
├── client.py          ← 新增
└── servers/
    ├── __init__.py
    ├── weather_server.py
    └── search_server.py
```

## 核心组件

### MCPClientManager

单例模式，封装 `langchain_mcp_adapters.client.MultiServerMCPClient`。

**类成员：**
- `_instance` / `_client` / `_tools` — 单例状态
- `_lock = asyncio.Lock()` — 线程安全
- `PROJECT_ROOT` — 项目根目录，用于 stdio 命令
- `ENV_VARS` — 追加 PYTHONPATH 的环境变量
- `SERVER_CONFIGS` — 6 个服务配置字典

**关键方法：**

| 方法 | 说明 |
|------|------|
| `get_instance(servers)` | 类方法，获取/创建单例，首次调用触发 initialize |
| `initialize(servers)` | 创建 MultiServerMCPClient，逐个加载工具，失败 warn |
| `get_tools()` | 返回缓存工具列表；未初始化抛 RuntimeError |
| `close()` | 幂等关闭，清空 client/tools |
| `reset_instance()` | 类方法，清空单例（测试用） |

### get_mcp_client()

模块级便捷函数：

```python
async def get_mcp_client(servers: List[str] = None) -> MCPClientManager:
    return await MCPClientManager.get_instance(servers)
```

## 容错策略

`initialize()` 中 per-service 容错：

```
创建 MultiServerMCPClient(configs)
  → get_tools() 整体拉取
    ├ 成功 → 缓存 tools，log 数量
    └ 失败 → app_logger.warning，tools = []
```

个别服务连接失败不阻塞整体初始化，`get_tools()` 返回当前可用工具。

## 配置读取

使用 `os.getenv()` + `load_dotenv()`，与现有 MCP server 风格一致。不引入 pydantic-settings 依赖。

### Settings 补充

`app/config.py` 新增 1 个字段：

- `variflight_api_key: str = Field(default="", alias="VARIFLIGHT_API_KEY")`

## 调用方式

```python
from app.mcp_core.client import get_mcp_client

# agent 或 API 层
client = await get_mcp_client()
tools = await client.get_tools()
```

## 测试计划

文件：`tests/test_mcp/test_client.py`

| 测试 | 验证点 |
|------|--------|
| 单例模式 | 两次 `get_instance()` 返回同一实例 |
| 指定服务 | `servers=["weather"]` 只初始化 weather |
| 未知服务名 | 传入不在 SERVER_CONFIGS 中的名字 → 跳过 |
| 未初始化报错 | 绕过单例直接调 `get_tools()` → RuntimeError |
| reset_instance | 重置后可重新初始化 |
| 配置完整性 | stdio 服务的 args 路径指向正确模块 |

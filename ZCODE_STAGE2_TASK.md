# 🤖 ZCode (GLM-5.3) 任务卡：阶段一（网关层 MCP 自动装配与 Provider 工厂）

请作为 AgentBridge 核心工程师，实现项目第二阶段的核心架构能力：**网关层 MCP 自动装配与 Provider 工厂**。

---

## 🎯 详细开发需求

### 1. `src/agentbridge/providers/factory.py`
* 实现工厂函数：`create_upstream_client(config: UpstreamConfig) -> UpstreamClient`
* 路由逻辑：
  - 若 `config.name.lower() == "glm"` 或 `config.base_url` 包含 `"bigmodel.cn"` 或 `"z.ai"`：返回 `GLMProvider(config)`。
  - 否则：返回标准 `UpstreamClient(config)`。
* 在 `src/agentbridge/providers/__init__.py` 中导出 `create_upstream_client` 并加入 `__all__`。

### 2. `src/agentbridge/config.py`
* 在 `BridgeConfig` 中新增字段：
  - `enable_mcp: bool = False`
  - `mcp_config_path: str | None = None`
* 在 `BridgeConfig.from_env()` 中支持读取环境变量：
  - `AGENTBRIDGE_ENABLE_MCP`: 解析 "true", "1", "yes" 为 True，否则 False。
  - `AGENTBRIDGE_MCP_CONFIG`: 读取配置路径字符串，若无则为 None。

### 3. `src/agentbridge/server.py`
* 引用 `create_upstream_client` 替换原有直接实例化的 `UpstreamClient`。
* 引入 `MCPProxy`（来自 `agentbridge.mcp.proxy`）。
* 在 `create_app(config: BridgeConfig) -> FastAPI` 中：
  - 若 `config.enable_mcp` 为 True，实例化 `mcp_proxy = MCPProxy(config_path=config.mcp_config_path)`，否则为 None。
  - 在 `lifespan(app: FastAPI)` 中：
    - 启动时：若 `mcp_proxy` 存在，执行 `await mcp_proxy.start_all()`。
    - 关闭时：若 `mcp_proxy` 存在，执行 `await mcp_proxy.stop_all()`；随后执行 `await client.close()`。
  - 挂载 `app.state.client = client` 与 `app.state.mcp_proxy = mcp_proxy`。
  - 新增管理端点 `GET /v1/mcp/tools`：
    - 若启用 MCP，返回 `{"status": "ok", "tools": mcp_proxy.get_anthropic_tools(), "count": len(mcp_proxy.tools)}`。
    - 若未启用，返回 `{"status": "disabled", "tools": [], "count": 0}`。
  - 在 `POST /v1/messages` 中：
    - 若 `mcp_proxy` 存在，调用 `mcp_proxy.get_anthropic_tools()` 获取当前活跃工具列表。
    - 将未重复的 MCP 工具自动合并注入到 `payload["tools"]` 中，确保上游模型感知到可用 MCP 工具。

### 4. `src/agentbridge/cli.py`
* 在 `start` 子命令中添加 CLI 参数：
  - `--enable-mcp`: `action="store_true"`, 帮助描述 `"Automatically discover and relay local MCP servers"`
  - `--mcp-config`: `default=None`, 帮助描述 `"Path to custom mcp.json config file"`
  - `--provider`: `choices=["auto", "openai", "glm"]`, `default="auto"`, 帮助描述 `"Upstream provider type (default: auto)"`
* 在 `main()` 中将这些参数同步写入 `config`。若 `--provider glm`，显式设置 `config.upstream.name = "glm"`。
* 优化 `print_startup_banner`：在终端展示 MCP Relay 启用状态与 Provider 类型。

### 5. 编写单元测试
* 创建 `tests/test_factory.py`：测试各种 URL 与名称输入下的 Provider 自动路由与 fallback。
* 创建 `tests/test_server_mcp.py`：使用 `httpx.AsyncClient(app=app, base_url="http://test")` 或 `starlette.testclient.TestClient` 测试：
  - 未启用 MCP 时 `/v1/mcp/tools` 返回 disabled。
  - 启用 MCP 时端点响应及 `/v1/messages` 工具自动合并注入行为。

---

## 📋 交付与质量规范
1. 运行 `pytest` 确保既有 106 个测试与新增测试 100% 通过。
2. 运行 `ruff check .` 与 `ruff format --check .` 确保无任何语法与风格警告。

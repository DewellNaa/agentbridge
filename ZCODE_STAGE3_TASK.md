# 🤖 ZCode (GLM-5.3) 任务卡：阶段二（Rich 终端实时流量与 Token 监控看板）

请作为 AgentBridge 核心工程师，实现项目第二阶段的核心能力：**Rich 终端实时流量、推理 Token 与性能指标监控（Telemetry & Dashboard）**。

---

## 🎯 详细开发需求

### 1. `src/agentbridge/telemetry/metrics.py` (新增模块)
* 定义数据结构：
  - `RequestMetric`:
    - `request_id: str`
    - `timestamp: float`
    - `client_agent: str` (从请求头 `User-Agent` 获取，若无默认为 `"Unknown"`)
    - `model_requested: str` (客户端请求的模型名，如 `claude-3-7-sonnet-20250219`)
    - `model_upstream: str` (转译后发往上游的模型名，如 `GLM-5.3-Flash`)
    - `prompt_tokens: int = 0`
    - `completion_tokens: int = 0`
    - `reasoning_tokens: int = 0` (专门记录 DeepSeek-R1 / GLM-5.3 的 thinking 推理 token)
    - `total_tokens: int = 0`
    - `duration_ms: float = 0.0`
    - `ttft_ms: float = 0.0` (Time to First Token，首 token 延迟)
    - `tools_called: list[str] = field(default_factory=list)` (记录调用的 tool 名称列表)
    - `status_code: int = 200`
    - `is_stream: bool = True`
    - `error_message: str | None = None`
* 定义管理类 `TelemetryManager`:
  - 维护内存环形队列（最近 `max_history=100` 条请求）。
  - 维护聚合指标：`total_requests`, `successful_requests`, `failed_requests`, `total_prompt_tokens`, `total_completion_tokens`, `total_reasoning_tokens`, `start_time`。
  - 方法：
    - `start_request(req_id: str, model_req: str, model_upstream: str, client_agent: str, is_stream: bool) -> RequestMetric`
    - `record_first_token(metric: RequestMetric)` (记录 `ttft_ms`)
    - `record_tokens(metric: RequestMetric, prompt: int = 0, completion: int = 0, reasoning: int = 0)`
    - `record_tool_call(metric: RequestMetric, tool_name: str)`
    - `finish_request(metric: RequestMetric, status_code: int = 200, error_message: str | None = None)`
    - `get_summary() -> dict`: 返回全局概览（包括运行时间 `uptime_seconds`、请求量、各类 Token 总计、平均耗时、最近请求简报等）。
  - 导出于 `src/agentbridge/telemetry/__init__.py`。

### 2. `src/agentbridge/telemetry/display.py` (新增模块)
* 使用 `rich` 库提供终端可视化展示工具函数：
  - `log_request_metric(metric: RequestMetric, console: Console | None = None)`：
    - 在每次请求完成时，输出美观紧凑的单行或卡片式 Rich 日志：
      例如：`[bold cyan]19:45:01[/bold cyan] [green]200 OK[/green] | [yellow]claude-3-7-sonnet[/yellow] -> [bold magenta]GLM-5.3-Flash[/bold magenta] | in: 1,240, out: 450 ([blue]think: 180[/blue]) | [dim]TTFT: 120ms, Tot: 1.8s (250 tok/s)[/dim] | tools: [mcp__bash]`
  - `render_dashboard_panel(manager: TelemetryManager) -> Panel`：
    - 生成用于 CLI 查看当前汇总指标的 Rich Panel 汇总看板。

### 3. `src/agentbridge/server.py` 集成
* 在 `create_app(config: BridgeConfig) -> FastAPI` 中：
  - 初始化 `telemetry = TelemetryManager()` 并挂载到 `app.state.telemetry = telemetry`。
  - 注册监控端点 `GET /v1/telemetry`：返回 `telemetry.get_summary()`。
  - 在 `POST /v1/messages` 中：
    - 创建并追踪 `RequestMetric`。
    - 非流式（non-stream）：记录完成耗时、从 upstream_resp 中提取 usage（含 prompt_tokens, completion_tokens 等），并调用 `log_request_metric`。
    - 流式（stream_generator）：
      - 监控首个 chunk 到达时间更新 `ttft_ms`。
      - 统计输出 chunk 文本长度及 reasoning_content 增长（估算或从 usage chunk 解析）。
      - 捕获流式出现的 `tool_use` 名称。
      - 流结束或抛出异常时执行 `finish_request` 并调用 `log_request_metric`。

### 4. `src/agentbridge/cli.py` 增强
* 在 `start_parser` 中新增参数：
  - `--no-telemetry-log`: `action="store_true"`, 允许用户关闭控制台每次请求的详细 metrics 输出。
* 新增子命令 `agentbridge status` 或 `agentbridge metrics`：
  - 支持向运行中的本地网关 `http://{host}:{port}/v1/telemetry` 发送查询并在终端渲染好看的看板。

### 5. 编写单元测试
* 创建 `tests/test_telemetry.py`：
  - 测试 `TelemetryManager` 记录首 token、累加各类 token、环形历史、聚合数据统计。
  - 测试 `GET /v1/telemetry` API 端点响应。
  - 测试非流式与流式请求下的指标追踪与异常记录。

---

## 📋 质量规范
1. 所有代码符合严格类型提示与 docstring。
2. `pytest` 全量测试通过，保持零报错零破坏现有功能。
3. `ruff check .` 与 `ruff format --check .` 100% 通过。

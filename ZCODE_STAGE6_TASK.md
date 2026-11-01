# 🤖 ZCode (GLM-5.3) 任务卡：阶段六（多 Key 轮换与 429/5xx 智能故障转移 Failover）

请作为 AgentBridge 核心工程师，实现企业级高可用核心能力：**多 API Key 负载轮换与 429/5xx 智能故障转移（Multi-Key Load Balancing & Fault-Tolerant Failover）**。

---

## 🎯 详细开发需求

### 1. `src/agentbridge/routing/pool.py` (新增模块)
实现高可用上游连接池与故障转移路由引擎：
* 数据结构 `UpstreamEndpoint`:
  ```python
  @dataclass
  class UpstreamEndpoint:
      name: str
      client: UpstreamClient
      weight: int = 1
      is_healthy: bool = True
      failure_count: int = 0
      cooldown_until: float = 0.0  # 冷却截止时间戳（perf_counter 或 time）
      cooldown_seconds: float = 30.0  # 故障冷却时长
  ```
* 类 `FailoverRouter`:
  - 初始化：接收 `endpoints: list[UpstreamEndpoint]`，以及 `strategy: str = "priority"`（`"priority"` 或 `"round_robin"`）。
  - 触发转移的错误类型：
    - HTTP 状态码：`429` (Rate Limited), `500` (Server Error), `502` (Bad Gateway), `503` (Service Unavailable), `504` (Gateway Timeout), `402` (Payment/Quota Exhausted)。
    - 网络超时与连接失败（`httpx.ConnectError`, `httpx.ConnectTimeout`, `httpx.ReadTimeout`）。
  - 方法：
    - `get_next_endpoint() -> UpstreamEndpoint`:
      - 检查并恢复已过冷却期的 endpoint。
      - 根据策略（priority 始终取首个健康节点；round_robin 在所有健康节点间轮转）返回活跃 endpoint。
      - 若所有节点均不可用，抛出 `AllUpstreamsFailedError`。
    - `async complete(request: dict) -> tuple[dict, str]`:
      - 尝试在当前 endpoint 上完成请求。
      - 若遇到触发 failover 的异常：
        - 标记该 endpoint 进入冷却 (`cooldown_until = now + cooldown_seconds`)，`failure_count += 1`。
        - 记录警告日志（例如 `[yellow]Failover:[/yellow] {endpoint.name} failed ({error}), failing over to next upstream...`）。
        - 尝试下一个可用 endpoint，直到成功或所有节点穷尽。
      - 返回 `(upstream_response, endpoint_name)`。
    - `async stream_complete(request: dict) -> AsyncIterator[dict]`:
      - 流式请求故障转移：
        - 若在建立连接或产生首个 chunk 之前发生 429/5xx/连接超时：无缝切换到备用 endpoint 重试！
        - 一旦已开始向客户端 yield 数据（首 chunk 已发送），后续异常记录错误并退出（避免流污染）。
    - 状态探针方法：
      - `get_pool_status() -> list[dict]`: 返回所有 endpoint 的名称、健康状态、失败次数、剩余冷却时间。

### 2. `src/agentbridge/config.py` 配置扩展
* 支持配置多个 Key 与备用上游：
  - `UpstreamConfig` 支持：
    - `api_keys: list[str] = Field(default_factory=list)`：支持逗号分隔解析。
    - `fallback_url: str | None = None`
    - `fallback_key: str | None = None`
    - `fallback_model: str | None = None`
    - `failover_strategy: str = "priority"`  # "priority" 或 "round_robin"
  - `from_env()`：
    - 支持环境变量 `AGENTBRIDGE_UPSTREAM_KEYS`（逗号分隔读取）。
    - 支持 `AGENTBRIDGE_FALLBACK_URL`, `AGENTBRIDGE_FALLBACK_KEY`, `AGENTBRIDGE_FALLBACK_MODEL`。
    - 支持 `AGENTBRIDGE_FAILOVER_STRATEGY`。

### 3. `src/agentbridge/server.py` 与工厂集成
* 工厂或服务端接入 `FailoverRouter`：
  - 若配置了多个 `api_keys`，为每个 key 构建同配置 endpoint 并命名为 `f"key-{i+1}"`。
  - 若配置了 `fallback_url`，构建备用 endpoint 并命名为 `"fallback"`。
  - 在 `create_app` 中使用 `FailoverRouter` 封装上游请求调用。
  - 提供健康检查端点 `GET /v1/failover/status`：返回各节点健康情况。

### 4. `src/agentbridge/cli.py` 命令行扩展
* `start_parser` 新增参数：
  - `--upstream-keys`: 接收逗号分隔的多个 API Key（如 `key1,key2,key3`）。
  - `--fallback-url`, `--fallback-key`, `--fallback-model`：备用上游配置。
  - `--strategy`: `choices=["priority", "round_robin"]`, `default="priority"`。
* 启动横幅（`print_startup_banner`）增强：
  - 若启用了多 Key 或 Fallback，展示 `[bold cyan]Failover Pool:[/bold cyan] Active (Strategy: <strategy>, Endpoints: <N>)`。

### 5. 编写完整单元测试
* 创建 `tests/test_failover.py`:
  - 测试多 Key round-robin 均匀轮转分配。
  - 测试 429 / 503 报错时自动触发 failover 切换至下一个健康 endpoint。
  - 测试全部节点失败时抛出 `AllUpstreamsFailedError`。
  - 测试冷却时间到期后的自动恢复行为。
  - 测试流式请求在首 chunk 发送前的优雅 failover 重试。
  - 测试 `GET /v1/failover/status` 端点响应。

---

## 📋 提交与验证规范
1. `pytest` 全量测试通过（需达到 340+ 测试，零回归）。
2. 运行 `ruff check .` 与 `ruff format --check .` 100% 保持规范。

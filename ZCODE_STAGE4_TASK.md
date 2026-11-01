# 🤖 ZCode (GLM-5.3) 任务卡：阶段四（智能体一键启动包装器 agentbridge run 与 IDE 配置生成）

请作为 AgentBridge 核心工程师，实现用户急需的核心体验特性：**智能体一键启动包装器 (`agentbridge run`) 与 IDE 零配置生成器 (`agentbridge config`)**。

---

## 🎯 详细开发需求

### 1. `src/agentbridge/launcher.py` (新增模块)
实现网关探测、自动拉起与子进程智能体环境注入：
* `check_server_healthy(host: str = "127.0.0.1", port: int = 8080, timeout: float = 0.8) -> bool`
  - 使用 `httpx.get(f"http://{host}:{port}/health", timeout=timeout)` 判断现有网关是否存活（状态码 200 即存活）。
* `wait_for_gateway(host: str = "127.0.0.1", port: int = 8080, timeout: float = 8.0, interval: float = 0.25) -> bool`
  - 轮询等待网关直到 `/health` 就绪，超时返回 False。
* `get_agent_env(agent: str, host: str, port: int, existing_env: dict | None = None) -> dict[str, str]`
  - 返回注入了网关所需环境变量的字典：
    - `ANTHROPIC_BASE_URL`: 对 `claude` 为 `f"http://{host}:{port}"`；对 `aider` 为 `f"http://{host}:{port}/v1"`；通用默认 `f"http://{host}:{port}"`。
    - `ANTHROPIC_API_KEY`: `"agentbridge"`。
    - 继承原系统 `os.environ`。
* `resolve_executable(cmd: str) -> str | None`
  - 使用 `shutil.which(cmd)` 解析可执行文件路径（在 Windows 上自动解析 `.cmd` / `.exe`，如 `claude.cmd`）。
* `run_agent(agent_args: list[str], host: str = "127.0.0.1", port: int = 8080, auto_start: bool = True, upstream_args: list[str] | None = None) -> int`
  - 若 `check_server_healthy` 为 False 且 `auto_start=True`：
    - 通过后台子进程 `subprocess.Popen([sys.executable, "-m", "agentbridge.cli", "start", "--host", host, "--port", str(port)] + (upstream_args or []))` 启动网关。
    - 调用 `wait_for_gateway` 等待启动成功；若失败打印友好错误并返回 1。
    - 记录 `spawned_gateway = True`。
  - 提取目标执行程序（例如 `claude` 或 `aider`），通过 `resolve_executable` 查找。若未找到，给出清晰提示：“未找到该命令，请先安装该智能体（如 npm install -g @anthropic-ai/claude-code 或 pip install aider-chat）”，返回 127。
  - 构造环境变量并使用 `subprocess.run(resolved_args, env=env)` 执行（保持 stdio 透传交互）。
  - 在 `finally` 块中：如果网关是由本次命令拉起的 (`spawned_gateway=True`)，则优雅终止该后台网关进程并释放资源。
  - 返回子进程的 exit_code。

### 2. `src/agentbridge/templates.py` (新增模块)
提供 Cursor、VS Code Cline、Continue 等 IDE 的一键配置模板：
* `generate_ide_config(target: str, host: str = "127.0.0.1", port: int = 8080) -> str`
  - `cursor`: 生成包含 Claude 3.7 / 3.5 映射与 OpenAI 兼容端点的配置指南与 `.cursorrules` 示例。
  - `cline`: 生成 `.cline/mcp_settings.json` 或 VS Code settings.json 覆盖片段。
  - `continue`: 生成 `~/.continue/config.json` 中的 `models` 列表 JSON 配置片段。
  - 若 target 未知，返回支持的 target 列表。

### 3. `src/agentbridge/cli.py` 命令行扩展
* 新增子命令 `agentbridge run`:
  ```bash
  agentbridge run <agent> [agent_args...]
  # 示例:
  # agentbridge run claude
  # agentbridge run claude -- -p "explain main.py"
  # agentbridge run aider -- --model claude-3-5-sonnet-20241022
  ```
  - 支持参数：
    - `agent`: 必填，目标智能体名称（如 `claude`、`aider`）或自定义命令。
    - `args`: 可变参数，使用 `nargs=argparse.REMAINDER` 收集传递给目标 agent 的所有参数。
    - `--host`, `--port`: 网关地址与端口（默认 `127.0.0.1:8080`）。
    - `--no-auto-start`: 禁止自动拉起后台网关（仅连现有网关）。
* 新增子命令 `agentbridge config`:
  ```bash
  agentbridge config <target>
  # target 可选: cursor, cline, continue
  ```
  - 在终端高亮输出对应 IDE 的一键配置代码或 JSON 片段。

### 4. 编写完整单元测试
* 创建 `tests/test_launcher.py`:
  - 覆盖 `check_server_healthy` 的成功与超时失败分支。
  - 覆盖 `get_agent_env` 对 `claude`、`aider` 及自定义命令的环境变量生成。
  - 覆盖 `wait_for_gateway` 的成功与超时。
  - 覆盖 `run_agent` 在已有网关、自动启动网关、以及命令不存在 (exit 127) 时的行为（使用 mock subprocess 与 httpx）。
* 创建 `tests/test_templates.py`:
  - 测试 `generate_ide_config` 对 `cursor`, `cline`, `continue` 的输出内容与格式合法性。
* 编写 CLI 命令行测试（在 `tests/test_cli.py` 或加入现有测试）验证 `run` 与 `config` 解析。

---

## 📋 提交与验证规范
1. `pytest` 全量测试通过（需达到 170+ 测试，零回归）。
2. `ruff check .` 与 `ruff format --check .` 100% 保持规范。

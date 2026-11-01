# 🤖 ZCode (GLM-5.3) 任务卡：阶段五（Ollama 本地大模型一键自发现与零配置接入）

请作为 AgentBridge 核心工程师，实现第四大方向能力：**Ollama 本地大模型智能自发现、代码模型评分优选与零配置一键接入 (`agentbridge start --ollama`)**。

---

## 🎯 详细开发需求

### 1. `src/agentbridge/discovery/ollama.py` (新增模块)
实现与本地 Ollama 实例的探测与模型自发现协议：
* 数据结构：
  ```python
  @dataclass
  class OllamaModelInfo:
      name: str
      size_bytes: int = 0
      parameter_size: str = ""  # 如 "7B", "14B", "32B"
      quantization_level: str = ""
      modified_at: str = ""
      is_coding_model: bool = False
      score: int = 0  # 针对编程任务的适配评分
  ```
* 编程模型评分优选算法 `score_ollama_model(model_name: str) -> int`:
  - 核心编程关键词加权：
    - 包含 `"qwen2.5-coder"`: +100
    - 包含 `"deepseek-coder"`: +90
    - 包含 `"deepseek-r1"`: +85 (高推理能力)
    - 包含 `"codellama"`: +70
    - 包含 `"starcoder"`: +65
    - 包含 `"code"`: +50
    - 包含 `"llama3"` / `"qwen2"` / `"mistral"`: +30
  - 参数量加权（例如 32b > 14b > 7b > 1.5b）：
    - 包含 `"32b"` 或 `"70b"`: +15
    - 包含 `"14b"` 或 `"8b"`: +10
    - 包含 `"7b"`: +5
  - 返回综合得分。
* 探测函数：
  - `check_ollama_running(host: str = "127.0.0.1", port: int = 11434, timeout: float = 1.0) -> bool`
    - 向 `http://{host}:{port}/api/version` 发送请求，判断服务是否存活。
  - `discover_ollama_models(host: str = "127.0.0.1", port: int = 11434, timeout: float = 2.0) -> list[OllamaModelInfo]`
    - 请求 `http://{host}:{port}/api/tags`，解析返回的 `models` 列表并计算各自的编程评分。
  - `select_best_ollama_model(models: list[OllamaModelInfo]) -> OllamaModelInfo | None`
    - 自动返回得分最高的推荐模型；若无任何模型返回 None。

### 2. `src/agentbridge/providers/ollama.py` (新增模块)
针对 Ollama OpenAI-compatible 协议的专用适配器：
* 类 `OllamaProvider(UpstreamClient)`:
  - 默认 base_url 为 `http://127.0.0.1:11434/v1`，默认 api_key 为 `"ollama"`。
  - 特殊处理：
    - 针对 `deepseek-r1` 系列，兼容 Ollama 返回的流式 thinking content（包含 `<think>...</think>` 标签或 `reasoning_content` delta）。
    - 自动去除不支持的 OpenAI 专属请求参数（如某些采样参数），避免 Ollama 报 400。
* 在 `src/agentbridge/providers/factory.py` 中更新 `create_upstream_client`:
  - 若 `config.name.lower() == "ollama"` 或 `config.base_url` 包含 `":11434"` 或 `"ollama"`：
    返回 `OllamaProvider(config)`。
  - 导出并加入 `providers/__init__.py`。

### 3. `src/agentbridge/config.py` 配置支持
* `BridgeConfig` 与 `UpstreamConfig` 新增支持：
  - 支持配置属性 `ollama_mode: bool = False`。
  - 支持从环境变量 `AGENTBRIDGE_OLLAMA=true` 或 `OLLAMA_HOST` 中读取默认配置。

### 4. `src/agentbridge/cli.py` 命令行扩展
* 在 `start_parser` 中新增 `--ollama`:
  - `nargs="?"`, `const="auto"`, `default=None`, 帮助描述 `"Connect to local Ollama with auto-discovery (optionally specify model name, e.g. --ollama qwen2.5-coder:14b)"`。
  - 在 `start` 执行逻辑中：
    - 若指定 `--ollama`:
      - 检查本地 Ollama 是否运行；若未运行，友好提示并退出（退出码 1）：
        `[red]Error:[/red] Ollama is not running at http://127.0.0.1:11434. Please start Ollama or download it from https://ollama.com`
      - 若 `--ollama auto`（用户只传了 `--ollama`）：
        自动调用 `discover_ollama_models` 并用 `select_best_ollama_model` 挑选最佳编程模型；
        若模型库为空，友好提示 `ollama pull qwen2.5-coder`。
      - 若用户指定了模型名（如 `--ollama deepseek-r1:14b`）：使用该模型。
      - 自动将 `config.upstream.base_url` 设为 `http://127.0.0.1:11434/v1`，`config.upstream.name = "ollama"`，`config.upstream.default_model` 设为选中的模型，`config.upstream.api_key = "ollama"`。
* 启动横幅（`print_startup_banner`）增强：
  - 当为 Ollama 模式时，展示 `[bold cyan]Ollama Auto-Discovery:[/bold cyan] Active (Target: <model>)` 以及探测到的本地模型总数。
* 新增子命令 `agentbridge ollama`:
  - `agentbridge ollama`: 快速列出本地所有 Ollama 模型、显示其参数量、量化级别以及编程适配评分，高亮推荐模型。

### 5. 编写单元测试
* 创建 `tests/test_ollama_discovery.py`:
  - 测试各种模型名称评分算法（qwen2.5-coder > deepseek-coder > deepseek-r1 > 普通模型）。
  - 测试 `check_ollama_running` 存活与超时失败。
  - 测试 `discover_ollama_models` 解析响应与字段填充。
  - 测试 `select_best_ollama_model` 在空列表与多模型下的选优。
* 创建 `tests/test_ollama_provider.py`:
  - 测试 `OllamaProvider` 的默认请求头、base_url 和 payload 转换。
  - 测试 `create_upstream_client` 自动路由到 `OllamaProvider`。
* 在 CLI 测试中补充 `--ollama` 参数解析与 banner 格式。

---

## 📋 提交与验证规范
1. 运行 `pytest` 确保 250+ 测试全部通过，零回归。
2. 保持 `ruff check .` 与 `ruff format --check .` 100% 干净。

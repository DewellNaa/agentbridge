# 🤖 ZCode (GLM-5.3) 任务卡：阶段七（上下文历史智能压缩 Context Budget Squeezer）

请作为 AgentBridge 核心工程师，实现项目第三大核心能力：**长文本历史智能压缩与上下文预算阀门（Context Budget Squeezer & Smart Sliding Window）**。

---

## 🎯 详细开发需求

### 1. `src/agentbridge/context/compactor.py` (新增模块)
实现上下文体积评估与多阶段无损/低损压缩引擎：
* 估算与分析函数：
  - `estimate_tokens(text: str) -> int`: 基于 ~4 字符/token（中文/代码加权）的高效无依赖估算器。
  - `estimate_payload_tokens(messages: list[dict], system: Any = None) -> int`: 遍历所有 message content 块与 system 提示估算总 token 数。
* 核心类 `ContextCompactor`:
  - 初始化参数：
    - `max_context_tokens: int | None = None` (最大上下文预算，None 表示不设硬限)
    - `preserve_recent_turns: int = 4` (最新 N 轮对话绝对保留原文，不作裁剪)
    - `max_tool_result_chars: int = 2000` (旧轮次中超长工具输出的最大保留字符数)
    - `preserve_first_turn: bool = True` (永久保留第一轮用户需求，确保智能体不忘最初目标)
  - 方法 `compact(messages: list[dict], system: Any = None) -> tuple[list[dict], dict]`:
    - 若 `max_context_tokens` 为 None，或者 `estimate_payload_tokens <= max_context_tokens`：原样返回 `(messages, {"compressed": False, "saved_tokens": 0})`。
    - 若超过预算，按两阶段递进压缩：
      - **第一阶段（工具结果折叠 Tool Result Truncation）**：
        - 检查距离当前轮次超过 `preserve_recent_turns` 的所有旧轮次。
        - 对其中的 `tool_result` 内容块或超长文本（字符数 > `max_tool_result_chars`）：
          保留前部（如 `head=800` 字符）和后部（如 `tail=800` 字符），中间插入明确的折叠提示：
          `\n... [AgentBridge: Truncated {diff} chars of older tool output (saved ~{saved} tokens)] ...\n`。
        - 重新计算 token 数，若已满足 `<= max_context_tokens`，立即返回。
      - **第二阶段（滑动窗口轮次折叠 Sliding Window Fold）**：
        - 若第一阶段后仍超标：
          保持第 1 轮（用户原始输入）、系统提示与最近 `preserve_recent_turns` 轮次；
          对中间积压的多余轮次进行合并摘要折叠，替换为紧凑的占位上下文块：
          `{"role": "user", "content": f"[AgentBridge: Compressed {N} intermediate conversation turns to preserve context budget]"}` 与对应 assistant 回执。
    - 返回 `(compacted_messages, stats)`，其中 stats 包含 `compressed: bool`、`original_tokens`、`compacted_tokens`、`saved_tokens`、`truncated_tools`、`folded_turns`。

### 2. `src/agentbridge/config.py` 配置扩展
* `BridgeConfig` 新增支持：
  - `max_context_tokens: int | None = None`
  - `compact_context: bool = False`
  - `preserve_recent_turns: int = 4`
  - `max_tool_result_chars: int = 2000`
* `from_env()`：
  - 读取 `AGENTBRIDGE_MAX_CONTEXT_TOKENS` (int 或 None)
  - 读取 `AGENTBRIDGE_COMPACT_CONTEXT` ("true"/"1" 解析为 True)
  - 读取 `AGENTBRIDGE_PRESERVE_RECENT_TURNS` (int, default 4)
  - 读取 `AGENTBRIDGE_MAX_TOOL_RESULT_CHARS` (int, default 2000)

### 3. `src/agentbridge/server.py` 集成
* 在 `create_app` 中根据配置实例化 `ContextCompactor`。
* 在 `POST /v1/messages` 收到请求时：
  - 若启用了压缩（`config.compact_context` 或 `config.max_context_tokens` 非空）：
    在调用 `adapter.convert_request` 之前，先执行 `compactor.compact(payload.get("messages", []), payload.get("system"))`。
    将压缩后的 messages 写回 payload。
  - 将 `stats` 中的 `saved_tokens` 记录到当前请求的 Telemetry Metric 中。
  - 控制台日志（Rich Log）中若有节省 token，追加显示 `[dim](saved {stats['saved_tokens']} ctx tokens)[/dim]`。

### 4. `src/agentbridge/cli.py` 命令行扩展
* `start_parser` 新增参数：
  - `--max-context-tokens`: `type=int, default=None`, 帮助描述 `"Maximum context budget in tokens (auto-enables compaction)"`
  - `--compact-context`: `action="store_true"`, 帮助描述 `"Enable smart context compaction for long agent sessions"`
  - `--preserve-recent-turns`: `type=int, default=4`, 帮助描述 `"Number of recent conversation turns to keep untouched"`
  - `--max-tool-chars`: `type=int, default=2000`, 帮助描述 `"Maximum characters per tool output in older turns"`
* 启动横幅（`print_startup_banner`）增强：
  - 若启用压缩，展示 `[bold cyan]Context Compactor:[/bold cyan] Active (Budget: <N> tokens, Preserve: <M> turns)`。

### 5. 编写单元测试
* 创建 `tests/test_compactor.py`:
  - 测试 `estimate_tokens` 针对纯英文、代码与混合文本的估算合理性。
  - 测试未超预算时不发生任何修改。
  - 测试第一阶段：仅对旧轮次中的超长 tool_result 进行首尾截断折叠，最近 N 轮和短文本原样保留。
  - 测试第二阶段：极端超长多轮对话下，保留首轮与末尾 N 轮，中间轮次折叠并满足预算上限。
  - 测试非破坏性（不原地污染传入的 raw dict/list）。
  - 测试与 `server.py` `/v1/messages` 的集成，验证 saved_tokens 写入 telemetry。

---

## 📋 提交与验证规范
1. 运行 `pytest` 确保达到 400+ 测试，零回归。
2. 保持 `ruff check .` 与 `ruff format --check .` 100% 规范。

# 🚀 AgentBridge

[![CI](https://github.com/DewellNaa/agentbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/DewellNaa/agentbridge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Ultra-lightweight, zero-config local AI coding agent gateway & protocol bridge.**  
> Seamlessly connect **Claude Code**, **Cursor**, and **Aider** to **DeepSeek-R1/V3**, **GLM-5.3**, and **Ollama** with 100% SSE stream & thinking block fidelity.

---

## ✨ Features

- **🚀 One-Shot Agent Launcher**: Run `agentbridge run claude` or `agentbridge run aider` — automatically manages gateway lifecycle and environment variable injection with zero configuration.
- **🛠️ Instant IDE Configuration**: Run `agentbridge config <cursor|cline|continue>` to generate drop-in configuration snippets.
- **⚡ Zero-Config Local Gateway**: Instant startup via `agentbridge start`, zero database required, lightweight async architecture.
- **🧠 Full Thinking/Reasoning Translation**: Bridges DeepSeek-R1 / GLM-5.3 `delta.reasoning_content` to Claude 3.7 `thinking` content blocks and back in real-time.
- **🛡️ Prompt Caching Fault Tolerance**: Automatically detects and sanitizes Anthropic `cache_control: {"type": "ephemeral"}` metadata recursively across system prompts, messages, and tools, preventing `400 Bad Request` errors on standard upstream LLMs.
- **🔄 Multi-Key Load Balancing & Failover Pool**: Distribute requests evenly across multiple API keys (`round_robin`) and automatically fail over (`priority`) on `429` rate limits, `503` server overloads, or connection timeouts.
- **✂️ Context Budget Squeezer**: Set `--max-context-tokens` to prevent context length overflows and control token costs by automatically folding giant older `tool_result` outputs while preserving initial user goals and recent turns.
- **🦙 Ollama Local LLM Auto-Discovery**: Run `agentbridge start --ollama` to automatically discover local Ollama models, benchmark and score coding candidates (`qwen2.5-coder`, `deepseek-r1`), and enable 100% offline, privacy-first AI coding.
- **🇨🇳 Native Zhipu BigModel / GLM-5.3 Adapter**: First-class support for BigModel authentication, GLM-5.3 / GLM-5.3-Flash model aliasing, thinking effort levels, and SSE streaming normalization.
- **🔌 Local MCP (Model Context Protocol) Relay**: Auto-discovers `~/.claude/mcp.json` or custom configurations, manages local stdio MCP subprocesses, namespaces tools (`mcp__<server>__<tool>`), and dynamically multiplexes them into agent requests.
- **📊 Live Terminal Telemetry**: Real-time TTFT (Time to First Token) tracker, reasoning token throughput counter, and Rich dashboard via `agentbridge metrics`.
- **🛠️ Robust Tool Calling**: Automatically converts Anthropic `tool_use` / `tool_result` into OpenAI function calling and stitches streaming partial JSON chunks without crashing.
- **📡 Strict SSE Stream Conformance**: Zero dropped frames, accurate `message_start` -> `content_block_start` -> `content_block_delta` -> `content_block_stop` -> `message_stop` state sequences.
- **🔄 Model Aliasing & Routing**: Transparently alias `claude-3-5-sonnet-20241022` or `claude-3-7-sonnet-20250219` to your favorite low-cost or local models.

---

## 🚀 Quickstart

### 1. Installation

#### One-Line Installer
```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/DewellNaa/agentbridge/main/scripts/install.sh | bash

# Windows PowerShell
irm https://raw.githubusercontent.com/DewellNaa/agentbridge/main/scripts/install.ps1 | iex
```

#### Via uv or pip
```bash
pip install -U "git+https://github.com/DewellNaa/agentbridge.git"
```

### 2. Zero-Config One-Shot Launch (`agentbridge run`)

No manual environment variable configuration required! `agentbridge run` automatically detects whether the gateway is running, launches a background gateway if needed, and injects the proper Anthropic endpoints:

```bash
# Launch Claude Code directly
agentbridge run claude

# Pass arguments directly to Claude Code
agentbridge run claude -- -p "explain src/agentbridge/launcher.py"

# Launch Aider directly
agentbridge run aider -- --model claude-3-5-sonnet-20241022
```

### 3. Or Start the Standalone Gateway (`agentbridge start`)

#### Option A: With DeepSeek / OpenAI-compatible upstream
```bash
agentbridge start \
  --upstream-url "https://api.deepseek.com/v1" \
  --upstream-key "your-deepseek-api-key" \
  --model "deepseek-chat" \
  --port 8080
```

#### Option B: With Zhipu BigModel / GLM-5.3 (Auto-Detected)
```bash
agentbridge start \
  --upstream-url "https://open.bigmodel.cn/api/paas/v4" \
  --upstream-key "your-bigmodel-api-key" \
  --model "glm-5.3-flash" \
  --enable-mcp \
  --port 8080
```

#### Option C: With Local Ollama (Zero-Config Auto-Discovery)
```bash
# Auto-detects local Ollama instance and selects best coding model (e.g. qwen2.5-coder, deepseek-r1)
agentbridge start --ollama

# Or specify an exact local Ollama model
agentbridge start --ollama deepseek-r1:14b

# Inspect installed Ollama models & coding scores
agentbridge ollama
```

### 4. Connect Your Coding Agent

#### 🤖 Claude Code
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
export ANTHROPIC_API_KEY="agentbridge"

claude
```

#### 🧑‍💻 Aider
```bash
aider --anthropic-api-base "http://127.0.0.1:8080/v1" --model claude-3-5-sonnet-20241022
```

---

## 📊 Terminal Telemetry & Live Metrics

AgentBridge tracks real-time reasoning metrics, token throughput, and Time to First Token (TTFT).

### Real-Time Request Stream Log
When the gateway runs, each request is logged with rich visual formatting:
```text
19:45:01 200 OK | claude-3-7-sonnet -> GLM-5.3-Flash | in: 1,420, out: 850 (think: 320) | TTFT: 110ms, Tot: 1.8s (472 tok/s) | tools: [mcp__bash]
```

### Live Metrics Dashboard
Inspect the live status, token consumption, and latency metrics of your running gateway:
```bash
agentbridge metrics
# or
agentbridge status --port 8080
```

---

## 🔌 Model Context Protocol (MCP) Relay

AgentBridge automatically discovers MCP servers configured in `~/.claude/mcp.json` or custom configurations:

```bash
agentbridge start --enable-mcp --mcp-config /path/to/custom_mcp.json
```

- Inspect loaded MCP tools via `curl http://127.0.0.1:8080/v1/mcp/tools`.
- Tools are namespaced as `mcp__<server>__<tool>` to prevent collision with native agent tools.

---

## 🛠️ Instant IDE Configuration (`agentbridge config`)

Generate ready-to-paste configurations for Cursor, VS Code Cline, or Continue with one command:

```bash
# Print Cursor setup guide & .cursorrules snippet
agentbridge config cursor

# Print VS Code Cline settings snippet
agentbridge config cline

# Print Continue ~/.continue/config.json models block
agentbridge config continue
```

---

## 🛡️ Multi-Key Load Balancing & Failover Pool

Prevent session interruptions from `429 Too Many Requests` or `503 Service Unavailable`:

```bash
# Rotate requests across multiple API keys in round-robin
agentbridge start \
  --upstream-url "https://api.deepseek.com/v1" \
  --upstream-keys "key1,key2,key3" \
  --strategy round_robin

# Automatically fail over to a backup provider if the primary endpoint fails
agentbridge start \
  --upstream-url "https://api.deepseek.com/v1" \
  --upstream-key "deepseek-key" \
  --fallback-url "https://open.bigmodel.cn/api/paas/v4" \
  --fallback-key "glm-key" \
  --fallback-model "glm-5.3-flash"
```

Check the health and failover status of all pool endpoints in real time:
```bash
curl http://127.0.0.1:8080/v1/failover/status
```

---

## ✂️ Context Budget Squeezer

Keep long agent sessions running smoothly without blowing token budgets or context limits:

```bash
# Cap context budget at 32k tokens and fold giant tool outputs in older turns
agentbridge start \
  --max-context-tokens 32000 \
  --preserve-recent-turns 4 \
  --max-tool-chars 1500
```

- **Phase 1 (Tool Folding)**: Truncates older giant tool outputs (e.g. massive file reads or compiler logs) to head/tail previews while keeping recent turns pristine.
- **Phase 2 (Turn Folding)**: Seamlessly folds intermediate conversation turns into concise markers while permanently safeguarding the initial user goal and system instructions.

---

## 🏛️ Architecture

```
[ Coding Agent: Claude Code / Cursor / Aider ]
                     │  (Anthropic Messages API v1 / SSE)
                     ▼
             ┌───────────────┐
             │  AgentBridge  │
             │  FastAPI/ASGI │
             └───────┬───────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
[Thinking Block  [Tool Call Delta  [Model Routing
 Transformer]     Stitcher]         & Failover]
    │                │                │
    └────────────────┼────────────────┘
                     ▼  (OpenAI / v1 / SSE Stream)
  [ Upstream: DeepSeek-R1 / GLM-5.3 / Ollama / vLLM ]
```

---

## 🧪 Testing & Verification

Run the test suite with `pytest`:

```bash
pytest -v
```

Lint with `ruff`:

```bash
ruff check .
ruff format --check .
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

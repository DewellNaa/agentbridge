# 🚀 AgentBridge

[![CI](https://github.com/DewellNaa/agentbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/DewellNaa/agentbridge/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Ultra-lightweight, zero-config local AI coding agent gateway & protocol bridge.**  
> Seamlessly connect **Claude Code**, **Cursor**, and **Aider** to **DeepSeek-R1/V3**, **GLM-5.3**, and **Ollama** with 100% SSE stream & thinking block fidelity.

---

## ✨ Features

- **⚡ Zero-Config Local Gateway**: Instant startup via `agentbridge start`, zero database required, lightweight async architecture.
- **🧠 Full Thinking/Reasoning Translation**: Bridges DeepSeek-R1 / GLM-5.3 `delta.reasoning_content` to Claude 3.7 `thinking` content blocks and back in real-time.
- **🛡️ Prompt Caching Fault Tolerance**: Automatically detects and sanitizes Anthropic `cache_control: {"type": "ephemeral"}` metadata recursively across system prompts, messages, and tools, preventing `400 Bad Request` errors on standard upstream LLMs.
- **🇨🇳 Native Zhipu BigModel / GLM-5.3 Adapter**: First-class support for BigModel authentication, GLM-5.3 / GLM-5.3-Flash model aliasing, thinking effort levels, and SSE streaming normalization.
- **🔌 Local MCP (Model Context Protocol) Relay**: Auto-discovers `~/.claude/mcp.json` or custom configurations, manages local stdio MCP subprocesses, namespaces tools (`mcp__<server>__<tool>`), and dynamically multiplexes them into agent requests.
- **🛠️ Robust Tool Calling**: Automatically converts Anthropic `tool_use` / `tool_result` into OpenAI function calling and stitches streaming partial JSON chunks without crashing.
- **📡 Strict SSE Stream Conformance**: Zero dropped frames, accurate `message_start` -> `content_block_start` -> `content_block_delta` -> `content_block_stop` -> `message_stop` state sequences.
- **🔄 Model Aliasing & Routing**: Transparently alias `claude-3-5-sonnet-20241022` or `claude-3-7-sonnet-20250219` to your favorite low-cost or local models.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/DewellNaa/agentbridge.git
cd agentbridge
pip install -e .
```

### 2. Start the Gateway

#### Option A: With DeepSeek / OpenAI-compatible upstream
```bash
agentbridge start \
  --upstream-url "https://api.deepseek.com/v1" \
  --upstream-key "your-deepseek-api-key" \
  --model "deepseek-chat" \
  --port 8080
```

#### Option B: With Zhipu BigModel / GLM-5.3
```bash
agentbridge start \
  --upstream-url "https://open.bigmodel.cn/api/paas/v4" \
  --upstream-key "your-bigmodel-api-key" \
  --model "glm-5.3-flash" \
  --port 8080
```

### 3. Connect Your Coding Agent

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

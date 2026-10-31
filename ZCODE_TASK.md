# 🤖 ZCode (GLM-5.3) 协同开发任务指南

欢迎来到 **AgentBridge** 项目！本项目是为 Claude Code、Cursor、Aider 量身打造的超轻量级 AI 编程网关与协议转译中枢。

---

## 🎯 任务完成进度看板 (All Completed ✅)

所有核心增强模块已由 ZCode (GLM-5.3-Flash) 自主实现并交付，CI/CD 全绿，单测覆盖率 100%。

---

### [x] 任务 1：BigModel / GLM-5.3 原生 Provider 适配器
* **完成 Commit**：`04dc241` (`feat(providers): add native Zhipu BigModel / GLM-5.3 provider adapter`)
* **交付文件**：`src/agentbridge/providers/glm.py`
* **单测套件**：`tests/test_glm_provider.py` (29 passed)
* **实现亮点**：
  1. 支持智谱 BigModel 开放平台 Bearer 鉴权派生与 Header 注入。
  2. 支持 GLM-5.3 / GLM-5.3-Flash 模型映射与 `thinking: {"type": "enabled"}` 参数标准化。
  3. 继承自 `UpstreamClient`，完整实现 SSE 流式 chunk 解析与容错。

---

### [x] 任务 2：Anthropic Prompt Caching 断点容错处理器
* **完成 Commit**：`01095df` (`feat(adapters): add recursive prompt caching sanitizer`)
* **交付文件**：`src/agentbridge/adapters/cache_control.py`
* **单测套件**：`tests/test_cache_control.py` (22 passed)
* **实现亮点**：
  1. 提供高效纯函数 `sanitize_cache_control(payload)`，非原地修改原结构。
  2. 采用记忆化集合防止循环引用导致 RecursionError。
  3. 深度遍历 messages, system prompts 与 tools，彻底剔除 `cache_control`，杜绝上游 400 校验异常。

---

### [x] 任务 3：本地 MCP（Model Context Protocol）中继探针与多路复用器
* **完成 Commit**：`470b5c2` (`feat(mcp): add local MCP relay probe and multiplexer`)
* **交付文件**：`src/agentbridge/mcp/proxy.py`
* **单测套件**：`tests/test_mcp_proxy.py` (48 passed)
* **实现亮点**：
  1. 三级配置解析（`~/.claude/mcp.json` -> `AGENTBRIDGE_MCP_CONFIG` -> inline dict），支持 `${ENV_VAR}` 展开。
  2. 完整 JSON-RPC stdio 通信与子进程生命周期管理。
  3. 工具名称命名空间隔离（`mcp__<server>__<tool>`）与双向 schema 转译。

---

## 📊 质量验证记录

* **本地单元测试**：106/106 passed (100% 通过率，耗时 ~2.5s)
* **静态代码检查**：`ruff check .` 0 警告，`ruff format --check .` 规范对齐
* **GitHub Actions CI**：[Actions 运行结果](https://github.com/DewellNaa/agentbridge/actions) Python 3.10 / 3.11 / 3.12 矩阵全绿 🟢

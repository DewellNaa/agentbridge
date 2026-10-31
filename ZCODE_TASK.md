# 🤖 ZCode (GLM-5.3) 协同开发任务指南

欢迎来到 **AgentBridge** 项目！本项目是为 Claude Code、Cursor、Aider 量身打造的超轻量级 AI 编程网关与协议转译中枢。

---

## 🎯 任务目标：实现三大核心增强模块

请作为核心开发者，在 `D:\agentbridge` 代码库中依次实现以下三个关键能力，并编写对应的 `pytest` 单元测试。

---

### 任务 1：BigModel / GLM-5.3 原生 Provider 适配器

* **目标文件**：`src/agentbridge/providers/glm.py`
* **功能需求**：
  1. 支持智谱开放平台（BigModel / Z.ai）的鉴权方式与请求头（包括从客户端 token 或标准 API Key 派生 Authorization）。
  2. 支持 GLM-5.3 / GLM-5.3-Flash 的专属参数（如 `thinking: {"type": "enabled"}` 或 `do_sample` 等控制项）。
  3. 继承自 `UpstreamClient` 或实现通用 provider 协议。
* **单测文件**：`tests/test_glm_provider.py`
  * 覆盖鉴权头注入、模型名称映射、流式 chunk 解析。

---

### 任务 2：Anthropic Prompt Caching 断点容错处理器

* **目标文件**：`src/agentbridge/adapters/cache_control.py`
* **功能需求**：
  1. Claude Code 极其频繁地在 messages 和 tools 中插入 `cache_control: {"type": "ephemeral"}`。
  2. 很多上游厂商（如未开启 Prompt Cache 的端点）接收到带 `cache_control` 的字段会直接报错 `400 Bad Request: Extra inputs are not permitted`。
  3. 提供 `sanitize_cache_control(payload: dict) -> dict` 函数，递归清洗所有消息块、系统提示和工具中的 `cache_control` 标记。
* **单测文件**：`tests/test_cache_control.py`
  * 测试在嵌套消息、系统提示、Tools 中彻底清除 `cache_control` 且不改变其他原始字段。

---

### 任务 3：本地 MCP（Model Context Protocol）中继探针

* **目标文件**：`src/agentbridge/mcp/proxy.py`
* **功能需求**：
  1. 允许 AgentBridge 自动读取本地 `~/.claude/mcp.json` 或环境变量中的 MCP 服务配置。
  2. 在网关层将 MCP 工具列表动态注入到 Anthropic Tools 声明中，使任何上游模型都能调用本地 MCP 工具。
* **单测文件**：`tests/test_mcp_proxy.py`
  * 测试 MCP 工具定义转换与注册。

---

## 📋 提交与验证规范

1. 完成代码后，必须通过以下两项本地质量检验：
   ```bash
   ruff check .
   pytest -v
   ```
2. 保持代码注释完整、类型提示（Type Hints）严谨，遵守 `pyproject.toml` 中的规范。

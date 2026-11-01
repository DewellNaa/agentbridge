# 📢 AgentBridge Community Promotion & Launch Kit

This guide contains ready-to-publish announcements and launch copies for technical communities (V2EX, Reddit, Twitter/X, Zhihu, Juejin) and submission snippets for Awesome lists.

---

## 📌 Ready-to-Copy Links
- **GitHub Repository**: https://github.com/DewellNaa/agentbridge
- **Latest Release**: https://github.com/DewellNaa/agentbridge/releases/latest
- **Actions CI**: https://github.com/DewellNaa/agentbridge/actions

---

## 🇨🇳 国内技术社区（V2EX / 掘金 / 知乎）

### 推荐标题
> **受够了 Claude Code 昂贵账单和 429 报错？写了个超轻量本地网关 AgentBridge，一条命令用上 DeepSeek/本地 Ollama**

### 核心亮点速记
1. `agentbridge run claude`：免配环境变量，自动拉起网关并在退出时回收；
2. `agentbridge start --ollama`：自动探测本地 Ollama 实例并智能评选最强代码模型（如 `qwen2.5-coder:32b`）；
3. 原生转译：Prompt Cache 断点递归清洗 + Reasoning 思考流双向对齐；
4. 容灾与压缩：多 Key 轮询故障转移（防 429）+ 上下文预算两阶段智能压缩；
5. 质量保障：425 项单测 100% CI 全绿。

---

## 🌐 海外技术社区 (Reddit r/LocalLLaMA & Twitter/X)

### Reddit Title
> **[Project] AgentBridge: Run Claude Code with local Ollama models (Qwen2.5-Coder / DeepSeek-R1) or DeepSeek-V3 with zero-config**

### Twitter / X Hook
> Sick of insane Anthropic API bills or 429 rate limits when using Claude Code?
> Check out AgentBridge: an ultra-lightweight local protocol bridge connecting Claude Code, Cursor & Aider to DeepSeek-R1/V3 and local @ollama!
> ⭐ https://github.com/DewellNaa/agentbridge

---

## 🎯 Awesome Lists 提交候选
- `deepseek-ai/awesome-deepseek-integration`
- `hesreallyhim/awesome-claude-code`
- `jqueryscript/awesome-claude-code`

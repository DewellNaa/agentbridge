"""Local model discovery package for AgentBridge."""

from agentbridge.discovery.ollama import (
    OllamaModelInfo,
    check_ollama_running,
    discover_ollama_models,
    parse_ollama_host,
    score_ollama_model,
    select_best_ollama_model,
)

__all__ = [
    "OllamaModelInfo",
    "check_ollama_running",
    "discover_ollama_models",
    "parse_ollama_host",
    "score_ollama_model",
    "select_best_ollama_model",
]

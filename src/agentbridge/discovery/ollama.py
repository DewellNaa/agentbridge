"""Local Ollama instance detection and coding-model auto-discovery.

Talks to the Ollama native HTTP API (``/api/version``, ``/api/tags``) to
detect a running local instance, enumerate the installed models and rank
them by how well they fit coding-agent workloads so ``agentbridge start
--ollama`` can wire the bridge up with zero configuration.
"""

import re
from dataclasses import dataclass

import httpx

DEFAULT_OLLAMA_HOST = "127.0.0.1"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_BASE_URL = f"http://{DEFAULT_OLLAMA_HOST}:{DEFAULT_OLLAMA_PORT}"

# (keyword, weight) pairs for coding-capable model families. A model name
# accumulates the weight of every keyword it contains, so "qwen2.5-coder"
# also picks up the generic "code" bonus.
_CODING_KEYWORDS = (
    ("qwen2.5-coder", 100),
    ("deepseek-coder", 90),
    ("deepseek-r1", 85),
    ("codellama", 70),
    ("starcoder", 65),
    ("code", 50),
    ("llama3", 30),
    ("qwen2", 30),
    ("mistral", 30),
)

# (keywords, weight) pairs for parameter-count tiers; bigger models score
# higher because coding agents reward capability over footprint.
_PARAM_SIZE_KEYWORDS = (
    (("32b", "70b"), 15),
    (("14b", "8b"), 10),
    (("7b",), 5),
)

# Substrings that mark a model as purpose-built for code (used for the
# ``is_coding_model`` flag; general-purpose families stay False).
_CODING_MODEL_MARKERS = ("qwen2.5-coder", "deepseek-coder", "codellama", "starcoder")

_HOST_PORT_RE = re.compile(r"^(?P<host>[^:/]+)(?::(?P<port>\d+))?$")


@dataclass
class OllamaModelInfo:
    """Metadata about a locally installed Ollama model."""

    name: str
    size_bytes: int = 0
    parameter_size: str = ""  # e.g. "7B", "14B", "32B"
    quantization_level: str = ""
    modified_at: str = ""
    is_coding_model: bool = False
    score: int = 0  # fitness score for coding tasks


def score_ollama_model(model_name: str) -> int:
    """Score how well ``model_name`` fits coding-agent workloads."""
    name = model_name.lower()
    score = 0
    for keyword, weight in _CODING_KEYWORDS:
        if keyword in name:
            score += weight
    for keywords, weight in _PARAM_SIZE_KEYWORDS:
        if any(keyword in name for keyword in keywords):
            score += weight
    return score


def parse_ollama_host(value: str) -> tuple[str, int]:
    """Parse an ``OLLAMA_HOST``-style value into ``(host, port)``.

    Accepts ``"127.0.0.1:11434"``, bare hosts (``"localhost"``) and
    ``http://`` URLs (``"http://192.168.1.5:11434"``); missing parts fall
    back to the defaults.
    """
    value = (value or "").strip()
    if not value:
        return DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_PORT
    if "://" in value:
        stripped = value.split("://", 1)[1]
    else:
        stripped = value
    match = _HOST_PORT_RE.match(stripped)
    if not match:
        return DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_PORT
    host = match.group("host") or DEFAULT_OLLAMA_HOST
    port = int(match.group("port")) if match.group("port") else DEFAULT_OLLAMA_PORT
    return host, port


def check_ollama_running(
    host: str = DEFAULT_OLLAMA_HOST,
    port: int = DEFAULT_OLLAMA_PORT,
    timeout: float = 1.0,
) -> bool:
    """Return ``True`` when an Ollama instance answers at ``host:port``."""
    url = f"http://{host}:{port}/api/version"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code < 400


def discover_ollama_models(
    host: str = DEFAULT_OLLAMA_HOST,
    port: int = DEFAULT_OLLAMA_PORT,
    timeout: float = 2.0,
) -> list[OllamaModelInfo]:
    """List local models from ``/api/tags`` with coding scores attached."""
    url = f"http://{host}:{port}/api/tags"
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    models: list[OllamaModelInfo] = []
    for entry in payload.get("models") or []:
        details = entry.get("details") or {}
        name = entry.get("name") or entry.get("model") or ""
        if not name:
            continue
        lower = name.lower()
        is_coding = any(marker in lower for marker in _CODING_MODEL_MARKERS) or (
            "code" in lower
        )
        models.append(
            OllamaModelInfo(
                name=name,
                size_bytes=int(entry.get("size") or 0),
                parameter_size=details.get("parameter_size", ""),
                quantization_level=details.get("quantization_level", ""),
                modified_at=entry.get("modified_at", ""),
                is_coding_model=is_coding,
                score=score_ollama_model(name),
            )
        )
    return models


def select_best_ollama_model(
    models: list[OllamaModelInfo],
) -> OllamaModelInfo | None:
    """Return the highest-scoring model, or ``None`` when none exist."""
    if not models:
        return None
    return max(models, key=lambda model: (model.score, model.size_bytes))

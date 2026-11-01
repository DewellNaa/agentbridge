"""Tests for local Ollama detection and coding-model auto-discovery."""

import httpx
import pytest

from agentbridge.discovery.ollama import (
    OllamaModelInfo,
    check_ollama_running,
    discover_ollama_models,
    parse_ollama_host,
    score_ollama_model,
    select_best_ollama_model,
)


class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "http://test"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# Scoring algorithm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("qwen2.5-coder:7b", 100 + 50 + 30 + 5),
        ("deepseek-coder:7b", 90 + 50 + 5),
        ("deepseek-r1:14b", 85 + 10),
        ("codellama:13b", 70 + 50),
        ("starcoder2:15b", 65 + 50),
        ("codegemma:7b", 50 + 5),
        ("llama3.1:8b", 30 + 10),
        ("qwen2:14b", 30 + 10),
        ("mistral:7b", 30 + 5),
        ("gemma2:9b", 0),
    ],
)
def test_score_ollama_model_keyword_weights(model_name, expected):
    assert score_ollama_model(model_name) == expected


@pytest.mark.parametrize(
    ("model_name", "bonus"),
    [
        ("qwen2.5-coder:32b", 15),
        ("deepseek-coder-v2:70b", 15),
        ("qwen2.5-coder:14b", 10),
        ("llama3:8b", 10),
        ("qwen2.5-coder:7b", 5),
    ],
)
def test_score_ollama_model_parameter_bonus(model_name, bonus):
    # Base keyword score without the parameter suffix.
    base = score_ollama_model(model_name.rsplit(":", 1)[0])
    assert score_ollama_model(model_name) == base + bonus


def test_score_is_case_insensitive():
    assert score_ollama_model("Qwen2.5-Coder:7B") == score_ollama_model(
        "qwen2.5-coder:7b"
    )


def test_coding_models_score_above_reasoning_above_plain():
    coding = score_ollama_model("qwen2.5-coder:14b")
    reasoning = score_ollama_model("deepseek-r1:14b")
    plain = score_ollama_model("llama3:8b")
    assert coding > reasoning > plain >= 0


# ---------------------------------------------------------------------------
# Host parsing
# ---------------------------------------------------------------------------


def test_parse_ollama_host_default():
    assert parse_ollama_host("") == ("127.0.0.1", 11434)
    assert parse_ollama_host("127.0.0.1:11434") == ("127.0.0.1", 11434)


def test_parse_ollama_host_custom():
    assert parse_ollama_host("192.168.1.5:11500") == ("192.168.1.5", 11500)
    assert parse_ollama_host("http://localhost:11434") == ("localhost", 11434)
    assert parse_ollama_host("myserver") == ("myserver", 11434)


# ---------------------------------------------------------------------------
# check_ollama_running
# ---------------------------------------------------------------------------


def test_check_ollama_running_true_when_alive(monkeypatch):
    captured = {}

    def fake_get(url, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(200, {"version": "0.5.7"})

    monkeypatch.setattr(
        "agentbridge.discovery.ollama.httpx.get", fake_get, raising=True
    )
    assert check_ollama_running("127.0.0.1", 11434, timeout=0.5) is True
    assert captured["url"] == "http://127.0.0.1:11434/api/version"
    assert captured["timeout"] == 0.5


def test_check_ollama_running_false_on_timeout(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("agentbridge.discovery.ollama.httpx.get", fake_get)
    assert check_ollama_running() is False


def test_check_ollama_running_false_on_http_error(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("agentbridge.discovery.ollama.httpx.get", fake_get)
    assert check_ollama_running() is False


def test_check_ollama_running_false_on_error_status(monkeypatch):
    monkeypatch.setattr(
        "agentbridge.discovery.ollama.httpx.get",
        lambda url, timeout: FakeResponse(500),
    )
    assert check_ollama_running() is False


# ---------------------------------------------------------------------------
# discover_ollama_models
# ---------------------------------------------------------------------------


TAGS_PAYLOAD = {
    "models": [
        {
            "name": "qwen2.5-coder:14b",
            "model": "qwen2.5-coder:14b",
            "modified_at": "2026-01-15T10:00:00Z",
            "size": 8_989_452_339,
            "details": {
                "family": "qwen2",
                "parameter_size": "14.8B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "deepseek-r1:7b",
            "size": 4_700_000_000,
            "details": {
                "parameter_size": "7.6B",
                "quantization_level": "Q4_0",
            },
        },
        {
            "name": "llama3:8b",
            "size": 4_700_000_000,
            "details": {},
        },
    ]
}


def test_discover_ollama_models_parses_fields(monkeypatch):
    monkeypatch.setattr(
        "agentbridge.discovery.ollama.httpx.get",
        lambda url, timeout: FakeResponse(200, TAGS_PAYLOAD),
    )
    models = discover_ollama_models("127.0.0.1", 11434)

    assert [m.name for m in models] == [
        "qwen2.5-coder:14b",
        "deepseek-r1:7b",
        "llama3:8b",
    ]
    coder = models[0]
    assert coder.size_bytes == 8_989_452_339
    assert coder.parameter_size == "14.8B"
    assert coder.quantization_level == "Q4_K_M"
    assert coder.modified_at == "2026-01-15T10:00:00Z"
    assert coder.is_coding_model is True
    assert coder.score == score_ollama_model("qwen2.5-coder:14b")


def test_discover_ollama_models_flags_and_scores(monkeypatch):
    monkeypatch.setattr(
        "agentbridge.discovery.ollama.httpx.get",
        lambda url, timeout: FakeResponse(200, TAGS_PAYLOAD),
    )
    coder, r1, llama = discover_ollama_models()

    assert coder.is_coding_model is True
    assert r1.is_coding_model is False  # reasoning model, not code-specific
    assert llama.is_coding_model is False
    assert coder.score > r1.score > llama.score


def test_discover_ollama_models_empty_library(monkeypatch):
    monkeypatch.setattr(
        "agentbridge.discovery.ollama.httpx.get",
        lambda url, timeout: FakeResponse(200, {"models": []}),
    )
    assert discover_ollama_models() == []


def test_discover_ollama_models_skips_nameless_entries(monkeypatch):
    payload = {"models": [{"size": 100}, {"name": "qwen2:7b"}]}
    monkeypatch.setattr(
        "agentbridge.discovery.ollama.httpx.get",
        lambda url, timeout: FakeResponse(200, payload),
    )
    models = discover_ollama_models()
    assert [m.name for m in models] == ["qwen2:7b"]


def test_discover_ollama_models_propagates_http_errors(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("agentbridge.discovery.ollama.httpx.get", fake_get)
    with pytest.raises(httpx.ConnectError):
        discover_ollama_models()


# ---------------------------------------------------------------------------
# select_best_ollama_model
# ---------------------------------------------------------------------------


def make_model(name, score, size_bytes=0):
    return OllamaModelInfo(name=name, score=score, size_bytes=size_bytes)


def test_select_best_returns_none_for_empty_list():
    assert select_best_ollama_model([]) is None


def test_select_best_picks_highest_score():
    models = [
        make_model("llama3:8b", 40),
        make_model("qwen2.5-coder:14b", 145),
        make_model("deepseek-coder:7b", 145),
    ]
    best = select_best_ollama_model(models)
    assert best.name == "qwen2.5-coder:14b"


def test_select_best_tiebreak_prefers_larger_model():
    models = [
        make_model("deepseek-coder:7b", 145, size_bytes=4_000_000_000),
        make_model("deepseek-coder:14b", 145, size_bytes=9_000_000_000),
    ]
    assert select_best_ollama_model(models).name == "deepseek-coder:14b"


def test_select_best_single_model():
    model = make_model("qwen2.5-coder:32b", 180)
    assert select_best_ollama_model([model]) is model

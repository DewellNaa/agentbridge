"""Tests for the context budget squeezer (compactor) and its integration."""

import copy

import pytest
from fastapi.testclient import TestClient

import agentbridge.server as server_module
from agentbridge.config import BridgeConfig, UpstreamConfig
from agentbridge.context.compactor import (
    ContextCompactor,
    estimate_payload_tokens,
    estimate_tokens,
)
from agentbridge.telemetry.metrics import RequestMetric, TelemetryManager


def make_messages(n_pairs: int, text: str = "hello world") -> list[dict]:
    """Build an alternating user/assistant conversation."""
    messages = []
    for _ in range(n_pairs):
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": text})
    return messages


def tool_result_message(text: str, tool_use_id: str = "tu_1") -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": text,
            }
        ],
    }


def text_block_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_is_zero(self):
        assert estimate_tokens("") == 0

    def test_short_english_text(self):
        # 11 alnum/space chars at ~0.25 tokens each -> 3 tokens.
        assert estimate_tokens("hello world") == 3

    def test_ascii_ratio_is_about_four_chars_per_token(self):
        text = "the quick brown fox jumps over the lazy dog " * 20
        estimate = estimate_tokens(text)
        assert len(text) / 5 < estimate < len(text) / 3

    def test_chinese_is_weighted_heavier_than_ascii(self):
        chinese = estimate_tokens("中" * 100)
        ascii_text = estimate_tokens("a" * 100)
        assert chinese > ascii_text * 3

    def test_code_symbols_weighted_more_than_prose(self):
        code = estimate_tokens(";{}()[]" * 20)
        prose = estimate_tokens("abcdefg" * 20)
        assert code > prose

    def test_mixed_text_is_between_pure_bounds(self):
        mixed = estimate_tokens("abc " * 50 + "中文" * 50)
        assert mixed > estimate_tokens("abc " * 50)
        assert mixed < estimate_tokens("中文" * 50) * 2

    def test_returns_int(self):
        assert isinstance(estimate_tokens("hello"), int)

    def test_never_returns_zero_for_nonempty(self):
        assert estimate_tokens("a") >= 1


# ---------------------------------------------------------------------------
# estimate_payload_tokens
# ---------------------------------------------------------------------------


class TestEstimatePayloadTokens:
    def test_string_content(self):
        messages = [{"role": "user", "content": "hello world"}]
        assert estimate_payload_tokens(messages) == estimate_tokens("hello world")

    def test_text_blocks_are_summed(self):
        messages = [
            text_block_message("user", "hello world"),
            text_block_message("assistant", "hello"),
        ]
        expected = estimate_tokens("hello world") + estimate_tokens("hello")
        assert estimate_payload_tokens(messages) == expected

    def test_tool_result_string_content(self):
        messages = [tool_result_message("x" * 400)]
        assert estimate_payload_tokens(messages) == estimate_tokens("x" * 400)

    def test_tool_result_block_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [
                            {"type": "text", "text": "hello world"},
                            {"type": "text", "text": "second part"},
                        ],
                    }
                ],
            }
        ]
        expected = estimate_tokens("hello world") + estimate_tokens("second part")
        assert estimate_payload_tokens(messages) == expected

    def test_tool_use_blocks_contribute(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "run_tests",
                        "input": {"command": "pytest -q"},
                    }
                ],
            }
        ]
        assert estimate_payload_tokens(messages) > 0

    def test_system_string_counts(self):
        messages = [{"role": "user", "content": "hi"}]
        base = estimate_payload_tokens(messages)
        with_system = estimate_payload_tokens(messages, system="You are a bridge.")
        assert with_system > base

    def test_system_block_list_counts(self):
        messages = [{"role": "user", "content": "hi"}]
        base = estimate_payload_tokens(messages)
        system = [{"type": "text", "text": "You are a bridge."}]
        with_system = estimate_payload_tokens(messages, system=system)
        assert with_system == base + estimate_tokens("You are a bridge.")

    def test_empty_messages_are_zero(self):
        assert estimate_payload_tokens([]) == 0

    def test_unknown_block_type_is_tolerated(self):
        messages = [
            {"role": "user", "content": [{"type": "image", "source": {}}]},
            {"role": "user", "content": "hello world"},
        ]
        assert estimate_payload_tokens(messages) == estimate_tokens("hello world")


# ---------------------------------------------------------------------------
# No-op behaviour
# ---------------------------------------------------------------------------


class TestNoCompaction:
    def test_no_budget_returns_messages_unchanged(self):
        compactor = ContextCompactor()
        messages = make_messages(3, "x" * 5000)
        result, stats = compactor.compact(messages)
        assert result == messages
        assert stats["compressed"] is False
        assert stats["saved_tokens"] == 0

    def test_under_budget_returns_unchanged(self):
        compactor = ContextCompactor(max_context_tokens=100_000)
        messages = make_messages(4, "hello world")
        result, stats = compactor.compact(messages)
        assert result == messages
        assert stats["compressed"] is False
        assert stats["original_tokens"] == estimate_payload_tokens(messages)
        assert stats["compacted_tokens"] == stats["original_tokens"]
        assert stats["truncated_tools"] == 0
        assert stats["folded_turns"] == 0

    def test_no_budget_with_oversized_tool_results(self):
        compactor = ContextCompactor(max_context_tokens=None)
        messages = [tool_result_message("x" * 50_000)]
        result, _stats = compactor.compact(messages)
        assert result[0]["content"][0]["content"] == "x" * 50_000


# ---------------------------------------------------------------------------
# Stage 1: tool result truncation
# ---------------------------------------------------------------------------


class TestStageOneTruncation:
    def _old_turns(self, n_old=8, chars=5000):
        """Oversized old tool results (8 x ~1250 tok) plus a recent tail."""
        messages = [{"role": "user", "content": "Fix the failing tests"}]
        for i in range(n_old):
            messages.append(tool_result_message("x" * chars, tool_use_id=f"tu_{i}"))
        messages.extend(make_messages(2, "ok"))
        return messages

    def test_old_tool_result_is_head_tail_folded(self):
        compactor = ContextCompactor(max_context_tokens=8000)
        messages = self._old_turns()
        result, stats = compactor.compact(messages)
        folded = result[1]["content"][0]["content"]
        assert folded.startswith("x" * 800)
        assert folded.endswith("x" * 800)
        assert "Truncated 3400 chars of older tool output" in folded
        assert stats["compressed"] is True
        assert stats["truncated_tools"] == 8

    def test_marker_reports_saved_tokens(self):
        compactor = ContextCompactor(max_context_tokens=1000)
        result, _stats = compactor.compact(self._old_turns(n_old=1))
        folded = result[1]["content"][0]["content"]
        original = "x" * 5000
        head = "x" * 800
        tail = "x" * 800
        saved = estimate_tokens(original) - estimate_tokens(head + tail)
        assert f"saved ~{saved} tokens" in folded

    def test_recent_turns_untouched(self):
        compactor = ContextCompactor(max_context_tokens=8000)
        messages = self._old_turns()
        original = copy.deepcopy(messages)
        result, _stats = compactor.compact(messages)
        assert result[-4:] == original[-4:]

    def test_first_turn_preserved_from_truncation(self):
        compactor = ContextCompactor(max_context_tokens=10)
        messages = [
            {"role": "user", "content": "x" * 5000},
            *make_messages(4, "ok"),
        ]
        result, _stats = compactor.compact(messages)
        assert result[0]["content"] == "x" * 5000

    def test_short_tool_results_untouched(self):
        compactor = ContextCompactor(max_context_tokens=100_000)
        messages = self._old_turns(chars=100)
        result, stats = compactor.compact(messages)
        assert stats["compressed"] is False
        assert result[1]["content"][0]["content"] == "x" * 100

    def test_oversized_text_blocks_also_folded(self):
        compactor = ContextCompactor(max_context_tokens=1000)
        messages = [
            {"role": "user", "content": "goal"},
            text_block_message("assistant", "y" * 5000),
            *make_messages(2, "ok"),
        ]
        result, stats = compactor.compact(messages)
        folded = result[1]["content"][0]["text"]
        assert folded.startswith("y" * 800)
        assert "Truncated" in folded
        assert stats["truncated_tools"] == 1

    def test_oversized_plain_string_content_folded(self):
        compactor = ContextCompactor(max_context_tokens=1000)
        messages = [
            {"role": "user", "content": "goal"},
            {"role": "user", "content": "z" * 5000},
            *make_messages(2, "ok"),
        ]
        result, stats = compactor.compact(messages)
        assert result[1]["content"].startswith("z" * 800)
        assert stats["truncated_tools"] == 1

    def test_stage_one_satisfies_budget_without_folding(self):
        compactor = ContextCompactor(max_context_tokens=8000)
        result, stats = compactor.compact(self._old_turns())
        assert stats["folded_turns"] == 0
        assert stats["compacted_tokens"] <= 8000
        assert len(result) == 13

    def test_saved_tokens_positive_and_consistent(self):
        compactor = ContextCompactor(max_context_tokens=8000)
        messages = self._old_turns()
        result, stats = compactor.compact(messages)
        assert stats["saved_tokens"] > 0
        assert stats["saved_tokens"] == (
            stats["original_tokens"] - stats["compacted_tokens"]
        )
        assert estimate_payload_tokens(result) == pytest.approx(
            stats["compacted_tokens"]
        )


# ---------------------------------------------------------------------------
# Stage 2: sliding window fold
# ---------------------------------------------------------------------------


class TestStageTwoSlidingWindow:
    def _long_session(self, n_pairs=30, text="a" * 200):
        messages = [{"role": "user", "content": "Remember the original goal"}]
        messages.extend(make_messages(n_pairs, text))
        return messages

    def test_middle_turns_are_folded(self):
        compactor = ContextCompactor(max_context_tokens=1200, preserve_recent_turns=4)
        messages = self._long_session()
        result, stats = compactor.compact(messages)
        # 1 + 60 messages: index 0 and last 4 kept, 56 folded into one pair.
        assert stats["folded_turns"] == 56
        assert len(result) == 1 + 2 + 4
        assert stats["compressed"] is True

    def test_first_turn_kept_intact(self):
        compactor = ContextCompactor(max_context_tokens=1200)
        messages = self._long_session()
        result, _stats = compactor.compact(messages)
        assert result[0]["content"] == "Remember the original goal"

    def test_recent_turns_kept_intact(self):
        compactor = ContextCompactor(max_context_tokens=1200)
        messages = self._long_session()
        original = copy.deepcopy(messages)
        result, _stats = compactor.compact(messages)
        assert result[-4:] == original[-4:]

    def test_placeholder_pair_content(self):
        compactor = ContextCompactor(max_context_tokens=1200)
        result, _stats = compactor.compact(self._long_session())
        assert result[1] == {
            "role": "user",
            "content": (
                "[AgentBridge: Compressed 56 intermediate conversation turns "
                "to preserve context budget]"
            ),
        }
        assert result[2]["role"] == "assistant"
        assert "AgentBridge" in result[2]["content"]

    def test_final_payload_fits_budget(self):
        compactor = ContextCompactor(max_context_tokens=1200, preserve_recent_turns=4)
        result, stats = compactor.compact(self._long_session())
        assert stats["compacted_tokens"] <= 1200
        assert estimate_payload_tokens(result) <= 1200

    def test_first_turn_can_be_folded_when_disabled(self):
        compactor = ContextCompactor(max_context_tokens=600, preserve_first_turn=False)
        messages = self._long_session()
        result, stats = compactor.compact(messages)
        assert result[0]["role"] == "user"
        assert "Compressed 57 intermediate" in result[0]["content"]
        assert stats["folded_turns"] == 57

    def test_nothing_to_fold_when_all_recent(self):
        compactor = ContextCompactor(max_context_tokens=10)
        messages = make_messages(2, "hello world")
        result, stats = compactor.compact(messages)
        # Only the budget is exceeded; nothing can be safely folded.
        assert result == messages
        assert stats["compressed"] is True
        assert stats["folded_turns"] == 0

    def test_stage_one_runs_before_stage_two(self):
        # Oversized tool output plus excess turns: both counters populate.
        compactor = ContextCompactor(max_context_tokens=1000, preserve_recent_turns=4)
        messages = [{"role": "user", "content": "goal"}]
        for i in range(10):
            messages.append(tool_result_message("x" * 5000, tool_use_id=f"tu_{i}"))
        messages.extend(make_messages(2, "ok"))
        result, stats = compactor.compact(messages)
        assert stats["truncated_tools"] == 10
        assert stats["folded_turns"] == 10
        assert stats["compacted_tokens"] <= 1000
        assert len(result) == 1 + 2 + 4


# ---------------------------------------------------------------------------
# Non-destructiveness
# ---------------------------------------------------------------------------


class TestNonDestructive:
    def test_input_messages_not_mutated(self):
        compactor = ContextCompactor(max_context_tokens=8000)
        messages = [
            {"role": "user", "content": "goal"},
            tool_result_message("x" * 5000),
            *make_messages(2, "ok"),
        ]
        snapshot = copy.deepcopy(messages)
        compactor.compact(messages)
        assert messages == snapshot

    def test_nested_blocks_not_mutated_in_stage_two(self):
        compactor = ContextCompactor(max_context_tokens=1200)
        messages = [
            {"role": "user", "content": "goal"},
            *make_messages(30, "a" * 200),
        ]
        snapshot = copy.deepcopy(messages)
        compactor.compact(messages)
        assert messages == snapshot

    def test_system_argument_not_mutated(self):
        compactor = ContextCompactor(max_context_tokens=8000)
        system = [{"type": "text", "text": "sys " * 1000}]
        snapshot = copy.deepcopy(system)
        compactor.compact(
            [
                {"role": "user", "content": "goal"},
                tool_result_message("x" * 5000),
                *make_messages(2, "ok"),
            ],
            system=system,
        )
        assert system == snapshot


# ---------------------------------------------------------------------------
# Telemetry plumbing for saved tokens
# ---------------------------------------------------------------------------


class TestTelemetrySavedTokens:
    def test_record_context_savings_accumulates(self):
        manager = TelemetryManager()
        metric = manager.start_request("req_1", "m", "m", "client", False)
        manager.record_context_savings(metric, 120)
        manager.record_context_savings(metric, 30)
        assert metric.saved_tokens == 150
        assert manager.total_context_tokens_saved == 150

    def test_record_context_savings_ignores_non_positive(self):
        manager = TelemetryManager()
        metric = manager.start_request("req_1", "m", "m", "client", False)
        manager.record_context_savings(metric, 0)
        manager.record_context_savings(metric, -5)
        assert metric.saved_tokens == 0
        assert manager.total_context_tokens_saved == 0

    def test_summary_includes_saved_tokens(self):
        manager = TelemetryManager()
        metric = manager.start_request("req_1", "m", "m", "client", False)
        manager.record_context_savings(metric, 99)
        manager.finish_request(metric)
        summary = manager.get_summary()
        assert summary["total_context_tokens_saved"] == 99
        assert summary["recent_requests"][0]["saved_tokens"] == 99

    def test_metric_defaults_to_zero(self):
        metric = RequestMetric(request_id="req_1", timestamp=0.0)
        assert metric.saved_tokens == 0


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------


def _fake_upstream_response(payload):
    return {
        "id": "chatcmpl-test",
        "model": payload.get("model", "unknown"),
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def make_bridge_config(**overrides) -> BridgeConfig:
    defaults = {
        "upstream": UpstreamConfig(
            name="test-upstream",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            default_model="gpt-4o",
        ),
    }
    defaults.update(overrides)
    return BridgeConfig(**defaults)


def _compact_payload():
    return {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Remember the original goal"},
            tool_result_message("x" * 5000, tool_use_id="tu_1"),
            tool_result_message("x" * 5000, tool_use_id="tu_2"),
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
    }


class TestServerIntegration:
    def test_compaction_disabled_by_default(self):
        app = server_module.create_app(make_bridge_config())
        captured = {}

        async def fake_complete(payload):
            captured.update(payload)
            return _fake_upstream_response(payload)

        with TestClient(app) as client:
            app.state.client.complete = fake_complete
            resp = client.post("/v1/messages", json=_compact_payload())

        assert resp.status_code == 200
        tool_messages = [m for m in captured["messages"] if m["role"] == "tool"]
        assert all("x" * 4000 in m["content"] for m in tool_messages)

    def test_max_context_tokens_auto_enables_compaction(self):
        app = server_module.create_app(make_bridge_config(max_context_tokens=2500))
        captured = {}

        async def fake_complete(payload):
            captured.update(payload)
            return _fake_upstream_response(payload)

        with TestClient(app) as client:
            app.state.client.complete = fake_complete
            resp = client.post("/v1/messages", json=_compact_payload())

        assert resp.status_code == 200
        tool_messages = [m for m in captured["messages"] if m["role"] == "tool"]
        assert len(tool_messages) == 2
        for message in tool_messages:
            assert "Truncated" in message["content"]
            assert "x" * 4000 not in message["content"]

    def test_saved_tokens_recorded_in_telemetry(self):
        app = server_module.create_app(make_bridge_config(max_context_tokens=2500))
        with TestClient(app) as client:

            async def fake_complete(payload):
                return _fake_upstream_response(payload)

            app.state.client.complete = fake_complete
            resp = client.post("/v1/messages", json=_compact_payload())
            assert resp.status_code == 200
            summary = client.get("/v1/telemetry").json()

        assert summary["total_context_tokens_saved"] > 0
        assert summary["recent_requests"][0]["saved_tokens"] > 0

    def test_no_savings_when_under_budget(self):
        app = server_module.create_app(make_bridge_config(max_context_tokens=100_000))
        with TestClient(app) as client:

            async def fake_complete(payload):
                return _fake_upstream_response(payload)

            app.state.client.complete = fake_complete
            resp = client.post("/v1/messages", json=_compact_payload())
            assert resp.status_code == 200
            summary = client.get("/v1/telemetry").json()

        assert summary["total_context_tokens_saved"] == 0
        assert summary["recent_requests"][0]["saved_tokens"] == 0

    def test_app_state_exposes_compactor(self):
        app = server_module.create_app(make_bridge_config(compact_context=True))
        assert app.state.compactor is not None

    def test_app_state_compactor_none_when_disabled(self):
        app = server_module.create_app(make_bridge_config())
        assert app.state.compactor is None


# ---------------------------------------------------------------------------
# Config & CLI wiring
# ---------------------------------------------------------------------------


class TestConfigCompactionSettings:
    def test_defaults(self):
        config = BridgeConfig()
        assert config.max_context_tokens is None
        assert config.compact_context is False
        assert config.preserve_recent_turns == 4
        assert config.max_tool_result_chars == 2000

    def test_from_env_parses_compaction_settings(self, monkeypatch):
        monkeypatch.setenv("AGENTBRIDGE_MAX_CONTEXT_TOKENS", "64000")
        monkeypatch.setenv("AGENTBRIDGE_COMPACT_CONTEXT", "true")
        monkeypatch.setenv("AGENTBRIDGE_PRESERVE_RECENT_TURNS", "6")
        monkeypatch.setenv("AGENTBRIDGE_MAX_TOOL_RESULT_CHARS", "1200")
        config = BridgeConfig.from_env()
        assert config.max_context_tokens == 64000
        assert config.compact_context is True
        assert config.preserve_recent_turns == 6
        assert config.max_tool_result_chars == 1200

    def test_from_env_defaults_when_unset(self, monkeypatch):
        for key in (
            "AGENTBRIDGE_MAX_CONTEXT_TOKENS",
            "AGENTBRIDGE_COMPACT_CONTEXT",
            "AGENTBRIDGE_PRESERVE_RECENT_TURNS",
            "AGENTBRIDGE_MAX_TOOL_RESULT_CHARS",
        ):
            monkeypatch.delenv(key, raising=False)
        config = BridgeConfig.from_env()
        assert config.max_context_tokens is None
        assert config.compact_context is False
        assert config.preserve_recent_turns == 4
        assert config.max_tool_result_chars == 2000

    def test_from_env_invalid_max_context_is_none(self, monkeypatch):
        monkeypatch.setenv("AGENTBRIDGE_MAX_CONTEXT_TOKENS", "not-a-number")
        assert BridgeConfig.from_env().max_context_tokens is None

    def test_from_env_compact_context_one_is_true(self, monkeypatch):
        monkeypatch.setenv("AGENTBRIDGE_COMPACT_CONTEXT", "1")
        assert BridgeConfig.from_env().compact_context is True


class TestCliCompactionFlags:
    def _parser(self):
        from agentbridge.cli import create_parser

        return create_parser()

    def test_flags_default_off(self):
        args = self._parser().parse_args(["start"])
        assert args.max_context_tokens is None
        assert args.compact_context is False
        assert args.preserve_recent_turns == 4
        assert args.max_tool_chars == 2000

    def test_flags_parsed(self):
        args = self._parser().parse_args(
            [
                "start",
                "--max-context-tokens",
                "32000",
                "--compact-context",
                "--preserve-recent-turns",
                "6",
                "--max-tool-chars",
                "1500",
            ]
        )
        assert args.max_context_tokens == 32000
        assert args.compact_context is True
        assert args.preserve_recent_turns == 6
        assert args.max_tool_chars == 1500

    def test_banner_shows_compactor_row(self, capsys):
        from agentbridge import cli

        config = BridgeConfig(
            max_context_tokens=32000, compact_context=True, preserve_recent_turns=6
        )
        cli.print_startup_banner(config)
        out = capsys.readouterr().out
        assert "Context Compactor:" in out
        assert "Active (Budget: 32000 tokens, Preserve: 6 turns)" in out

    def test_banner_omits_compactor_when_disabled(self, capsys):
        from agentbridge import cli

        cli.print_startup_banner(BridgeConfig())
        out = capsys.readouterr().out
        assert "Context Compactor:" not in out

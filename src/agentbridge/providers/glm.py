"""Zhipu BigModel / Z.ai (GLM) native provider adapter.

Talks to the GLM OpenAI-compatible endpoint hosted at
``https://open.bigmodel.cn/api/paas/v4`` (mainland) or
``https://api.z.ai/api/paas/v4`` (international), applying the
GLM-specific request customizations (auth scheme, model aliases and
the ``thinking`` / ``do_sample`` controls) on behalf of coding agents
that speak Claude/OpenAI dialects.
"""

import copy
import json
from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx

from agentbridge.config import UpstreamConfig
from agentbridge.providers.client import UpstreamClient
from agentbridge.streaming.buffer import StreamBuffer


class GLMProvider(UpstreamClient):
    """Async HTTP client for the Zhipu BigModel / Z.ai GLM API."""

    BIGMODEL_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

    # base_url fallbacks when the config still carries a non-GLM endpoint.
    _REGION_BASE_URLS = {
        "cn": BIGMODEL_BASE_URL,
        "zai": ZAI_BASE_URL,
    }
    _OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

    CHAT_COMPLETIONS_PATH = "/chat/completions"

    DEFAULT_MODEL = "glm-5.3"
    DEFAULT_FLASH_MODEL = "glm-5.3-flash"

    # Coding agents emit Claude model ids; map them onto GLM counterparts.
    MODEL_ALIASES: Dict[str, str] = {
        "glm-5.3": "glm-5.3",
        "glm-5.3-flash": "glm-5.3-flash",
        "claude-3-7-sonnet-20250219": "glm-5.3",
        "claude-3-5-sonnet-20241022": "glm-5.3",
        "claude-sonnet-4-20250514": "glm-5.3",
        "claude-3-5-haiku-20241022": "glm-5.3-flash",
    }

    VALID_THINKING_TYPES = {"enabled", "disabled"}

    def __init__(
        self,
        config: UpstreamConfig,
        client_token: str = "",
        region: str = "cn",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """Build a GLM provider.

        Args:
            config: Upstream configuration; ``api_key`` (if set) wins over
                ``client_token``.
            client_token: Credential forwarded by the connecting agent
                (e.g. the ``Authorization``/``x-api-key`` value of the
                incoming request). Used when the config carries no key.
            region: ``"cn"`` for BigModel or ``"zai"`` for Z.ai; only used
                when ``config.base_url`` is empty or the OpenAI default.
            transport: Optional httpx transport (used by tests).

        Raises:
            ValueError: If neither a config API key nor a client token
                is available.
        """
        base_url = self._resolve_base_url(config.base_url, region)
        api_key = self.resolve_api_key(config.api_key, client_token)
        if not api_key:
            raise ValueError(
                "GLMProvider requires an API key in config or a client token"
            )
        self.api_key = api_key

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.auth_headers(api_key),
            **config.extra_headers,
        }
        # Same attribute contract as UpstreamClient; built here so the
        # GLM auth headers and base URL are applied from the start.
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=config.timeout,
            transport=transport,
        )

    # ------------------------------------------------------------------
    # Credential & endpoint resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_api_key(api_key: str, client_token: str = "") -> str:
        """Pick the credential to use, stripping any ``Bearer`` prefix."""
        key = _strip_bearer(api_key)
        if key:
            return key
        return _strip_bearer(client_token)

    @staticmethod
    def auth_headers(api_key: str) -> Dict[str, str]:
        """BigModel/Z.ai auth header: ``Authorization: Bearer <key>``."""
        return {"Authorization": f"Bearer {_strip_bearer(api_key)}"}

    @classmethod
    def _resolve_base_url(cls, base_url: str, region: str) -> str:
        """Fall back to the regional GLM endpoint for non-GLM base URLs."""
        if base_url and base_url.rstrip("/") != cls._OPENAI_DEFAULT_BASE_URL:
            return base_url.rstrip("/")
        try:
            return cls._REGION_BASE_URLS[region]
        except KeyError:
            raise ValueError(
                f"Unknown GLM region {region!r}; expected one of "
                f"{sorted(cls._REGION_BASE_URLS)}"
            ) from None

    # ------------------------------------------------------------------
    # Model mapping & GLM-specific parameters
    # ------------------------------------------------------------------

    def resolve_model(self, model: Optional[str]) -> str:
        """Map an incoming model name onto a GLM model id."""
        if not model:
            return self._default_glm_model()
        if model in self.MODEL_ALIASES:
            return self.MODEL_ALIASES[model]
        if model.startswith("glm"):
            return model
        # Unknown third-party id: use the configured default when it is a
        # GLM model, otherwise fall back to the flagship GLM model.
        return self._default_glm_model()

    def _default_glm_model(self) -> str:
        if self.config.default_model.startswith("glm"):
            return self.config.default_model
        return self.DEFAULT_MODEL

    @staticmethod
    def normalize_thinking(
        value: Optional[Union[bool, str, Dict[str, Any]]],
    ) -> Optional[Dict[str, str]]:
        """Normalize a ``thinking`` control into ``{"type": ...}`` form.

        Accepts booleans, the strings ``"enabled"``/``"disabled"`` and
        dicts carrying a ``type`` key, mirroring the GLM API contract.

        Raises:
            ValueError: If the value cannot be interpreted.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return {"type": "enabled" if value else "disabled"}
        if isinstance(value, str):
            if value.lower() not in GLMProvider.VALID_THINKING_TYPES:
                raise ValueError(
                    f"Invalid thinking type {value!r}; expected "
                    f"{sorted(GLMProvider.VALID_THINKING_TYPES)}"
                )
            return {"type": value.lower()}
        if isinstance(value, dict):
            thinking_type = value.get("type")
            if (
                not isinstance(thinking_type, str)
                or thinking_type.lower() not in GLMProvider.VALID_THINKING_TYPES
            ):
                raise ValueError(
                    f"Invalid thinking payload {value!r}; 'type' must be one of "
                    f"{sorted(GLMProvider.VALID_THINKING_TYPES)}"
                )
            return {"type": thinking_type.lower()}
        raise ValueError(f"Unsupported thinking value: {value!r}")

    def prepare_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Turn an incoming chat payload into a GLM-compatible one.

        - Maps the model name onto GLM-5.3 / GLM-5.3-Flash.
        - Normalizes the ``thinking`` control (kept only when set; GLM
          applies its own server-side default otherwise).
        - Drops ``temperature``/``top_p`` when ``do_sample`` is False, as
          required by the GLM API.
        - Strips Anthropic-style ``cache_control`` markers that upstream
          endpoints reject with 400.
        """
        prepared = copy.deepcopy(payload)

        prepared["model"] = self.resolve_model(payload.get("model"))
        thinking = self.normalize_thinking(payload.get("thinking"))
        if thinking is not None:
            prepared["thinking"] = thinking

        do_sample = payload.get("do_sample")
        if do_sample is not None:
            if not isinstance(do_sample, bool):
                raise ValueError(f"do_sample must be a boolean, got {do_sample!r}")
            prepared["do_sample"] = do_sample
            if do_sample is False:
                # GLM requires the default greedy-decoding values when
                # sampling is disabled; omit them entirely.
                prepared.pop("temperature", None)
                prepared.pop("top_p", None)

        return _strip_cache_control(prepared)

    # ------------------------------------------------------------------
    # HTTP surface
    # ------------------------------------------------------------------

    async def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a non-streaming chat completion to GLM."""
        response = await self.client.post(
            self.CHAT_COMPLETIONS_PATH, json=self.prepare_payload(payload)
        )
        response.raise_for_status()
        return response.json()

    async def stream_complete(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a GLM chat completion, yielding parsed SSE data chunks.

        GLM uses the same ``data: {...}`` / ``data: [DONE]`` SSE framing
        as OpenAI; deltas may carry ``reasoning_content`` alongside
        ``content`` and are yielded verbatim for downstream adapters.
        """
        prepared = self.prepare_payload(payload)
        prepared["stream"] = True

        buffer = StreamBuffer()
        async with self.client.stream(
            "POST", self.CHAT_COMPLETIONS_PATH, json=prepared
        ) as response:
            response.raise_for_status()

            async for raw_text in response.aiter_text():
                for line in buffer.feed(raw_text):
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue


def _strip_bearer(token: str) -> str:
    """Strip a case-insensitive ``Bearer `` prefix from a token."""
    token = (token or "").strip()
    if token[:7].lower() == "bearer ":
        return token[7:].strip()
    return token


def _strip_cache_control(obj: Any) -> Any:
    """Recursively remove ``cache_control`` keys from dicts/lists."""
    if isinstance(obj, dict):
        return {
            key: _strip_cache_control(value)
            for key, value in obj.items()
            if key != "cache_control"
        }
    if isinstance(obj, list):
        return [_strip_cache_control(item) for item in obj]
    return obj

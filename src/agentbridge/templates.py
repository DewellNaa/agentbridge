"""One-click IDE configuration templates for AgentBridge.

Generates ready-to-paste config snippets/guides for popular AI IDEs so they
can talk to a local AgentBridge gateway instead of the real upstream API.
"""

import json

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_API_KEY = "agentbridge"
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
FALLBACK_MODEL = "claude-3-5-sonnet-20241022"

SUPPORTED_TARGETS = ("cursor", "cline", "continue")


def _unsupported_message(target: str) -> str:
    return (
        f"Unsupported IDE target: '{target}'.\n"
        f"Supported targets: {', '.join(SUPPORTED_TARGETS)}"
    )


def _generate_cursor(host: str, port: int) -> str:
    base = f"http://{host}:{port}"
    return f"""# Cursor <-> AgentBridge one-click configuration guide

## 1. Start the local gateway
    agentbridge start --host {host} --port {port}

## 2. Anthropic-protocol mapping (Claude model requests)
Cursor's Claude model requests are transparently mapped to the upstream
provider configured on the gateway:
  - {DEFAULT_MODEL} -> upstream default model
  - {FALLBACK_MODEL} -> upstream default model
  Endpoint: {base}/v1/messages
  API Key:  {DEFAULT_API_KEY}  (any non-empty value works)

## 3. OpenAI-compatible endpoint
Cursor custom OpenAI-compatible providers can also point at the gateway:
  Base URL: {base}/v1
  API Key:  {DEFAULT_API_KEY}

## 4. Example `.cursorrules` for your repository
```
# .cursorrules
You are an expert coding assistant.

# All model traffic is routed through the local AgentBridge gateway.
# No extra setup is needed inside Cursor: the gateway handles
# Anthropic -> upstream protocol translation.
```

## 5. Manual override (optional)
Set these environment variables before launching Cursor if you want its
built-in Anthropic integration to hit the gateway directly:
  ANTHROPIC_BASE_URL={base}
  ANTHROPIC_API_KEY={DEFAULT_API_KEY}
"""


def _generate_cline(host: str, port: int) -> str:
    config = {
        "apiProvider": "anthropic",
        "anthropicBaseUrl": f"http://{host}:{port}",
        "anthropicApiKey": DEFAULT_API_KEY,
        "anthropicModel": DEFAULT_MODEL,
        "openAiBaseUrl": f"http://{host}:{port}/v1",
        "openAiApiKey": DEFAULT_API_KEY,
        "openAiModelId": FALLBACK_MODEL,
    }
    return json.dumps(config, indent=2)


def _generate_continue(host: str, port: int) -> str:
    config = {
        "models": [
            {
                "title": "AgentBridge (Claude via local gateway)",
                "provider": "anthropic",
                "model": DEFAULT_MODEL,
                "apiKey": DEFAULT_API_KEY,
                "apiBase": f"http://{host}:{port}",
            },
            {
                "title": "AgentBridge (OpenAI-compatible)",
                "provider": "openai",
                "model": FALLBACK_MODEL,
                "apiKey": DEFAULT_API_KEY,
                "apiBase": f"http://{host}:{port}/v1",
            },
        ]
    }
    return json.dumps(config, indent=2)


def generate_ide_config(
    target: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> str:
    """Return a one-click configuration snippet/guide for the given IDE.

    Supported targets: cursor, cline, continue. Unknown targets return a
    message listing the supported targets.
    """
    normalized = target.strip().lower()
    if normalized == "cursor":
        return _generate_cursor(host, port)
    if normalized == "cline":
        return _generate_cline(host, port)
    if normalized == "continue":
        return _generate_continue(host, port)
    return _unsupported_message(target)

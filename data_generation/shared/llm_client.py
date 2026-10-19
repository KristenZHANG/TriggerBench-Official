"""OpenAI-compatible chat client for TriggerBench data generation.

Configure via environment variables (do not hardcode credentials or endpoints):

  OPENAI_API_KEY           Preferred API key.
  AZURE_OPENAI_API_KEY     Accepted as a fallback (same as AZURE_OPENAI_KEY).
  OPENAI_BASE_URL          Optional chat base URL. For Azure AI Foundry use
                           ``https://<resource>.services.ai.azure.com/openai/v1``.
  AZURE_OPENAI_ENDPOINT    Fallback if OPENAI_BASE_URL is unset. Foundry project
                           URLs (``.../api/projects/...``) are rewritten to
                           ``.../openai/v1`` automatically.
  OPENAI_MODEL             Optional default model name (overridden by CLI ``--model``).

Example (OpenAI)::

    export OPENAI_API_KEY=sk-...

Example (Azure AI Foundry / OpenAI-compatible v1)::

    export OPENAI_API_KEY=<azure-key>
    export OPENAI_BASE_URL=https://<resource>.services.ai.azure.com/openai/v1
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from openai import OpenAI

_CLIENT: OpenAI | None = None


def _api_key() -> str | None:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_KEY")
    )


def _normalize_base_url(url: str) -> str:
    """Map Foundry project endpoints onto the OpenAI-compatible ``/openai/v1`` surface."""
    url = url.strip().rstrip("/")
    if not url:
        return url
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "/api/projects/" in parsed.path or (
        host.endswith("services.ai.azure.com") and "/openai/" not in parsed.path
    ):
        return f"{parsed.scheme}://{parsed.netloc}/openai/v1"
    return url


def _base_url() -> str | None:
    raw = os.environ.get("OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not raw:
        return None
    return _normalize_base_url(raw)


def get_client() -> OpenAI:
    """Return a process-wide OpenAI client constructed from environment variables."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = _api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) is not set. "
            "Export it before running the generation pipeline."
        )

    kwargs: dict = {"api_key": api_key}
    base_url = _base_url()
    if base_url:
        kwargs["base_url"] = base_url

    _CLIENT = OpenAI(**kwargs)
    return _CLIENT


def chat(
    messages: list[dict],
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Send a ChatML message list and return the assistant text."""
    client = get_client()
    create_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # GPT-5 series uses max_completion_tokens; GPT-4 series uses max_tokens.
    if model.startswith("gpt-5"):
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**create_kwargs)
    content = completion.choices[0].message.content
    return content or ""


def complete(
    system_prompt: str | None,
    user_prompt: str,
    model: str = "gpt-4.1",
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """Single-turn completion with an optional system prompt."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)

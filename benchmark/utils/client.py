"""OpenAI-compatible chat client for TriggerBench evaluation.

API keys and endpoints come from the environment (see
``prospective_memory/source/.env``). Foundry project URLs
(``.../api/projects/...``) are rewritten to ``.../openai/v1``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import yaml
from openai import APIStatusError, AuthenticationError, OpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

try:
    from dotenv import load_dotenv
except ImportError:  # optional
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

load_dotenv()

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
_GEN_ENV = Path(__file__).resolve().parent.parent.parent / "data_generation" / ".env"
if _GEN_ENV.is_file():
    load_dotenv(_GEN_ENV, override=False)


class ContentFilterError(Exception):
    """Raised when a content filter blocks or truncates a response."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


CONFIG: dict[str, Any] = _load_yaml(_CONFIGS_DIR / "llm.yaml")
_DEFAULT_MAX_TOKENS: int = CONFIG["max_tokens"]["default"]

_OPENAI_CLIENT: OpenAI | None = None


def _before_sleep_print(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    print(f"  retry in {wait:.1f}s: {type(exc).__name__}: {exc}")


def _wait_by_error_type(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, AuthenticationError):
        return 2
    return float(CONFIG["retry"]["wait_seconds"])


_llm_retry = retry(
    stop=stop_after_attempt(CONFIG["retry"]["max_attempts"]),
    wait=_wait_by_error_type,
    retry=retry_if_exception_type((APIStatusError, json.JSONDecodeError, ContentFilterError)),
    before_sleep=_before_sleep_print,
    reraise=True,
)


def _parse_env_list(var_name: str) -> list[str]:
    raw = os.environ.get(var_name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_api_key() -> str | None:
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    return None


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


def _get_default_deployment() -> str:
    deployments = _parse_env_list("AZURE_OPENAI_DEPLOYMENTS")
    if deployments:
        return deployments[0]
    model = os.environ.get("OPENAI_MODEL", "").strip()
    if model:
        return model
    raise ValueError("Set AZURE_OPENAI_DEPLOYMENTS (or OPENAI_MODEL)")


def get_openai_client() -> OpenAI:
    """Process-wide OpenAI client from environment variables."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "export AZURE_OPENAI_API_KEY=...  (OPENAI_API_KEY is also accepted)"
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = _base_url()
    if base_url:
        kwargs["base_url"] = base_url
    _OPENAI_CLIENT = OpenAI(**kwargs)
    return _OPENAI_CLIENT


def _get_client_kwargs(endpoint: str, api_version: str) -> dict[str, Any]:
    """Kwargs for the remaining Azure embeddings helpers (rag / amem / letta).

    Chat evaluation uses get_openai_client() instead. Kept so those drivers
    keep importing this name.
    """
    api_key = _get_api_key()
    if api_key:
        return {"azure_endpoint": endpoint, "api_key": api_key, "api_version": api_version}
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return {
        "azure_endpoint": endpoint,
        "azure_ad_token_provider": token_provider,
        "api_version": api_version,
    }


class LLMClient:
    """Thin OpenAI-compatible wrapper supporting chat and JSON mode."""

    def __init__(
        self,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
    ):
        self.deployment = deployment or _get_default_deployment()
        self.endpoint = endpoint or _base_url()
        self.api_version = api_version
        self._client = get_openai_client()

    @_llm_retry
    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> str:
        create_kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": messages,
            "temperature": temperature,
        }
        if str(self.deployment).startswith("gpt-5"):
            create_kwargs["max_completion_tokens"] = max_tokens
        else:
            create_kwargs["max_tokens"] = max_tokens
        response = self._client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content or ""

    @_llm_retry
    def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.deployment,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


def run_parallel(
    items: list[Any],
    func: Callable[[LLMClient, Any], Any],
    client: LLMClient | None = None,
    max_concurrency: int = 8,
) -> list[Any]:
    """Simple asyncio.to_thread-based parallel runner over a shared client."""
    client = client or LLMClient()

    async def _run() -> list[Any]:
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = [None] * len(items)

        async def process(i: int, item: Any) -> None:
            async with semaphore:
                results[i] = await asyncio.to_thread(func, client, item)

        await asyncio.gather(*[process(i, it) for i, it in enumerate(items)])
        return results

    return asyncio.run(_run())

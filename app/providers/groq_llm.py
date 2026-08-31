"""Groq-backed :class:`~app.providers.llm.LLMProvider`.

Same contract as :mod:`app.providers.ollama_llm`, which is untouched. The two
differ only in how the schema reaches the model.

**Structured output mode: full JSON schema (`json_schema`), verified against the
live API on 2026-08-31 — not loose `json_object` mode.** That matters, because
loose mode would guarantee only "some JSON" and would push the burden of field
names and types onto the validate-and-retry loop in ``app.structure``. With
``json_schema`` + ``strict: true`` the decoder is constrained the same way
Ollama's ``format=`` constrains it, so behaviour mirrors the local provider and
malformed shape stays unrepresentable.

One wrinkle found by probing rather than reading: Groq rejects pydantic's schema
as-is with

    invalid JSON schema for response_format: `additionalProperties:false`
    must be set on every object

so :func:`harden_schema` post-processes the schema before sending it. The 400 is
a schema-shape requirement, not a missing feature — it would be easy to read it
as "json_schema unsupported, fall back to json_object" and quietly lose the
grammar.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.providers.llm import (
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

#: Status codes worth one retry. 429 is rate limiting, 5xx is Groq having a
#: moment; both are transient. 400/401/404 are not, and retrying them only
#: delays a clear error.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

# Models sometimes wrap JSON in a fence even under a constrained decoder.
# Cheap to strip, and a fence would otherwise fail json.loads with a message
# that says nothing about the real cause.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def harden_schema(node: Any) -> Any:
    """Make a pydantic JSON schema acceptable to Groq's strict mode, in place.

    Two edits on every object node:

    * ``additionalProperties: false`` — required by Groq, not emitted by
      pydantic.
    * ``required`` listing every property — OpenAI-compatible strict mode
      requires it. This is a no-op for our models, whose fields are all
      required-but-nullable by design (see the note in ``app.structure`` on why
      a field with a default gets silently omitted by the model), but it makes
      the transform correct for any schema handed to it.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
            node["required"] = list((node.get("properties") or {}).keys())
        for value in node.values():
            harden_schema(value)
    elif isinstance(node, list):
        for item in node:
            harden_schema(item)
    return node


def _strip_fence(content: str) -> str:
    return _FENCE_RE.sub("", content.strip()).strip()


class GroqLLMProvider(LLMProvider):
    """Structured extraction against Groq's hosted OpenAI-compatible API.

    Uses httpx directly rather than the ``groq`` SDK: httpx is already a
    dependency for the job sources, the surface used here is two endpoints, and
    an SDK would add a package to pin for no capability we need.
    """

    name = "groq"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = self._settings.groq_model
        self._api_key = self._settings.groq_api_key
        self._base_url = self._settings.groq_base_url.rstrip("/")
        # Injectable for tests, matching AdzunaSource and the Ollama providers.
        self._client = client
        self._owns_client = client is None
        #: Rate-limit headers from the most recent response. Free-tier limits
        #: are low enough to matter (see DECISIONS 22.5), so they are kept
        #: rather than discarded.
        self.last_rate_limit: dict[str, str] = {}

    # --- plumbing ---------------------------------------------------------

    def _require_key(self) -> str:
        if not self._api_key:
            raise LLMUnavailableError(
                "GROQ_API_KEY is not set. Add it to .env, or set "
                "PARSE_PROVIDER=ollama to run entirely locally."
            )
        return self._api_key

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._settings.groq_timeout_seconds)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """One chat completion, with one retry on transient failure.

        Retries once and no more: the caller (``app.structure``) has its own
        validate-and-retry loop for *semantic* failures, and stacking retries
        multiplies worst-case latency, which is the entire reason for using a
        hosted provider here.
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                response = self._http().post(url, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                # Distinct from a connection failure, for the same reason as in
                # ollama_llm: a timeout is not an unreachable host, and saying
                # so sends whoever reads it to the wrong fix.
                last_error = exc
                logger.warning(
                    "Groq timed out after %.0fs (attempt %s/2)",
                    self._settings.groq_timeout_seconds, attempt,
                )
                if attempt == 2:
                    raise LLMUnavailableError(
                        f"Groq did not respond within "
                        f"{self._settings.groq_timeout_seconds:g}s "
                        f"(model {self.model!r}) after 2 attempts."
                    ) from exc
                continue
            except httpx.TransportError as exc:
                last_error = exc
                if attempt == 2:
                    raise LLMUnavailableError(
                        f"Cannot reach Groq at {self._base_url}: {exc}"
                    ) from exc
                continue

            self.last_rate_limit = {
                key: value
                for key, value in response.headers.items()
                if key.lower().startswith("x-ratelimit")
            }

            if response.status_code in _RETRYABLE_STATUS and attempt == 1:
                # Honour Retry-After when Groq sends one; it knows better than a
                # fixed backoff how long the token bucket needs.
                delay = _retry_after_seconds(response) or 1.0
                logger.warning(
                    "Groq returned %s; retrying once in %.1fs",
                    response.status_code, delay,
                )
                time.sleep(delay)
                continue

            if response.status_code == 401:
                raise LLMUnavailableError(
                    "Groq rejected GROQ_API_KEY (401). Check the value in .env."
                )
            if response.status_code == 404:
                raise LLMUnavailableError(
                    f"Groq has no model {self.model!r} (404). Set GROQ_MODEL to "
                    "one the API lists at /openai/v1/models."
                )
            if response.status_code == 429:
                raise LLMUnavailableError(
                    f"Groq rate limit exceeded (429). "
                    f"{self.last_rate_limit or 'no rate-limit headers returned'}"
                )
            if response.status_code >= 400:
                raise LLMResponseError(
                    f"Groq returned {response.status_code}: {response.text[:500]}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise LLMResponseError(
                    f"Groq returned non-JSON body: {response.text[:200]!r}"
                ) from exc

        raise LLMUnavailableError(f"Groq request failed: {last_error!r}")

    def _content(self, payload: dict[str, Any]) -> str:
        try:
            choice = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Unexpected Groq response shape: {json.dumps(payload)[:300]}"
            ) from exc
        if not choice:
            raise LLMResponseError("Groq returned an empty message")

        usage = payload.get("usage") or {}
        logger.debug(
            "Groq %s: prompt=%s completion=%s",
            self.model, usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )
        return choice

    # --- the contract -----------------------------------------------------

    def complete(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = self._post(
            {
                "model": self.model,
                "temperature": 0,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": harden_schema(schema.model_json_schema()),
                    },
                },
            }
        )

        content = _strip_fence(self._content(payload))
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Groq returned non-JSON despite json_schema mode: {content[:200]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMResponseError(
                f"Expected a JSON object, got {type(data).__name__}"
            )
        return data

    def complete_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens

        return self._content(self._post(body)).strip()


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None

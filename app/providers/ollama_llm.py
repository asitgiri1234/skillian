"""Ollama-backed :class:`~app.providers.llm.LLMProvider`.

Uses Ollama's structured-output support: the pydantic model's JSON schema is
passed as ``format=``, so the decoder is constrained to emit conforming JSON. The
prompt therefore never has to say "reply with JSON" — see the note in
:meth:`OllamaLLMProvider.complete`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import ollama
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.providers.llm import (
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


class OllamaLLMProvider(LLMProvider):
    """Structured extraction via a locally running Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        settings: Settings | None = None,
        client: ollama.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = self._settings.ollama_llm_model
        # Injectable for tests, matching how AdzunaSource takes an httpx client.
        self._client = client or ollama.Client(
            host=self._settings.ollama_host,
            timeout=self._settings.ollama_timeout_seconds,
        )

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

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                # The schema constrains decoding at the token level. This is
                # strictly stronger than asking for JSON in the prompt: malformed
                # or extra-key output becomes unrepresentable rather than
                # merely discouraged.
                format=schema.model_json_schema(),
                options={
                    # Extraction is not a creative task — the same resume must
                    # produce the same parse, or the retry loop below and any
                    # cached result become meaningless.
                    "temperature": 0,
                },
            )
        except ollama.ResponseError as exc:
            # 404 here means the model was never pulled, which is by far the most
            # common first-run failure; say so instead of surfacing a bare 404.
            if getattr(exc, "status_code", None) == 404:
                raise LLMUnavailableError(
                    f"Ollama has no model {self.model!r}. Run: ollama pull {self.model}"
                ) from exc
            raise LLMUnavailableError(f"Ollama error: {exc}") from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise LLMUnavailableError(
                f"Cannot reach the Ollama daemon at {self._settings.ollama_host}. "
                "Is it running? Try: ollama serve"
            ) from exc

        content = response.message.content
        if not content:
            raise LLMResponseError("Ollama returned an empty message")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            # Should be unreachable while format= is honoured; kept because a
            # silent fallback to unconstrained decoding would otherwise surface
            # as a confusing validation error two layers up.
            raise LLMResponseError(
                f"Ollama returned non-JSON despite a constrained format: {content[:200]!r}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMResponseError(
                f"Expected a JSON object, got {type(data).__name__}"
            )
        return data

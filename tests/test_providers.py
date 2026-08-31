"""Provider error mapping. No daemon needed — the client is injected.

These exist because the *message* is the product here. A provider that fails
with the wrong explanation sends an operator to fix something that is not broken,
and that is exactly what happened during day-3 verification: a resume extraction
that ran past the timeout reported "Is it running? Try: ollama serve" about a
daemon that was up and answering.
"""

from __future__ import annotations

from typing import Any

import httpx
import ollama
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.providers import (
    EmbeddingUnavailableError,
    LLMResponseError,
    LLMUnavailableError,
    OllamaEmbeddingProvider,
    OllamaLLMProvider,
)


class Tiny(BaseModel):
    value: str | None


class _RaisingClient:
    """Stands in for ollama.Client, raising a chosen exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def chat(self, **kwargs: Any) -> Any:
        raise self._exc

    def embed(self, **kwargs: Any) -> Any:
        raise self._exc


class _ReplyingClient:
    def __init__(self, content: str) -> None:
        self._content = content

    def chat(self, **kwargs: Any) -> Any:
        class _Message:
            content = self._content

        class _Response:
            message = _Message()

        return _Response()


@pytest.fixture
def settings() -> Settings:
    return Settings(ollama_timeout_seconds=42.0)


class TestTimeoutIsNotUnreachable:
    def test_llm_timeout_names_the_timeout_not_the_daemon(self, settings) -> None:
        provider = OllamaLLMProvider(
            settings=settings, client=_RaisingClient(httpx.ReadTimeout("too slow"))
        )
        with pytest.raises(LLMUnavailableError) as excinfo:
            provider.complete("hello", Tiny)

        message = str(excinfo.value)
        assert "42s" in message
        assert "OLLAMA_TIMEOUT_SECONDS" in message
        # The wrong advice must not appear: the daemon is up.
        assert "ollama serve" not in message

    def test_llm_timeout_is_caught_before_transport_error(self, settings) -> None:
        """httpx.TimeoutException subclasses TransportError, so ordering of the
        except clauses is the whole fix. This pins it."""
        assert issubclass(httpx.ReadTimeout, httpx.TransportError)
        provider = OllamaLLMProvider(
            settings=settings, client=_RaisingClient(httpx.ConnectTimeout("slow"))
        )
        with pytest.raises(LLMUnavailableError, match="OLLAMA_TIMEOUT_SECONDS"):
            provider.complete("hello", Tiny)

    def test_embedding_timeout_names_the_batch(self, settings) -> None:
        provider = OllamaEmbeddingProvider(
            settings=settings, client=_RaisingClient(httpx.ReadTimeout("too slow"))
        )
        with pytest.raises(EmbeddingUnavailableError) as excinfo:
            provider.embed_batch(["a", "b", "c"])

        message = str(excinfo.value)
        assert "3 text(s)" in message
        assert "ollama serve" not in message


class TestConnectionFailureStillSaysSoStartTheDaemon:
    def test_llm_connect_error(self, settings) -> None:
        provider = OllamaLLMProvider(
            settings=settings, client=_RaisingClient(httpx.ConnectError("refused"))
        )
        with pytest.raises(LLMUnavailableError, match="ollama serve"):
            provider.complete("hello", Tiny)

    def test_embedding_connect_error(self, settings) -> None:
        provider = OllamaEmbeddingProvider(
            settings=settings, client=_RaisingClient(httpx.ConnectError("refused"))
        )
        with pytest.raises(EmbeddingUnavailableError, match="ollama serve"):
            provider.embed("hello")


class TestMissingModel:
    def test_404_suggests_pulling_it(self, settings) -> None:
        error = ollama.ResponseError("model not found")
        error.status_code = 404
        provider = OllamaLLMProvider(settings=settings, client=_RaisingClient(error))
        with pytest.raises(LLMUnavailableError, match="ollama pull"):
            provider.complete("hello", Tiny)


class TestCompleteText:
    def test_returns_stripped_prose(self, settings) -> None:
        provider = OllamaLLMProvider(
            settings=settings, client=_ReplyingClient("  hello world  ")
        )
        assert provider.complete_text("say hi") == "hello world"

    def test_empty_reply_is_a_response_error(self, settings) -> None:
        provider = OllamaLLMProvider(settings=settings, client=_ReplyingClient(""))
        with pytest.raises(LLMResponseError, match="empty"):
            provider.complete_text("say hi")

    def test_complete_rejects_non_json(self, settings) -> None:
        """Unreachable while format= is honoured, but a silent fallback to
        unconstrained decoding must not surface two layers up as a confusing
        validation error."""
        provider = OllamaLLMProvider(
            settings=settings, client=_ReplyingClient("not json at all")
        )
        with pytest.raises(LLMResponseError, match="non-JSON"):
            provider.complete("hello", Tiny)

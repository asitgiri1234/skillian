"""Groq provider and the Ollama fallback. No network — httpx is mocked.

The two things worth pinning here are the ones that were found by probing the
live API rather than by reading docs:

* Groq rejects pydantic's schema unless every object carries
  ``additionalProperties: false``. That is a schema-shape requirement, not a
  missing feature, and it would be very easy to misread the resulting 400 as
  "json_schema unsupported" and silently downgrade to loose JSON mode.
* A resume upload must not fail because the network is down.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.providers import (
    FallbackLLMProvider,
    GroqLLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)
from app.providers.groq_llm import harden_schema
from app.structure import ParsedResume


class Tiny(BaseModel):
    value: str | None


@pytest.fixture
def settings() -> Settings:
    return Settings(groq_api_key="gsk_test", groq_model="qwen/qwen3.8-27b")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(content: str, usage: dict[str, Any] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
            },
            headers={"x-ratelimit-limit-tokens": "8000"},
        )

    return handler


class TestHardenSchema:
    def test_sets_additional_properties_false_on_every_object(self) -> None:
        """The exact thing Groq 400s on. Nested $defs included — the live error
        named /$defs/ProjectRef, not the root."""
        schema = harden_schema(ParsedResume.model_json_schema())
        assert schema["additionalProperties"] is False
        for name, definition in schema.get("$defs", {}).items():
            assert definition["additionalProperties"] is False, name

    def test_marks_every_property_required(self) -> None:
        """OpenAI-compatible strict mode demands it. A no-op for our models,
        whose fields are all required-but-nullable by design."""
        schema = harden_schema(ParsedResume.model_json_schema())
        assert set(schema["required"]) == set(schema["properties"])
        for definition in schema.get("$defs", {}).values():
            assert set(definition["required"]) == set(definition["properties"])

    def test_is_idempotent(self) -> None:
        once = harden_schema(ParsedResume.model_json_schema())
        twice = harden_schema(json.loads(json.dumps(once)))
        assert once == twice


class TestComplete:
    def test_sends_json_schema_mode_not_json_object(self, settings) -> None:
        """The whole point of choosing Groq's strict mode: shape stays
        guaranteed, so structure.py's retry loop covers accuracy only."""
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return _ok('{"value": "x"}')(request)

        provider = GroqLLMProvider(settings=settings, client=_client(handler))
        provider.complete("hi", Tiny)

        fmt = seen["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"]["additionalProperties"] is False
        assert seen["temperature"] == 0

    def test_strips_markdown_fences(self, settings) -> None:
        """Models fence JSON even under a constrained decoder; a fence would
        otherwise fail json.loads with a message about the wrong thing."""
        provider = GroqLLMProvider(
            settings=settings,
            client=_client(_ok('```json\n{"value": "x"}\n```')),
        )
        assert provider.complete("hi", Tiny) == {"value": "x"}

    def test_rejects_a_non_object_payload(self, settings) -> None:
        provider = GroqLLMProvider(settings=settings, client=_client(_ok("[1, 2]")))
        with pytest.raises(LLMResponseError, match="Expected a JSON object"):
            provider.complete("hi", Tiny)

    def test_rejects_non_json(self, settings) -> None:
        provider = GroqLLMProvider(settings=settings, client=_client(_ok("not json")))
        with pytest.raises(LLMResponseError, match="non-JSON"):
            provider.complete("hi", Tiny)

    def test_records_rate_limit_headers(self, settings) -> None:
        """Free-tier limits are low enough to matter before a demo."""
        provider = GroqLLMProvider(
            settings=settings, client=_client(_ok('{"value": "x"}'))
        )
        provider.complete("hi", Tiny)
        assert provider.last_rate_limit["x-ratelimit-limit-tokens"] == "8000"


class TestErrors:
    def test_missing_key_names_the_setting(self) -> None:
        provider = GroqLLMProvider(settings=Settings(groq_api_key=None))
        with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
            provider.complete("hi", Tiny)

    def test_401_is_unavailable_not_a_response_error(self, settings) -> None:
        provider = GroqLLMProvider(
            settings=settings,
            client=_client(lambda r: httpx.Response(401, json={"error": "bad key"})),
        )
        with pytest.raises(LLMUnavailableError, match="401"):
            provider.complete("hi", Tiny)

    def test_404_suggests_listing_models(self, settings) -> None:
        """The model list moves — llama-3.3-70b-versatile was gone by the time
        this was written."""
        provider = GroqLLMProvider(
            settings=settings,
            client=_client(lambda r: httpx.Response(404, json={"error": "no model"})),
        )
        with pytest.raises(LLMUnavailableError, match="/openai/v1/models"):
            provider.complete("hi", Tiny)

    def test_retries_once_on_500_then_succeeds(self, settings) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, json={"error": "boom"})
            return _ok('{"value": "x"}')(request)

        provider = GroqLLMProvider(settings=settings, client=_client(handler))
        assert provider.complete("hi", Tiny) == {"value": "x"}
        assert calls["n"] == 2

    def test_retries_once_and_no_more(self, settings) -> None:
        """Stacking retries multiplies worst-case latency, which is the entire
        reason for using a hosted provider."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={"error": "boom"})

        provider = GroqLLMProvider(settings=settings, client=_client(handler))
        with pytest.raises((LLMUnavailableError, LLMResponseError)):
            provider.complete("hi", Tiny)
        assert calls["n"] == 2

    def test_timeout_says_timeout_not_unreachable(self, settings) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        provider = GroqLLMProvider(settings=settings, client=_client(handler))
        with pytest.raises(LLMUnavailableError, match="did not respond within"):
            provider.complete("hi", Tiny)


class _Boom:
    name = "boom"
    model = "boom"

    def complete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise LLMUnavailableError("network down")

    def complete_text(self, *args: Any, **kwargs: Any) -> str:
        raise LLMUnavailableError("network down")


class _Fine:
    name = "fine"
    model = "fine"

    def complete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"value": "local"}

    def complete_text(self, *args: Any, **kwargs: Any) -> str:
        return "local prose"


class TestFallback:
    def test_uses_the_primary_when_it_works(self) -> None:
        provider = FallbackLLMProvider(primary=_Fine(), fallback=_Boom())
        assert provider.complete("hi", Tiny) == {"value": "local"}
        assert provider.fallback_count == 0

    def test_falls_back_on_llm_error(self) -> None:
        """A resume upload must not fail because the network is down."""
        provider = FallbackLLMProvider(primary=_Boom(), fallback=_Fine())
        assert provider.complete("hi", Tiny) == {"value": "local"}
        assert provider.fallback_count == 1

    def test_falls_back_for_free_text_too(self) -> None:
        provider = FallbackLLMProvider(primary=_Boom(), fallback=_Fine())
        assert provider.complete_text("hi") == "local prose"

    def test_logs_the_fallback_at_warning(self, caplog) -> None:
        """Silent degradation to a 30x slower path is exactly what goes
        unnoticed until a demo."""
        provider = FallbackLLMProvider(primary=_Boom(), fallback=_Fine())
        with caplog.at_level("WARNING"):
            provider.complete("hi", Tiny)
        assert any(
            "falling back" in record.message.lower()
            or "falling back" in record.getMessage().lower()
            for record in caplog.records
        )

    def test_is_per_call_not_per_process(self) -> None:
        """A transient outage must not pin the app to the slow path forever."""
        provider = FallbackLLMProvider(primary=_Boom(), fallback=_Fine())
        provider.complete("hi", Tiny)
        provider.complete("hi", Tiny)
        assert provider.fallback_count == 2

    def test_a_fallback_failure_propagates(self) -> None:
        provider = FallbackLLMProvider(primary=_Boom(), fallback=_Boom())
        with pytest.raises(LLMUnavailableError):
            provider.complete("hi", Tiny)


class TestParseProviderWiring:
    def test_defaults_to_groq_with_ollama_fallback(self) -> None:
        from app.providers import get_parse_provider

        provider = get_parse_provider(Settings(groq_api_key="gsk_test"))
        assert isinstance(provider, FallbackLLMProvider)
        assert provider.primary.name == "groq"
        assert provider.fallback.name == "ollama"

    def test_no_wrapper_when_parse_and_llm_agree(self) -> None:
        """Wrapping ollama in a fallback to ollama would add a layer that can
        never fire."""
        from app.providers import get_parse_provider

        provider = get_parse_provider(Settings(parse_provider="ollama"))
        assert provider.name == "ollama"

    def test_unknown_provider_lists_the_options(self) -> None:
        from app.providers import get_parse_provider

        with pytest.raises(ValueError, match="Available"):
            get_parse_provider(Settings(parse_provider="nope"))

    def test_embeddings_stay_on_ollama(self) -> None:
        """Groq serves no embeddings, and local batched embeddings measured
        0.51s/chunk."""
        from app.providers import get_embedding_provider

        assert get_embedding_provider(Settings()).name == "ollama"

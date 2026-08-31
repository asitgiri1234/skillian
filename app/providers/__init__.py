"""Provider registry.

``LLM_PROVIDER`` and ``EMBEDDING_PROVIDER`` in .env select the implementation.
Callers ask for the ABC and never import a concrete provider, so adding a hosted
backend later is a new file plus a registry entry.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.embeddings import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingUnavailableError,
)
from app.providers.llm import (
    LLMError,
    LLMProvider,
    LLMResponseError,
    LLMUnavailableError,
)
from app.providers.fallback import FallbackLLMProvider
from app.providers.groq_llm import GroqLLMProvider
from app.providers.ollama_embed import OllamaEmbeddingProvider
from app.providers.ollama_llm import OllamaLLMProvider

# Explicit dicts rather than package scanning: an unknown flag should fail with a
# list of what is valid, not with an ImportError from a mistyped module name.
LLM_PROVIDERS: dict[str, type[LLMProvider]] = {
    OllamaLLMProvider.name: OllamaLLMProvider,
    GroqLLMProvider.name: GroqLLMProvider,
}

EMBEDDING_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    OllamaEmbeddingProvider.name: OllamaEmbeddingProvider,
}


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the LLM provider named by ``LLM_PROVIDER``."""
    settings = settings or get_settings()
    key = settings.llm_provider.strip().lower()
    if key not in LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not registered. "
            f"Available: {', '.join(sorted(LLM_PROVIDERS))}"
        )
    return LLM_PROVIDERS[key](settings=settings)


def get_parse_provider(settings: Settings | None = None) -> LLMProvider:
    """The provider used for *resume parsing*, with an automatic local fallback.

    Separate from :func:`get_llm_provider` on purpose. Resume parsing is one
    call per upload and is the slowest thing a user waits on, so it gets the
    hosted provider. Job-skill extraction and explanations run per-job inside a
    search — hundreds of calls — and stay on ``LLM_PROVIDER`` because Groq's
    free tier is metered in tokens per minute. See DECISIONS 22.4.

    When ``PARSE_PROVIDER`` names something other than the LLM provider, the
    result is wrapped so that a failure of the primary degrades to the local
    model for that call rather than failing the upload.
    """
    settings = settings or get_settings()
    parse_key = settings.parse_provider.strip().lower()
    if parse_key not in LLM_PROVIDERS:
        raise ValueError(
            f"PARSE_PROVIDER={settings.parse_provider!r} is not registered. "
            f"Available: {', '.join(sorted(LLM_PROVIDERS))}"
        )

    primary = LLM_PROVIDERS[parse_key](settings=settings)
    fallback_key = settings.llm_provider.strip().lower()
    if fallback_key == parse_key:
        # Nothing to fall back to; wrapping would only add a layer that can
        # never fire.
        return primary
    return FallbackLLMProvider(
        primary=primary, fallback=LLM_PROVIDERS[fallback_key](settings=settings)
    )


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the embedding provider named by ``EMBEDDING_PROVIDER``."""
    settings = settings or get_settings()
    key = settings.embedding_provider.strip().lower()
    if key not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"EMBEDDING_PROVIDER={settings.embedding_provider!r} is not registered. "
            f"Available: {', '.join(sorted(EMBEDDING_PROVIDERS))}"
        )
    return EMBEDDING_PROVIDERS[key](settings=settings)


__all__ = [
    "EMBEDDING_PROVIDERS",
    "LLM_PROVIDERS",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingUnavailableError",
    "LLMError",
    "LLMProvider",
    "LLMResponseError",
    "LLMUnavailableError",
    "FallbackLLMProvider",
    "GroqLLMProvider",
    "OllamaEmbeddingProvider",
    "OllamaLLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
    "get_parse_provider",
]

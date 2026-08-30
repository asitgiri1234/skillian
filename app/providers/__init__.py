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
from app.providers.ollama_embed import OllamaEmbeddingProvider
from app.providers.ollama_llm import OllamaLLMProvider

# Explicit dicts rather than package scanning: an unknown flag should fail with a
# list of what is valid, not with an ImportError from a mistyped module name.
LLM_PROVIDERS: dict[str, type[LLMProvider]] = {
    OllamaLLMProvider.name: OllamaLLMProvider,
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
    "OllamaEmbeddingProvider",
    "OllamaLLMProvider",
    "get_embedding_provider",
    "get_llm_provider",
]

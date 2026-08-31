"""Application settings, loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of the process environment.

    Every field is validated at import of the first ``get_settings()`` call, so a
    missing credential fails at startup with a readable error instead of at the
    first HTTP request halfway through an ingestion run.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unrelated vars (PATH, CI runner noise) rather than erroring.
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database ---------------------------------------------------------
    database_url: str = Field(
        default="postgresql+psycopg://skillian:skillian@localhost:5432/skillian",
        description="SQLAlchemy URL. Must use the postgresql+psycopg (psycopg 3) driver.",
    )

    # --- Adzuna -----------------------------------------------------------
    # Optional so that `import app.config` works in CI/test runs that never make
    # a network call; AdzunaSource raises a clear error if they are missing.
    adzuna_app_id: str | None = None
    adzuna_app_key: str | None = None
    adzuna_country: str = "in"

    # --- HTTP -------------------------------------------------------------
    http_timeout_seconds: float = 20.0
    http_max_retries: int = 4
    http_backoff_base_seconds: float = 0.5

    # --- Ingestion --------------------------------------------------------
    # Adzuna rejects results_per_page > 50, so this is a hard ceiling, not taste.
    results_per_page: int = Field(default=50, ge=1, le=50)
    max_pages: int = Field(default=5, ge=1)

    # --- Providers --------------------------------------------------------
    # Selects the implementation in app/providers/. Everything downstream depends
    # on the ABC, so swapping to a hosted model is a config change, not a code
    # change. Validated against the registry at construction time.
    llm_provider: str = "ollama"
    embedding_provider: str = "ollama"

    # --- Ollama -----------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    # 7b, not 3b: extraction quality on messy resume text is the bottleneck here,
    # and this runs locally where a slower model costs time rather than money.
    ollama_llm_model: str = "qwen2.5:7b"
    # 768-dimensional. Must agree with models.EMBEDDING_DIM.
    ollama_embed_model: str = "nomic-embed-text"
    # A 7b model on CPU can take well over a minute for a long resume — measured
    # at 170s here, which made the previous 180s default a coin flip. Raised to
    # 300 after a verification run timed out mid-extraction.
    ollama_timeout_seconds: float = 300.0

    # --- Groq -------------------------------------------------------------
    # Hosted inference. Unlike Ollama, using it sends resume text off this
    # machine — a deliberate exception to the local-only premise, taken because
    # extraction drops from ~101s to ~3s. Optional: with no key, PARSE_PROVIDER
    # falls back to Ollama on every call.
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Chosen 2026-08-31 by listing /openai/v1/models and measuring all three
    # viable candidates. NOT llama-3.3-70b-versatile, which this account's API
    # does not offer — the model list moves, so verify before changing this.
    # qwen3.8 was fastest (2.5s) and, crucially, emits no reasoning tokens: the
    # gpt-oss models spent 894-996 completion tokens on hidden reasoning against
    # an 8000 tokens/minute limit. See DECISIONS 22.3.
    groq_model: str = "qwen/qwen3.8-27b"
    # Hosted and fast; a long ceiling here would only delay the Ollama fallback.
    groq_timeout_seconds: float = 60.0

    # --- Parsing ----------------------------------------------------------
    # Which provider parses resumes. Distinct from LLM_PROVIDER (which still
    # drives job-skill extraction and match explanations) because those run
    # per-job inside a search and would exhaust Groq's free-tier token budget;
    # resume parsing is one call per upload. See DECISIONS 22.4.
    # Falls back to Ollama automatically on any Groq failure.
    parse_provider: str = "groq"

    # --- Extraction -------------------------------------------------------
    # Attempts for the validate-and-retry loop in app/structure.py. Constrained
    # decoding fixes shape, not accuracy, so retries are about semantic failures.
    extraction_max_attempts: int = Field(default=3, ge=1, le=6)

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached because pydantic-settings re-reads and re-parses the .env file on every
    instantiation; the CLI, ingest orchestrator and sources all ask for settings.
    """
    return Settings()

"""Shared fixtures.

The suite splits in two, and the split is deliberate:

* Everything in ``test_scorer.py`` and ``test_chunking.py`` is pure and runs
  anywhere with no services. That is by design — the scorer was written to have
  no I/O precisely so its arithmetic could be pinned down exhaustively.
* ``test_pipeline.py`` needs a real Postgres, because the things worth testing
  there (ON CONFLICT upserts, the ``xmax = 0`` new-row trick, pgvector columns,
  JSONB round-trips) are Postgres behaviours that a SQLite stand-in would fake
  rather than verify. It skips cleanly when the database is unreachable.

No test calls Ollama. The LLM and embedding providers are injected everywhere
they are used, and the fakes below are what gets injected — a suite that needed
a 7b model would take an hour and would be testing the model, not the code.
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest
from pydantic import BaseModel

from app.models import EMBEDDING_DIM
from app.providers import EmbeddingProvider, LLMProvider


# --- fakes ------------------------------------------------------------------


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings with no model behind them.

    Hashes the text into a few dimensions and normalises. Two identical strings
    embed identically and two different ones do not, which is every property the
    pipeline actually relies on. It makes no claim to be *semantically*
    meaningful — nothing in the suite asserts that similar text scores highly,
    because that would be testing nomic-embed-text.
    """

    name = "fake"
    dimension = EMBEDDING_DIM

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for index, token in enumerate(text.split()):
            vector[hash(token) % self.dimension] += 1.0
            vector[(index * 7) % self.dimension] += 0.5
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            # An all-zero vector is a valid write but undefined under cosine;
            # give it a single nonzero component instead.
            vector[0] = 1.0
            norm = 1.0
        return [value / norm for value in vector]

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        self.calls.append([text])
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]


class FakeLLMProvider(LLMProvider):
    """Returns canned objects and counts calls.

    The call counters are load-bearing: the pipeline's central performance claim
    is that scoring makes *no* model calls, and ``llm.complete_calls`` is how the
    suite proves it rather than asserting it in a comment.
    """

    name = "fake"

    def __init__(
        self,
        skills_response: dict[str, Any] | None = None,
        text_response: str = "You match this role on Python and PostgreSQL.",
    ) -> None:
        self.skills_response = skills_response or {
            "skills": [
                {"name": "Python", "requirement": "required"},
                {"name": "PostgreSQL", "requirement": "required"},
                {"name": "Kubernetes", "requirement": "preferred"},
            ]
        }
        self.text_response = text_response
        self.complete_calls: list[str] = []
        self.complete_text_calls: list[str] = []

    def complete(
        self, prompt: str, schema: type[BaseModel], *, system: str | None = None
    ) -> dict[str, Any]:
        self.complete_calls.append(prompt)
        return self.skills_response

    def complete_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.complete_text_calls.append(prompt)
        return self.text_response


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


# --- vector helpers ---------------------------------------------------------


@pytest.fixture
def unit_vector():
    """A one-hot unit vector of the schema's width, for exact cosine assertions."""

    def _make(index: int = 0, dim: int = 8) -> list[float]:
        vector = [0.0] * dim
        vector[index] = 1.0
        return vector

    return _make


# --- database ---------------------------------------------------------------


@pytest.fixture(scope="session")
def db_available() -> bool:
    """True when the configured Postgres is reachable and migrated."""
    try:
        from sqlalchemy import text

        from app.db import engine

        with engine.connect() as connection:
            connection.execute(text("SELECT 1 FROM job_chunks LIMIT 1"))
        return True
    except Exception:  # noqa: BLE001 - any failure means "cannot run these tests"
        return False


@pytest.fixture
def db_session(db_available: bool):
    """A session on the real database, rolled back to a savepoint afterwards.

    Everything the test writes happens inside one transaction that is never
    committed, so the suite leaves no rows behind even though the pipeline calls
    ``session.commit()`` at every stage — the joining session turns those into
    nested-transaction releases, not real commits.
    """
    if not db_available:
        pytest.skip(
            "Postgres is not reachable or not migrated to 0003. "
            "Run: docker compose up -d && alembic upgrade head"
        )

    from app.db import engine
    from sqlalchemy.orm import Session

    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def session_factory(db_session):
    """A ``sessionmaker``-shaped callable handing out the test's own session.

    The pipeline takes a session factory so it can open its own session and its
    own recovery session. Both must land on the *test's* connection or their
    writes would be invisible to assertions and would escape the rollback.
    """

    class _Factory:
        def __call__(self, *args: Any, **kwargs: Any):
            return _NonClosingSession(db_session)

        # run_search's failure path uses `with session_factory() as s:`
        def __enter__(self):
            return _NonClosingSession(db_session)

        def __exit__(self, *args: Any) -> None:
            return None

    return _Factory()


class _NonClosingSession:
    """Proxies to the test session but ignores close(), which the test owns."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NonClosingSession":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@pytest.fixture
def user_row(db_session):
    from app.models import User

    user = User(email=f"test-{uuid.uuid4().hex[:8]}@example.com", name="Test User")
    db_session.add(user)
    db_session.flush()
    return user

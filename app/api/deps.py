"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.sources.adzuna import AdzunaSource
from app.sources.base import JobSource

#: The sources a search may fetch from. Mirrors SOURCE_REGISTRY in
#: scripts/run_ingest.py — the CLI and the API deliberately keep their own
#: registries so that enabling a source for scheduled ingestion is not the same
#: decision as exposing it to an HTTP caller.
SOURCE_REGISTRY: dict[str, type[JobSource]] = {
    AdzunaSource.name: AdzunaSource,
}


def get_session() -> Iterator[Session]:
    """Request-scoped session, closed when the response is done.

    Note the contrast with the background task, which builds its *own* session
    from SessionLocal: this one is closed the moment the POST /searches response
    is written, long before the search finishes.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def build_sources(names: list[str] | None) -> list[JobSource]:
    """Instantiate the requested sources, or all of them.

    Raises ValueError naming the valid options — the caller turns that into a
    422, which is more useful than a 500 from a KeyError.
    """
    selected = names or sorted(SOURCE_REGISTRY)
    unknown = [name for name in selected if name not in SOURCE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown source(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(SOURCE_REGISTRY))}"
        )
    return [SOURCE_REGISTRY[name]() for name in selected]

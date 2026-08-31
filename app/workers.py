"""Background execution for search runs, on FastAPI's ``BackgroundTasks``.

No Celery, no Redis, no broker. A search run is a single long function on one
machine with no fan-out, no cross-process scheduling and no retry semantics worth
speaking of — the state a queue would manage is already in ``ingestion_runs``,
which the pipeline updates stage by stage and which survives a process restart in
a way an in-memory Celery result never would. A broker here would add two
services, a serialization boundary and a deployment story in exchange for
nothing this workload needs. See DECISIONS 16.7.

What is genuinely given up: a task dies with the process. That is visible rather
than silent — the run row is left at ``running`` with a stale stage, exactly the
signal day 1 established for an abandoned run — and it is the thing to fix when
this outgrows one machine.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from app.db import SessionLocal
from app.matching.pipeline import SearchFilters, run_search
from app.providers import (
    EmbeddingProvider,
    LLMProvider,
    get_embedding_provider,
    get_llm_provider,
)
from app.sources.base import JobSource

logger = logging.getLogger(__name__)


def run_search_task(
    run_id: UUID,
    resume_id: UUID,
    filters: SearchFilters,
    sources: Sequence[JobSource],
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    session_factory: sessionmaker[Session] = SessionLocal,
) -> None:
    """Run one search to completion. Never raises.

    The single rule for a BackgroundTasks callable: it must not propagate. An
    exception escaping here is logged by Starlette and then lost, whereas
    ``run_search`` has already written the reason to ``ingestion_runs.error``
    before re-raising — so this catches, logs, and returns. The durable record is
    the row, and the client polling GET /runs/{id} reads it there.

    Providers are constructed here rather than in the request handler so that a
    dead Ollama daemon fails the *run* (recorded, pollable, with a message that
    names the fix) instead of the POST that queued it.
    """
    try:
        outcome = run_search(
            run_id=run_id,
            resume_id=resume_id,
            filters=filters,
            sources=sources,
            llm=llm or get_llm_provider(),
            embedder=embedder or get_embedding_provider(),
            session_factory=session_factory,
        )
        logger.info(
            "Background search %s completed: status=%s jobs=%s matches=%s",
            run_id, outcome.status, outcome.jobs_found, outcome.matches_written,
        )
    except Exception:  # noqa: BLE001 - terminal boundary; run_search already recorded it
        logger.exception("Background search %s failed", run_id)
    finally:
        # Sources hold an httpx.Client each. Nothing else will close them: the
        # task owns them for its whole lifetime and then the reference is gone.
        for source in sources:
            close = getattr(source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup must not mask anything
                    logger.warning("Failed to close source %s", source.name)

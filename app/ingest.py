"""Ingestion orchestration: fetch from sources, upsert into ``jobs``, record the run.

The contract this module keeps: an ``ingestion_runs`` row is created before any
network call and is *always* closed out — with ``success``, ``partial`` or
``failed`` and an error string. A run left in ``running`` means the process was
killed, which is itself useful information.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import literal_column, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from app.db import SessionLocal
from app.models import IngestionRun, Job
from app.sources.base import JobSource, NormalizedJob, SearchQuery

logger = logging.getLogger(__name__)

# Written to ingestion_runs.status.
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

# Columns refreshed when a job we already have comes back from a source.
# Deliberately excludes id, source, source_job_id (identity) and embedding — an
# embedding costs an API call, so a re-fetch must not null it out.
_UPSERT_FIELDS: tuple[str, ...] = (
    "dedup_hash", "title", "company", "location", "is_remote", "description",
    "apply_url", "salary_raw", "salary_min", "salary_max", "salary_currency",
    "salary_period", "experience_raw", "experience_min_years", "posted_date",
    "fetched_at",
)


@dataclass(frozen=True)
class StoredJob:
    """One job as persisted, plus whether this run was the first to see it."""

    job_id: UUID
    source: str
    source_job_id: str
    title: str
    company: str | None
    location: str | None
    salary_raw: str | None
    is_new: bool


@dataclass
class IngestionResult:
    """Outcome of a run, for the CLI to render and for callers to assert on."""

    run_id: UUID
    status: str
    jobs_found: int = 0
    stored: list[StoredJob] = field(default_factory=list)
    error: str | None = None
    # Per-source failures that did not sink the whole run (status "partial").
    source_errors: dict[str, str] = field(default_factory=dict)

    @property
    def new_count(self) -> int:
        return sum(1 for job in self.stored if job.is_new)

    @property
    def duplicate_count(self) -> int:
        return sum(1 for job in self.stored if not job.is_new)


def _now() -> datetime:
    """Timezone-aware UTC. Naive datetimes into a timestamptz column silently
    take on the server's timezone, which makes runs unorderable across machines."""
    return datetime.now(timezone.utc)


def _create_run(
    session: Session, sources: Sequence[str], resume_id: UUID | None
) -> IngestionRun:
    run = IngestionRun(
        resume_id=resume_id,
        status=STATUS_RUNNING,
        sources=list(sources),
        jobs_found=0,
        started_at=_now(),
    )
    session.add(run)
    # Flush + commit before fetching: if the process dies mid-fetch, the row is
    # already durable and the abandoned run is visible as status="running".
    session.commit()
    return run


def _upsert_job(session: Session, job: NormalizedJob) -> StoredJob:
    """Insert or refresh one job, keyed on (source, source_job_id).

    Uses INSERT ... ON CONFLICT DO UPDATE rather than SELECT-then-INSERT: a single
    statement is atomic, so two ingestions racing on the same posting cannot both
    decide it is new and trip the unique constraint.
    """
    values: dict[str, Any] = {
        "source": job.source,
        "source_job_id": job.source_job_id,
        "dedup_hash": job.dedup_hash,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "is_remote": job.is_remote,
        "description": job.description,
        "apply_url": job.apply_url,
        "salary_raw": job.salary_raw,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_currency": job.salary_currency,
        "salary_period": job.salary_period,
        "experience_raw": job.experience_raw,
        "experience_min_years": job.experience_min_years,
        "posted_date": job.posted_date,
        "fetched_at": _now(),
    }

    stmt = pg_insert(Job).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_jobs_source_source_job_id",
        set_={name: getattr(stmt.excluded, name) for name in _UPSERT_FIELDS},
    ).returning(
        Job.id,
        # Postgres sets xmax to 0 on a fresh insert and to the updating
        # transaction's id on a conflict update. This is the only way to learn
        # which branch fired without a second round-trip.
        literal_column("(xmax = 0)").label("inserted"),
    )

    row = session.execute(stmt).one()
    return StoredJob(
        job_id=row.id,
        source=job.source,
        source_job_id=job.source_job_id,
        title=job.title,
        company=job.company,
        location=job.location,
        salary_raw=job.salary_raw,
        is_new=bool(row.inserted),
    )


def _finalize_run(
    session: Session,
    run_id: UUID,
    status: str,
    jobs_found: int,
    error: str | None,
) -> None:
    """Close out the run row with a Core UPDATE.

    Core rather than the ORM so this works on a session whose identity map we do
    not trust — the failure path calls it from a brand-new session.
    """
    session.execute(
        update(IngestionRun)
        .where(IngestionRun.id == run_id)
        .values(
            status=status,
            jobs_found=jobs_found,
            # Truncated: a driver traceback can be enormous and this column is
            # for triage, not for storing full logs.
            error=error[:4000] if error else None,
            finished_at=_now(),
        )
    )
    session.commit()


def run_ingestion(
    query: SearchQuery,
    sources: Sequence[JobSource],
    resume_id: UUID | None = None,
    session_factory: sessionmaker[Session] = SessionLocal,
) -> IngestionResult:
    """Fetch from every source, upsert the results, and record the run.

    One source failing does not abort the others: its error is recorded and the
    run finishes as ``partial``. Only a failure outside the per-source loop
    (database down, bad query) marks the run ``failed``.

    Raises whatever went wrong *after* writing it to the run row — the caller
    still gets a non-zero exit, but the database never loses the reason.
    """
    session = session_factory()
    source_names = [source.name for source in sources]
    run: IngestionRun | None = None

    try:
        run = _create_run(session, source_names, resume_id)
        logger.info(
            "Ingestion run %s started: query=%r location=%r sources=%s",
            run.id, query.keywords, query.location, source_names,
        )

        result = IngestionResult(run_id=run.id, status=STATUS_RUNNING)

        for source in sources:
            try:
                fetched = source.fetch(query)
            except Exception as exc:  # noqa: BLE001 - one bad source must not sink the run
                # repr, not str: some httpx/psycopg exceptions stringify to "".
                logger.exception("Source %s failed", source.name)
                result.source_errors[source.name] = repr(exc)
                continue

            logger.info("Source %s returned %s jobs", source.name, len(fetched))
            for job in fetched:
                result.stored.append(_upsert_job(session, job))

        result.jobs_found = len(result.stored)
        # Commit the jobs before touching the run row so a failure while
        # finalising cannot roll back the work that actually succeeded.
        session.commit()

        if result.source_errors and not result.stored:
            result.status = STATUS_FAILED
        elif result.source_errors:
            result.status = STATUS_PARTIAL
        else:
            result.status = STATUS_SUCCESS

        result.error = (
            "; ".join(f"{name}: {err}" for name, err in result.source_errors.items())
            or None
        )
        _finalize_run(session, run.id, result.status, result.jobs_found, result.error)
        logger.info(
            "Ingestion run %s finished: status=%s new=%s duplicate=%s",
            run.id, result.status, result.new_count, result.duplicate_count,
        )
        return result

    except Exception as exc:
        logger.exception("Ingestion run failed")
        if run is not None:
            _record_failure(session_factory, run.id, exc)
        raise
    finally:
        session.close()


def _record_failure(
    session_factory: sessionmaker[Session], run_id: UUID, exc: BaseException
) -> None:
    """Mark a run failed on a *fresh* session.

    The session that raised is very likely in an aborted transaction where every
    further statement errors with "current transaction is aborted", so reusing it
    would lose the error message we are trying to save.
    """
    try:
        with session_factory() as recovery_session:
            _finalize_run(
                recovery_session, run_id, STATUS_FAILED, jobs_found=0, error=repr(exc)
            )
    except Exception:  # noqa: BLE001 - last resort; never mask the original error
        # If even this fails the database is unreachable; log and let the
        # original exception propagate untouched.
        logger.exception("Could not record failure for ingestion run %s", run_id)

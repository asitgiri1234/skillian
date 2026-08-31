"""The search pipeline: resume in, scored and explained matches out.

Eight stages, run in one background task, each one recording its progress on the
``ingestion_runs`` row so a client polling ``GET /runs/{id}`` sees movement
rather than a flat "running" for four minutes:

    a. load the resume and its skills
    b. build search queries          (queries.py, LLM-free)
    c. fetch every enabled source, dedupe on dedup_hash, upsert
    d. extract job skills by dictionary lookup               [NO model calls]
    e. chunk and embed jobs with no chunks                   [embeddings, batched]
    f. score every job                                       [NO model calls]
    g. bulk write matches
    h. explain the top 20                                    [LLM, capped]

The ordering exists to keep model calls out of the wide part of the funnel.
Stages d and f both run over *every* job in the result set, and neither calls a
model. Stage d used to: one generative call per job at ~45 seconds, which made
an 80-job search an hour long, to read technology names out of a document and
write them back. It is now a compiled regex over the `skills` vocabulary and
costs milliseconds for the whole batch (DECISIONS 24). Stage f is set
intersection and a few thousand dot products.

**Stage h is the only per-item model call left, and it is capped at 20.**
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from app import runs as run_status
from app.db import SessionLocal
from app.ingest import upsert_job  # single definition of the jobs upsert
from app.matching.chunking import chunk_description
from app.matching.explain import MAX_EXPLANATIONS, explain_match
from app.matching.queries import build_search_queries
from app.matching.scorer import (
    SCORER_VERSION,
    JobPosting,
    JobSkillRef,
    ResumeProfile,
    ScoreResult,
    score,
)
from app.matching.jd_skills import extract_skills, get_index
from app.models import IngestionRun, Job, JobChunk, JobSkill, Match, Resume, ResumeSkill, Skill
from app.providers import EmbeddingProvider, LLMProvider
from app.sources.base import JobSource, NormalizedJob

logger = logging.getLogger(__name__)

#: Texts per embedding round-trip. Ollama's /api/embed takes a list, and the
#: round-trip dominates, but an unbounded batch would hold every chunk of every
#: job in memory and in one request body.
EMBED_BATCH_SIZE = 32


@dataclass
class SearchFilters:
    """The user-controllable part of a search."""

    location: str | None = None
    remote_only: bool = False
    max_results: int = 100


@dataclass
class SearchOutcome:
    """What a completed run did, for logging and for tests to assert on."""

    run_id: UUID
    status: str
    jobs_found: int = 0
    new_jobs: int = 0
    matches_written: int = 0
    explanations_written: int = 0
    chunks_written: int = 0
    error: str | None = None
    source_errors: dict[str, str] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _set_stage(
    session: Session,
    run_id: UUID,
    stage: str,
    *,
    status: str | None = None,
    jobs_found: int | None = None,
) -> None:
    """Record progress, committing immediately.

    Committed on its own rather than riding along with the stage's work: the
    entire point is that a *different* connection — the one serving
    GET /runs/{id} — can see it while the run is still going. Buffered inside the
    pipeline's transaction it would be invisible until the run ended, which is
    exactly when it stops being useful.
    """
    values: dict[str, Any] = {"stage": stage}
    if status is not None:
        values["status"] = status
    if jobs_found is not None:
        values["jobs_found"] = jobs_found
    session.execute(update(IngestionRun).where(IngestionRun.id == run_id).values(**values))
    session.commit()
    logger.info("Run %s -> stage=%s status=%s", run_id, stage, status or "(unchanged)")


def create_search_run(
    session: Session, resume_id: UUID, source_names: Sequence[str]
) -> IngestionRun:
    """Insert the ``queued`` run row and return it.

    Called synchronously from POST /searches, before the background task starts,
    so the endpoint can return a run_id the client can poll immediately. The row
    existing before any work begins is also what makes a crashed run visible.
    """
    run = IngestionRun(
        resume_id=resume_id,
        status=run_status.STATUS_QUEUED,
        stage=run_status.STAGE_QUEUED,
        sources=list(source_names),
        jobs_found=0,
        started_at=_now(),
    )
    session.add(run)
    session.commit()
    return run


# --- stage a ----------------------------------------------------------------


def _load_resume_profile(session: Session, resume_id: UUID) -> tuple[ResumeProfile, Resume]:
    resume = session.get(Resume, resume_id)
    if resume is None:
        raise LookupError(f"No resume with id {resume_id}")

    skill_ids = set(
        session.execute(
            select(ResumeSkill.skill_id).where(ResumeSkill.resume_id == resume_id)
        ).scalars()
    )
    parsed = resume.parsed or {}
    # Renamed from total_years_experience when the schema was trimmed; a
    # resume parsed before that change reads as None here and simply scores
    # with experience_multiplier == 1.0, which is the documented behaviour for
    # missing data rather than a wrong answer. See DECISIONS 20.5.
    years = parsed.get("total_experience_years")

    profile = ResumeProfile(
        resume_id=resume_id,
        embedding=list(resume.embedding) if resume.embedding is not None else None,
        skill_ids=frozenset(skill_ids),
        total_years_experience=float(years) if isinstance(years, (int, float)) else None,
    )
    if profile.embedding is None:
        # Not fatal: scoring degrades to skills-only rather than failing. Worth a
        # warning because it means every semantic score in this run will be 0.
        logger.warning(
            "Resume %s has no embedding; every semantic score will be 0", resume_id
        )
    return profile, resume


# --- stage c ----------------------------------------------------------------


def _fetch_and_store(
    session: Session,
    sources: Sequence[JobSource],
    queries: Sequence[Any],
    source_errors: dict[str, str],
) -> tuple[list[UUID], list[UUID]]:
    """Fetch every (source, query) pair, dedupe, upsert. Returns (all, new) ids.

    Deduping on ``dedup_hash`` here is a departure from the day-1 CLI, which
    stores one row per board deliberately (each copy has its own apply_url). Both
    are right for their context: showing a user the same job three times is bad,
    and every extra copy costs a full local-model skill extraction. Identity is
    still (source, source_job_id) in the table — this only decides what a single
    run bothers to write. See DECISIONS 18.5.
    """
    seen_hashes: set[str] = set()
    fetched: list[NormalizedJob] = []

    for source in sources:
        for query in queries:
            try:
                results = source.fetch(query)
            except Exception as exc:  # noqa: BLE001 - one bad source must not sink the run
                logger.exception("Source %s failed on %r", source.name, query.keywords)
                # Keep the first error per source: later ones are usually the
                # same cause repeated once per query.
                source_errors.setdefault(source.name, repr(exc))
                continue

            for job in results:
                if job.dedup_hash in seen_hashes:
                    continue
                seen_hashes.add(job.dedup_hash)
                fetched.append(job)

    all_ids: list[UUID] = []
    new_ids: list[UUID] = []
    for job in fetched:
        stored = upsert_job(session, job)
        all_ids.append(stored.job_id)
        if stored.is_new:
            new_ids.append(stored.job_id)

    session.commit()
    logger.info("Stored %s jobs (%s new)", len(all_ids), len(new_ids))
    return all_ids, new_ids


# --- stage d ----------------------------------------------------------------


def _extract_skills_for(session: Session, job_ids: Sequence[UUID]) -> int:
    """Store job_skills for jobs that have none, by dictionary lookup.

    **No LLM.** This stage used to make one generative call per job at ~45s
    each; 80 jobs was an hour of wall clock spent reading technology names out
    of a document and writing them back. `app.matching.jd_skills` does the whole
    batch in well under a second against a compiled regex over the `skills`
    vocabulary. See DECISIONS 24.

    Filtered on "has no job_skills rows" rather than on "is new", so a job whose
    extraction was interrupted last run gets another chance while one already
    processed is skipped.

    No ``canonicalize`` step: the index maps surface forms straight to
    ``skills.id``, so a hit is already canonical. That is the other half of the
    speedup — the LLM path had to reconcile free text back onto rows.
    """
    if not job_ids:
        return 0

    already = set(
        session.execute(
            select(JobSkill.job_id).where(JobSkill.job_id.in_(job_ids)).distinct()
        ).scalars()
    )
    pending = [job_id for job_id in job_ids if job_id not in already]
    if not pending:
        return 0

    index = get_index(session)
    processed = 0
    empty = 0

    for job_id in pending:
        job = session.get(Job, job_id)
        if job is None:
            continue

        # Title as well as description: aggregator postings truncate the body
        # (Adzuna caps it at 500 chars) and the title often carries the only
        # technology name in the record.
        text = f"{job.title}\n\n{job.description or ''}"
        hits = extract_skills(text, index)
        if not hits:
            # Left with zero rows on purpose. skill_component reads that as
            # "requirements unparsed" and falls back to semantic scoring, which
            # is the honest answer for a posting naming nothing we recognise.
            empty += 1
            continue

        session.execute(
            pg_insert(JobSkill)
            .values(
                [
                    {
                        "job_id": job_id,
                        "skill_id": hit.skill_id,
                        "requirement": hit.requirement,
                    }
                    for hit in hits
                ]
            )
            # A concurrent run may have inserted the same pair already.
            .on_conflict_do_nothing(index_elements=["job_id", "skill_id"])
        )
        processed += 1

    # One commit for the batch, not one per job. The per-job commit existed
    # because each job cost a minute of model time worth protecting; the whole
    # batch is now sub-second and a partial commit buys nothing.
    session.commit()

    logger.info(
        "Extracted skills for %s job(s); %s matched nothing in the vocabulary",
        processed, empty,
    )
    return processed


# --- stage e ----------------------------------------------------------------


def _chunk_and_embed(
    session: Session, job_ids: Sequence[UUID], embedder: EmbeddingProvider
) -> int:
    """Chunk and embed every job with no ``job_chunks`` rows. Returns rows written."""
    if not job_ids:
        return 0

    already = set(
        session.execute(
            select(JobChunk.job_id).where(JobChunk.job_id.in_(job_ids)).distinct()
        ).scalars()
    )
    pending = [job_id for job_id in job_ids if job_id not in already]
    if not pending:
        return 0

    # Flatten every chunk of every pending job into one list, so batching spans
    # job boundaries. Batching per job would send mostly-empty requests, since
    # the median description yields two or three chunks.
    plan: list[tuple[UUID, int, str]] = []
    for job_id in pending:
        job = session.get(Job, job_id)
        if job is None:
            continue
        for index, text in enumerate(chunk_description(job.description)):
            plan.append((job_id, index, text))

    if not plan:
        logger.info("No chunkable descriptions among %s job(s)", len(pending))
        return 0

    written = 0
    for start in range(0, len(plan), EMBED_BATCH_SIZE):
        batch = plan[start : start + EMBED_BATCH_SIZE]
        vectors = embedder.embed_batch([text for _, _, text in batch])
        rows = [
            {
                "job_id": job_id,
                "chunk_index": index,
                "text": text,
                "embedding": vector,
            }
            for (job_id, index, text), vector in zip(batch, vectors)
        ]
        session.execute(
            pg_insert(JobChunk)
            .values(rows)
            # Idempotent re-chunking: a retried run rewrites nothing.
            .on_conflict_do_nothing(constraint="uq_job_chunks_job_id_index")
        )
        session.commit()
        written += len(rows)

    logger.info("Wrote %s chunk(s) for %s job(s)", written, len(pending))
    return written


# --- stage f ----------------------------------------------------------------


def _load_postings(session: Session, job_ids: Sequence[UUID]) -> list[JobPosting]:
    """Build the scorer's plain inputs with three bulk queries, not 3N.

    N+1 here would be the pipeline's real bottleneck: 200 jobs x 3 relationships
    is 600 round-trips to fetch data that three ``IN`` queries return whole.
    """
    if not job_ids:
        return []

    jobs = {
        job.id: job
        for job in session.execute(select(Job).where(Job.id.in_(job_ids))).scalars()
    }

    skills_by_job: dict[UUID, list[JobSkillRef]] = {}
    for job_id, skill_id, name, requirement in session.execute(
        select(JobSkill.job_id, JobSkill.skill_id, Skill.name, JobSkill.requirement)
        .join(Skill, Skill.id == JobSkill.skill_id)
        # active only: a blocklisted skill still has job_skills rows from
        # before it was pruned, and scoring must not count them.
        .where(JobSkill.job_id.in_(job_ids), Skill.active.is_(True))
    ):
        skills_by_job.setdefault(job_id, []).append(
            JobSkillRef(skill_id=skill_id, name=name, requirement=requirement)
        )

    chunks_by_job: dict[UUID, list[Sequence[float]]] = {}
    for job_id, embedding in session.execute(
        select(JobChunk.job_id, JobChunk.embedding)
        .where(JobChunk.job_id.in_(job_ids))
        .order_by(JobChunk.job_id, JobChunk.chunk_index)
    ):
        chunks_by_job.setdefault(job_id, []).append(list(embedding))

    postings: list[JobPosting] = []
    for job_id in job_ids:
        job = jobs.get(job_id)
        if job is None:
            continue
        postings.append(
            JobPosting(
                job_id=job_id,
                skills=tuple(skills_by_job.get(job_id, ())),
                chunk_embeddings=tuple(chunks_by_job.get(job_id, ())),
                required_years=(
                    float(job.experience_min_years)
                    if job.experience_min_years is not None
                    else None
                ),
            )
        )
    return postings


# --- stage g ----------------------------------------------------------------


def _write_matches(
    session: Session, resume_id: UUID, scored: list[tuple[UUID, ScoreResult]]
) -> int:
    """Bulk-upsert match rows in one statement.

    ``explanation`` is deliberately absent from the update set: stage h writes it
    afterwards, and a re-run that re-scored a job must not blank an explanation
    it is about to write again — or worse, one it will not reach because the job
    dropped out of the top 20 this time.
    """
    if not scored:
        return 0

    rows = [
        {
            "resume_id": resume_id,
            "job_id": job_id,
            # Numeric(5,4) columns: round here rather than letting the driver
            # decide, so the stored value matches what was ranked on.
            "overall_score": round(result.overall_score, 4),
            "semantic_score": round(result.semantic_score, 4),
            "skill_score": round(result.skill_score, 4),
            "skill_recall": round(result.skill_recall, 4),
            "skill_confidence": round(result.skill_confidence, 4),
            "parsed_count": result.parsed_count,
            "skills_unparsed": result.skills_unparsed,
            "tier": result.tier,
            "matching_skills": result.matching_skills,
            "missing_skills": result.missing_skills,
            "model_version": SCORER_VERSION,
        }
        for job_id, result in scored
    ]

    stmt = pg_insert(Match).values(rows)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["resume_id", "job_id"],
            set_={
                "overall_score": stmt.excluded.overall_score,
                "semantic_score": stmt.excluded.semantic_score,
                "skill_score": stmt.excluded.skill_score,
                "skill_recall": stmt.excluded.skill_recall,
                "skill_confidence": stmt.excluded.skill_confidence,
                "parsed_count": stmt.excluded.parsed_count,
                "skills_unparsed": stmt.excluded.skills_unparsed,
                "tier": stmt.excluded.tier,
                "matching_skills": stmt.excluded.matching_skills,
                "missing_skills": stmt.excluded.missing_skills,
                "model_version": stmt.excluded.model_version,
            },
        )
    )
    session.commit()
    return len(rows)


# --- stage h ----------------------------------------------------------------


def _explain_top(
    session: Session,
    resume_id: UUID,
    scored: list[tuple[UUID, ScoreResult]],
    profile: ResumeProfile,
    llm: LLMProvider,
    limit: int = MAX_EXPLANATIONS,
) -> int:
    """Explain the highest-scoring matches, updating those rows in place."""
    top = sorted(scored, key=lambda pair: pair[1].overall_score, reverse=True)[:limit]
    if not top:
        return 0

    jobs = {
        job.id: job
        for job in session.execute(
            select(Job).where(Job.id.in_([job_id for job_id, _ in top]))
        ).scalars()
    }

    written = 0
    for job_id, result in top:
        job = jobs.get(job_id)
        if job is None:
            continue
        explanation = explain_match(
            job.title,
            job.company,
            result,
            llm,
            candidate_years=profile.total_years_experience,
            required_years=(
                float(job.experience_min_years)
                if job.experience_min_years is not None
                else None
            ),
        )
        if explanation is None:
            continue
        session.execute(
            update(Match)
            .where(Match.resume_id == resume_id, Match.job_id == job_id)
            .values(explanation=explanation)
        )
        # Per row, for the same reason as stage d: each of these cost real
        # seconds and must survive a crash on the next one.
        session.commit()
        written += 1

    logger.info("Wrote %s explanation(s)", written)
    return written


# --- orchestration ----------------------------------------------------------


def run_search(
    run_id: UUID,
    resume_id: UUID,
    filters: SearchFilters,
    sources: Sequence[JobSource],
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    session_factory: sessionmaker[Session] = SessionLocal,
) -> SearchOutcome:
    """Run the whole pipeline against an existing ``queued`` run row.

    Every dependency is injected — sources, LLM, embedder, session factory — so
    the pipeline can be tested end to end against fakes with no network and no
    model, which is what the day-3 pipeline test does.

    Any exception marks the run ``failed`` with the error text written to the row
    and is then re-raised. Nothing is swallowed: the background task logs it, and
    the row is the durable record.
    """
    session = session_factory()
    outcome = SearchOutcome(run_id=run_id, status=run_status.STATUS_RUNNING)

    try:
        _set_stage(
            session,
            run_id,
            run_status.STAGE_LOADING_RESUME,
            status=run_status.STATUS_RUNNING,
        )
        profile, resume = _load_resume_profile(session, resume_id)

        _set_stage(session, run_id, run_status.STAGE_BUILDING_QUERIES)
        queries = build_search_queries(
            resume.parsed,
            location=filters.location,
            remote_only=filters.remote_only,
            max_results=filters.max_results,
        )

        _set_stage(session, run_id, run_status.STAGE_FETCHING)
        job_ids, new_ids = _fetch_and_store(
            session, sources, queries, outcome.source_errors
        )
        outcome.jobs_found = len(job_ids)
        outcome.new_jobs = len(new_ids)
        _set_stage(
            session,
            run_id,
            run_status.STAGE_EXTRACTING_SKILLS,
            jobs_found=outcome.jobs_found,
        )

        _extract_skills_for(session, job_ids)

        _set_stage(session, run_id, run_status.STAGE_EMBEDDING)
        outcome.chunks_written = _chunk_and_embed(session, job_ids, embedder)

        _set_stage(session, run_id, run_status.STAGE_SCORING)
        postings = _load_postings(session, job_ids)
        # The LLM-free stage. Everything it needs is already in memory.
        scored = [(posting.job_id, score(profile, posting)) for posting in postings]

        _set_stage(session, run_id, run_status.STAGE_WRITING_MATCHES)
        outcome.matches_written = _write_matches(session, resume_id, scored)

        _set_stage(session, run_id, run_status.STAGE_EXPLAINING)
        outcome.explanations_written = _explain_top(
            session, resume_id, scored, profile, llm
        )

        # Partial, not succeeded, when a source failed but others delivered —
        # the same three-terminal-status rule day 1 established.
        if outcome.source_errors and not job_ids:
            outcome.status = run_status.STATUS_FAILED
        elif outcome.source_errors:
            outcome.status = run_status.STATUS_PARTIAL
        else:
            outcome.status = run_status.STATUS_SUCCEEDED

        outcome.error = (
            "; ".join(f"{name}: {err}" for name, err in outcome.source_errors.items())
            or None
        )

        session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                status=outcome.status,
                stage=run_status.STAGE_DONE,
                jobs_found=outcome.jobs_found,
                error=outcome.error[:4000] if outcome.error else None,
                finished_at=_now(),
            )
        )
        session.commit()

        logger.info(
            "Run %s finished: status=%s jobs=%s matches=%s explanations=%s",
            run_id, outcome.status, outcome.jobs_found,
            outcome.matches_written, outcome.explanations_written,
        )
        return outcome

    except Exception as exc:
        logger.exception("Search run %s failed", run_id)
        _record_failure(session_factory, run_id, exc)
        raise
    finally:
        session.close()


def _record_failure(
    session_factory: sessionmaker[Session], run_id: UUID, exc: BaseException
) -> None:
    """Mark the run failed on a *fresh* session.

    Same reasoning as ingest._record_failure: the session that raised is very
    likely in an aborted transaction where every further statement fails with
    "current transaction is aborted", which would lose the error being saved.
    """
    try:
        with session_factory() as recovery:
            recovery.execute(
                update(IngestionRun)
                .where(IngestionRun.id == run_id)
                .values(
                    status=run_status.STATUS_FAILED,
                    error=repr(exc)[:4000],
                    finished_at=_now(),
                )
            )
            recovery.commit()
    except Exception:  # noqa: BLE001 - last resort; never mask the original error
        logger.exception("Could not record failure for search run %s", run_id)


def clear_matches(session: Session, resume_id: UUID) -> None:
    """Drop a resume's matches. Called when its skills or embedding change.

    Scores computed against the old skill set are not merely stale, they are
    wrong — and there is no way for a reader of the ``matches`` table to tell.
    """
    session.execute(delete(Match).where(Match.resume_id == resume_id))

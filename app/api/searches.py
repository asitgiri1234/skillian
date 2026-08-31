"""Search, run-status, match and job endpoints."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import runs as run_status
from app.api.deps import build_sources, get_session
from app.matching.pipeline import SearchFilters, create_search_run
from app.models import IngestionRun, Job, JobChunk, JobSkill, Match, Resume, Skill
from app.workers import run_search_task

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


# --- request / response models ----------------------------------------------


class SearchRequest(BaseModel):
    resume_id: UUID
    location: str | None = None
    remote_only: bool = False
    sources: list[str] | None = Field(
        default=None, description="Source names; defaults to every registered source"
    )
    max_results: int = Field(default=100, ge=1, le=1000)


class SearchAccepted(BaseModel):
    run_id: UUID
    status: str
    stage: str | None = None


class RunStatus(BaseModel):
    run_id: UUID
    resume_id: UUID | None
    status: str
    stage: str | None
    #: Human-readable stage, so a client does not re-implement the mapping.
    stage_label: str | None
    stage_number: int
    stage_total: int
    jobs_found: int
    sources: list[str]
    error: str | None
    started_at: Any
    finished_at: Any
    #: Terminal across *both* status vocabularies — clients should poll on this
    #: rather than comparing status to a literal. See app/runs.py.
    is_terminal: bool


class MatchOut(BaseModel):
    job_id: UUID
    resume_id: UUID
    overall_score: Decimal | None
    semantic_score: Decimal | None
    skill_score: Decimal | None
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str | None
    model_version: str | None
    # --- job fields, joined so a result list needs no follow-up requests ---
    title: str
    company: str | None
    location: str | None
    is_remote: bool
    salary_raw: str | None
    apply_url: str | None
    posted_date: Any


class ChunkOut(BaseModel):
    chunk_index: int
    text: str


class JobSkillOut(BaseModel):
    skill_id: UUID
    name: str
    requirement: str | None


class JobDetail(BaseModel):
    id: UUID
    source: str
    source_job_id: str
    title: str
    company: str | None
    location: str | None
    is_remote: bool
    description: str | None
    apply_url: str | None
    salary_raw: str | None
    salary_min: Decimal | None
    salary_max: Decimal | None
    salary_currency: str | None
    salary_period: str | None
    experience_raw: str | None
    experience_min_years: Decimal | None
    posted_date: Any
    fetched_at: Any
    skills: list[JobSkillOut]
    chunks: list[ChunkOut]


# --- endpoints --------------------------------------------------------------


@router.post(
    "/searches", response_model=SearchAccepted, status_code=status.HTTP_202_ACCEPTED
)
def create_search(
    body: SearchRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> SearchAccepted:
    """Queue a search and return its run id.

    202, not 201: nothing has been created that the client can fetch as a result
    yet, only a run to poll. The handler does three cheap things — validate the
    resume exists, insert the run row, hand the work to BackgroundTasks — and is
    comfortably inside the 100ms budget because it makes no network call and no
    model call. The pipeline behind it takes minutes.
    """
    resume = session.get(Resume, body.resume_id)
    if resume is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No resume with id {body.resume_id}"
        )
    # Checked here rather than in the background task: a resume that was never
    # parsed cannot produce search queries, and telling the caller now is far
    # better than a run that fails 200ms after they stop watching.
    if not resume.parsed:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Resume has not been parsed yet, so there is nothing to search on.",
        )

    try:
        sources = build_sources(body.sources)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    run = create_search_run(session, body.resume_id, [s.name for s in sources])

    background.add_task(
        run_search_task,
        run_id=run.id,
        resume_id=body.resume_id,
        filters=SearchFilters(
            location=body.location,
            remote_only=body.remote_only,
            max_results=body.max_results,
        ),
        sources=sources,
    )
    return SearchAccepted(run_id=run.id, status=run.status, stage=run.stage)


@router.get("/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: UUID, session: Session = Depends(get_session)) -> RunStatus:
    """Poll a run's progress."""
    run = session.get(IngestionRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run with id {run_id}")

    number, total = run_status.stage_progress(run.stage)
    return RunStatus(
        run_id=run.id,
        resume_id=run.resume_id,
        status=run.status,
        stage=run.stage,
        stage_label=run_status.STAGE_LABELS.get(run.stage or ""),
        stage_number=number,
        stage_total=total,
        jobs_found=run.jobs_found,
        sources=list(run.sources or []),
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        is_terminal=run_status.is_terminal(run.status),
    )


@router.get("/matches", response_model=list[MatchOut])
def list_matches(
    resume_id: UUID,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    min_score: float | None = Query(default=None, ge=0.0, le=1.0),
    session: Session = Depends(get_session),
) -> list[MatchOut]:
    """A resume's matches, best first, joined to the job fields a list needs.

    The join is the point: a result list that returned bare job ids would force
    the client into N follow-up requests to render a single page. Ordering is
    served by ix_matches_resume_id_overall_score, declared on day 1 for exactly
    this query.
    """
    stmt = (
        select(Match, Job)
        .join(Job, Job.id == Match.job_id)
        .where(Match.resume_id == resume_id)
        # NULLS LAST is not needed: overall_score is written on every row the
        # pipeline creates, and a NULL would sort last under DESC anyway.
        .order_by(Match.overall_score.desc())
        .limit(limit)
        .offset(offset)
    )
    if min_score is not None:
        stmt = stmt.where(Match.overall_score >= min_score)

    return [
        MatchOut(
            job_id=match.job_id,
            resume_id=match.resume_id,
            overall_score=match.overall_score,
            semantic_score=match.semantic_score,
            skill_score=match.skill_score,
            matching_skills=list(match.matching_skills or []),
            missing_skills=list(match.missing_skills or []),
            explanation=match.explanation,
            model_version=match.model_version,
            title=job.title,
            company=job.company,
            location=job.location,
            is_remote=job.is_remote,
            salary_raw=job.salary_raw,
            apply_url=job.apply_url,
            posted_date=job.posted_date,
        )
        for match, job in session.execute(stmt)
    ]


@router.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: UUID, session: Session = Depends(get_session)) -> JobDetail:
    """Full detail for one job, including its extracted skills and its chunks."""
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No job with id {job_id}")

    skills = [
        JobSkillOut(skill_id=skill_id, name=name, requirement=requirement)
        for skill_id, name, requirement in session.execute(
            select(JobSkill.skill_id, Skill.name, JobSkill.requirement)
            .join(Skill, Skill.id == JobSkill.skill_id)
            .where(JobSkill.job_id == job_id)
            .order_by(Skill.name)
        )
    ]
    chunks = [
        ChunkOut(chunk_index=index, text=text)
        for index, text in session.execute(
            select(JobChunk.chunk_index, JobChunk.text)
            .where(JobChunk.job_id == job_id)
            .order_by(JobChunk.chunk_index)
        )
    ]
    # Chunk *embeddings* are deliberately not returned: 768 floats per chunk is
    # a large payload that no client can do anything with, and the text is what
    # makes the score explicable.
    return JobDetail(
        id=job.id,
        source=job.source,
        source_job_id=job.source_job_id,
        title=job.title,
        company=job.company,
        location=job.location,
        is_remote=job.is_remote,
        description=job.description,
        apply_url=job.apply_url,
        salary_raw=job.salary_raw,
        salary_min=job.salary_min,
        salary_max=job.salary_max,
        salary_currency=job.salary_currency,
        salary_period=job.salary_period,
        experience_raw=job.experience_raw,
        experience_min_years=job.experience_min_years,
        posted_date=job.posted_date,
        fetched_at=job.fetched_at,
        skills=skills,
        chunks=chunks,
    )

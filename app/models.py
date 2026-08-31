"""SQLAlchemy models for the whole Skillian schema.

All eight tables are defined on day 1 even though only ``jobs`` and
``ingestion_runs`` are written today: the initial Alembic migration then covers
the full shape, and later days add code against existing tables rather than
stacking migrations on a half-built schema.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# nomic-embed-text's output width. Named once: changing the embedding model
# means one edit here plus one migration, not a grep across the schema.
# Must match EmbeddingProvider.dimension — scripts/check_ollama.py asserts it.
EMBEDDING_DIM = 768


def _uuid_pk() -> Mapped[uuid.UUID]:
    """UUID primary key generated in Python, not by the database.

    Avoids depending on pgcrypto / uuid-ossp being installed, and lets application
    code know a row's id before INSERT (useful for building FK graphs in memory).
    """
    return mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    resumes: Mapped[list["Resume"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Structured parse output (sections, dates, titles). The shape is still in
    # flux on day 1, so JSONB rather than premature columns.
    parsed: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # Nullable: a resume exists before it has been embedded.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user: Mapped[User] = relationship(back_populates="resumes")

    # Every resume lookup is "the resumes belonging to this user"; the FK
    # alone does not create an index in Postgres.
    __table_args__ = (Index("ix_resumes_user_id", "user_id"),)


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Postgres array rather than a join table: aliases are a small, read-mostly
    # bag of strings ("js", "ecmascript") that is only ever fetched alongside its
    # skill and never queried independently.
    aliases: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    # False for terms that are not skills — job titles ("Backend Engineer"),
    # abstract qualities ("Scalability"), and sentence fragments the early LLM
    # extractor wrote in ("exp", "Passion for quality"). Flagged rather than
    # deleted: job_skills and resume_skills still reference these rows, and the
    # flag keeps a record of what was pruned and is reversible.
    # Extraction and scoring both filter on active. See DECISIONS 25.
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )

    __table_args__ = (
        # Partial: every read filters `active = true`, and the inactive rows are
        # a small minority no query scans.
        Index("ix_skills_active", "active", postgresql_where=text("active")),
    )


class ResumeSkill(Base):
    """Association: which skills a resume evidences."""

    __tablename__ = "resume_skills"

    resume_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("skills.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # How the link was established: "parsed" | "llm" | "manual". Free text, not an
    # enum, because adding an extractor should not require a migration.
    source: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The composite PK already covers (resume_id, skill_id); this serves the
    # reverse lookup "which resumes have this skill".
    __table_args__ = (Index("ix_resume_skills_skill_id", "skill_id"),)


class JobSkill(Base):
    """Association: which skills a job asks for."""

    __tablename__ = "job_skills"

    job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("skills.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # "required" | "preferred" | "nice_to_have" — free text for the same reason.
    requirement: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Reverse lookup: "which jobs want this skill".
    __table_args__ = (Index("ix_job_skills_skill_id", "skill_id"),)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()

    # --- provenance -------------------------------------------------------
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_job_id: Mapped[str] = mapped_column(Text, nullable=False)
    # sha256 hex of normalised company + title + city. Deliberately NOT unique:
    # the same posting legitimately appears on several boards and we keep every
    # copy (each has its own apply_url) while still being able to group them.
    dedup_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # --- content ----------------------------------------------------------
    title: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_remote: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    apply_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- salary -----------------------------------------------------------
    # salary_raw is always kept: parsing is lossy and best-effort, and the raw
    # string is the only thing we can show a user without risking a wrong number.
    salary_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Numeric, not Float: money must not accumulate binary rounding error.
    # 14,2 comfortably holds annual INR figures.
    salary_min: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    salary_max: Mapped[Decimal | None] = mapped_column(Numeric(14, 2), nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    # "year" | "month" | "day" | "hour" — normalising everything to an annual
    # figure would need assumptions (hours/week) we cannot make safely at ingest.
    salary_period: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # --- experience -------------------------------------------------------
    experience_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_min_years: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 1), nullable=True
    )

    # --- timestamps -------------------------------------------------------
    posted_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # NOTE: there is deliberately no jobs.embedding column. Day 3 replaced the
    # single whole-description vector with per-chunk vectors in job_chunks —
    # see JobChunk below and DECISIONS 15.1.
    chunks: Mapped[list["JobChunk"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobChunk.chunk_index",
    )

    __table_args__ = (
        # The upsert target. (source, source_job_id) is the only identifier a
        # provider guarantees stable, so it — not dedup_hash — defines identity.
        UniqueConstraint("source", "source_job_id", name="uq_jobs_source_source_job_id"),
        Index("ix_jobs_dedup_hash", "dedup_hash"),
    )


class JobChunk(Base):
    """One passage of a job description, with its own embedding.

    Replaces the day-1 ``jobs.embedding`` column. A job description is a
    composite document — responsibilities, requirements, benefits, an equal-
    opportunity boilerplate paragraph — and averaging all of it into one vector
    pulls every posting towards the same bland centroid. Chunking lets the
    strongest *passage* speak for the job (see scorer.semantic_component).
    """

    __tablename__ = "job_chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    # 0-based position in the description. Ordering matters for display: a chunk
    # shown out of sequence reads as a different job.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Non-null: a chunk row exists only because it was embedded. Unlike
    # resumes.embedding there is no "created but not yet embedded" state —
    # chunking and embedding happen in the same pipeline stage.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    job: Mapped[Job] = relationship(back_populates="chunks")

    __table_args__ = (
        # Makes re-chunking idempotent: the pipeline can upsert chunk N without
        # first deleting, and a retry cannot double-insert.
        UniqueConstraint("job_id", "chunk_index", name="uq_job_chunks_job_id_index"),
        # "all chunks for this job" is the only access path scoring ever uses.
        Index("ix_job_chunks_job_id", "job_id"),
    )


class Match(Base):
    """Scored resume x job pair. Written from day 3; defined now."""

    __tablename__ = "matches"

    resume_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # 5,4 = 0.0000..1.0000. Scores are ratios, so fixed precision beats float.
    overall_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    semantic_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    skill_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    # Recall before the evidence discount, and the discount itself. Stored
    # separately from skill_score so a low score is diagnosable after the fact:
    # recall 1.0 with confidence 0.41 is "matched the one requirement we could
    # read", which is a completely different claim from recall 0.2.
    # See scorer.skill_confidence and DECISIONS 30.1.
    skill_recall: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    skill_confidence: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    #: How many requirements were parsed for the job.
    parsed_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: True when the job had no readable requirements at all. These are ranked
    #: in their own bucket by GET /matches, not interleaved — see DECISIONS 30.2.
    skills_unparsed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    matching_skills: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    missing_skills: Mapped[Any | None] = mapped_column(JSONB, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Which scorer produced this row, so results can be invalidated selectively
    # when the model or prompt changes instead of wiping the table.
    model_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # The ranking query: one resume's matches, best score first.
        Index(
            "ix_matches_resume_id_overall_score",
            "resume_id",
            text("overall_score DESC"),
        ),
        # GET /matches runs one query per bucket, each ordered by score.
        Index(
            "ix_matches_resume_unparsed_score",
            "resume_id",
            "skills_unparsed",
            text("overall_score DESC"),
        ),
    )


class IngestionRun(Base):
    """One invocation of the ingest pipeline — the audit trail for a fetch."""

    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Nullable: day-1 ingestion is not tied to a resume, later runs will be.
    # SET NULL on delete so removing a resume does not erase the run history.
    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Day 1 (CLI ingestion): "running" | "success" | "partial" | "failed".
    # Day 3 (search pipeline): "queued" | "running" | "succeeded" | "partial" |
    # "failed". See app/runs.py, which owns both vocabularies and the
    # is_terminal / is_success predicates every reader should use.
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # Which step of the pipeline is executing right now. Purely for progress
    # display while status == "running": GET /runs/{id} is polled and "running"
    # on its own tells a user nothing about whether to keep waiting.
    stage: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    jobs_found: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

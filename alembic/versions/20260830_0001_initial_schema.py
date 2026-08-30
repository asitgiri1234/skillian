"""Initial schema: users, resumes, skills, jobs, matches, ingestion_runs.

Revision ID: 0001
Revises:
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Kept in sync with app.models.EMBEDDING_DIM. Duplicated as a literal on purpose:
# a migration must describe the schema as it was at this revision, so importing
# the constant would let a later model change silently rewrite history.
EMBEDDING_DIM = 1536


def upgrade() -> None:
    # Idempotent, and repeated here (as well as in the Compose init script) so a
    # database provisioned outside Docker still gets the extension.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "skills",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "aliases",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_skills"),
        sa.UniqueConstraint("name", name="uq_skills_name"),
    )

    op.create_table(
        "resumes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("parsed", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_resumes_user_id", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_resumes"),
    )
    # Every resume lookup is "the resumes belonging to this user".
    op.create_index("ix_resumes_user_id", "resumes", ["user_id"])

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("source_job_id", sa.Text(), nullable=False),
        sa.Column("dedup_hash", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("company", sa.Text(), nullable=True),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column(
            "is_remote", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("apply_url", sa.Text(), nullable=True),
        sa.Column("salary_raw", sa.Text(), nullable=True),
        sa.Column("salary_min", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("salary_max", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("salary_currency", sa.String(length=3), nullable=True),
        sa.Column("salary_period", sa.String(length=16), nullable=True),
        sa.Column("experience_raw", sa.Text(), nullable=True),
        sa.Column(
            "experience_min_years", sa.Numeric(precision=4, scale=1), nullable=True
        ),
        sa.Column("posted_date", sa.Date(), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        # Named explicitly: ingest.py targets this constraint by name in its
        # ON CONFLICT clause, so the name is part of the contract.
        sa.UniqueConstraint(
            "source", "source_job_id", name="uq_jobs_source_source_job_id"
        ),
    )
    # Non-unique: several sources can carry the same posting and we keep them all.
    op.create_index("ix_jobs_dedup_hash", "jobs", ["dedup_hash"])

    op.create_table(
        "ingestion_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "sources",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "jobs_found", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["resume_id"],
            ["resumes.id"],
            name="fk_ingestion_runs_resume_id",
            # SET NULL, not CASCADE: deleting a resume must not erase the audit
            # trail of fetches that already happened.
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_runs"),
    )

    op.create_table(
        "resume_skills",
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("skill_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["resume_id"],
            ["resumes.id"],
            name="fk_resume_skills_resume_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            ["skills.id"],
            name="fk_resume_skills_skill_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("resume_id", "skill_id", name="pk_resume_skills"),
    )
    # The composite PK already indexes (resume_id, skill_id); this covers the
    # reverse lookup "which resumes have this skill".
    op.create_index("ix_resume_skills_skill_id", "resume_skills", ["skill_id"])

    op.create_table(
        "job_skills",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("skill_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requirement", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_job_skills_job_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["skill_id"],
            ["skills.id"],
            name="fk_job_skills_skill_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "skill_id", name="pk_job_skills"),
    )
    op.create_index("ix_job_skills_skill_id", "job_skills", ["skill_id"])

    op.create_table(
        "matches",
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("overall_score", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("semantic_score", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("skill_score", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column(
            "matching_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "missing_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("model_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], name="fk_matches_job_id", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["resume_id"],
            ["resumes.id"],
            name="fk_matches_resume_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("resume_id", "job_id", name="pk_matches"),
    )
    # The ranking query: a resume's matches, best first.
    op.create_index(
        "ix_matches_resume_id_overall_score",
        "matches",
        ["resume_id", sa.text("overall_score DESC")],
    )


def downgrade() -> None:
    # Reverse dependency order.
    op.drop_index("ix_matches_resume_id_overall_score", table_name="matches")
    op.drop_table("matches")
    op.drop_index("ix_job_skills_skill_id", table_name="job_skills")
    op.drop_table("job_skills")
    op.drop_index("ix_resume_skills_skill_id", table_name="resume_skills")
    op.drop_table("resume_skills")
    op.drop_table("ingestion_runs")
    op.drop_index("ix_jobs_dedup_hash", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_resumes_user_id", table_name="resumes")
    op.drop_table("resumes")
    op.drop_table("skills")
    op.drop_table("users")
    # The vector extension is intentionally NOT dropped: other schemas in the
    # same database may depend on it.

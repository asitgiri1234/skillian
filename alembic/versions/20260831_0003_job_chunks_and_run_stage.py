"""Chunk-level job embeddings, plus a progress stage on ingestion runs.

Three changes, one migration because they are one design change:

1. ``job_chunks`` — per-passage text + embedding for a job description.
2. ``jobs.embedding`` dropped — superseded by (1). A single vector over a whole
   job description averages requirements together with benefits boilerplate and
   pulls every posting toward the same centroid.
3. ``ingestion_runs.stage`` — which pipeline step is executing, so a polling
   client sees progress instead of a flat "running".

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Hardcoded rather than imported from app.models.EMBEDDING_DIM, for the same
# reason migration 0001 hardcodes it: a migration is a historical record of the
# schema at a point in time. Importing a constant would silently rewrite history
# the next time that constant changes.
EMBEDDING_DIM = 768


def upgrade() -> None:
    op.create_table(
        "job_chunks",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "chunk_index", name="uq_job_chunks_job_id_index"),
    )
    op.create_index("ix_job_chunks_job_id", "job_chunks", ["job_id"])

    # No data migration: an existing jobs.embedding cannot be split into chunk
    # vectors after the fact — the chunk text it would belong to was never
    # stored. Affected jobs are simply re-chunked on the next search run, which
    # the pipeline already does for any job with no job_chunks rows.
    op.drop_column("jobs", "embedding")

    op.add_column("ingestion_runs", sa.Column("stage", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_runs", "stage")
    # Comes back NULL for every row. Symmetric with the note above: the
    # information to reconstruct it does not exist in either direction.
    op.add_column(
        "jobs", sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True)
    )
    op.drop_index("ix_job_chunks_job_id", table_name="job_chunks")
    op.drop_table("job_chunks")

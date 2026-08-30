"""Resize embedding columns from 1536 to 768 for nomic-embed-text.

Day 1 sized the vector columns 1536 for an OpenAI-style embedding model. The
project now uses Ollama with nomic-embed-text, which emits 768 dimensions.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-30
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_DIM = 1536
NEW_DIM = 768


def _clear_embeddings() -> None:
    """Null out existing vectors before changing the column width.

    Two reasons, and the second is the important one:

    1. Postgres cannot cast a vector(1536) to a vector(768) — the ALTER would
       fail outright on any non-null row.
    2. Even if it could, an embedding produced by a *different model* is not
       comparable to one from nomic-embed-text. Distances between them are
       meaningless, so the vectors have to be regenerated regardless. Truncating
       them would silently produce plausible-looking garbage.

    Safe here because nothing generates embeddings yet, so every row is already
    NULL. Written explicitly so the migration is still correct if that changes.
    """
    op.execute(sa.text("UPDATE jobs SET embedding = NULL WHERE embedding IS NOT NULL"))
    op.execute(
        sa.text("UPDATE resumes SET embedding = NULL WHERE embedding IS NOT NULL")
    )


def upgrade() -> None:
    _clear_embeddings()
    op.alter_column(
        "jobs",
        "embedding",
        existing_type=Vector(OLD_DIM),
        type_=Vector(NEW_DIM),
        existing_nullable=True,
    )
    op.alter_column(
        "resumes",
        "embedding",
        existing_type=Vector(OLD_DIM),
        type_=Vector(NEW_DIM),
        existing_nullable=True,
    )


def downgrade() -> None:
    # Symmetric: the same "not comparable across models" argument applies in
    # reverse, so going back also discards the vectors rather than padding them.
    _clear_embeddings()
    op.alter_column(
        "resumes",
        "embedding",
        existing_type=Vector(NEW_DIM),
        type_=Vector(OLD_DIM),
        existing_nullable=True,
    )
    op.alter_column(
        "jobs",
        "embedding",
        existing_type=Vector(NEW_DIM),
        type_=Vector(OLD_DIM),
        existing_nullable=True,
    )

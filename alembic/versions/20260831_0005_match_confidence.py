"""Store the skill component's breakdown on matches.

Calibration over 326 real postings exposed two defects, and both need a column:

1. ``skill_score`` alone is unreadable after the fact. Recall 1.0 at confidence
   0.41 ("matched the single requirement we could read") and recall 0.41 at full
   confidence are different claims that produced the same number.
2. ``skills_unparsed`` was computed and thrown away, so the API could not
   separate unreadable postings from genuinely-poor matches — and they were
   being ranked against each other on numbers that do not mean the same thing.

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("matches", sa.Column("skill_recall", sa.Numeric(5, 4), nullable=True))
    op.add_column(
        "matches", sa.Column("skill_confidence", sa.Numeric(5, 4), nullable=True)
    )
    op.add_column("matches", sa.Column("parsed_count", sa.Integer(), nullable=True))
    op.add_column(
        "matches",
        sa.Column(
            "skills_unparsed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # GET /matches now runs two queries per request, one per bucket, each
    # ordered by score. This serves both.
    op.create_index(
        "ix_matches_resume_unparsed_score",
        "matches",
        ["resume_id", "skills_unparsed", sa.text("overall_score DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_matches_resume_unparsed_score", table_name="matches")
    for column in ("skills_unparsed", "parsed_count", "skill_confidence", "skill_recall"):
        op.drop_column("matches", column)

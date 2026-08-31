"""Add skills.active, so non-skills can be excluded without being deleted.

The `skills` table accumulated job titles ("Backend Engineer", matched in 25 of
80 postings and the single most-matched row in the corpus), abstract nouns
("architecture", "Scalability") and sentence fragments ("exp", "Passion for
quality") from the early LLM extractor. Each inflates the skill component of
every match score that touches it.

Deleting them is the wrong fix: `job_skills` and `resume_skills` reference these
rows, and a delete destroys the record of what was pruned and why. A flag is
reversible, keeps every foreign key valid, and — paired with the checked-in
blocklist in app/data/skill_blocklist.csv — makes the decision auditable.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "skills",
        sa.Column(
            "active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    # Partial index: every read filters `active = true`, and the inactive rows
    # are a small minority that no query scans.
    op.create_index(
        "ix_skills_active",
        "skills",
        ["active"],
        postgresql_where=sa.text("active"),
    )


def downgrade() -> None:
    op.drop_index("ix_skills_active", table_name="skills")
    op.drop_column("skills", "active")

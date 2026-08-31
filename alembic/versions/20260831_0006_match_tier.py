"""Store the tier label on matches.

Computed from the frozen absolute cutoffs in scorer (TIER_STRONG = 0.36,
TIER_MODERATE = 0.25) and stored rather than derived on read, so that changing a
threshold shows up as an explicit re-score rather than silently relabelling
every historical match.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("matches", sa.Column("tier", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("matches", "tier")

"""The skill blocklist: loading it, and refusing to recreate what it excludes.

The terms live in ``app/data/skill_blocklist.csv``, not in this module. That is
deliberate — the blocklist is a set of *judgements about data*, each with a
reason, and it should be reviewable in a diff by someone who does not read
Python. Adding a term is a one-line data change, not a code change.

Two mechanisms, because pruning once is not enough:

* :func:`apply_blocklist` flips ``skills.active`` to false on listed terms.
* :func:`is_disallowed` stops new rows being created for the same class of
  term, so the next extraction run cannot silently reintroduce them.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import Skill
from app.normalize import normalize_text

logger = logging.getLogger(__name__)

BLOCKLIST_PATH = Path(__file__).resolve().parent.parent / "data" / "skill_blocklist.csv"

#: A trailing job-title word. "Backend Engineer" was the most-matched row in the
#: corpus; these are the shapes that produce that class of mistake. Matched only
#: as the *final* word, so "Site Reliability Engineering" is caught but
#: "Engineering Productivity Tooling" is not.
_JOB_TITLE_TAIL = re.compile(
    r"\b(engineer|engineers|engineering|developer|developers|architect|"
    r"architects|manager|managers|analyst|analysts|specialist|specialists|"
    r"consultant|consultants|lead|intern)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BlockedTerm:
    term: str
    category: str
    reason: str


@lru_cache(maxsize=1)
def load_blocklist() -> tuple[BlockedTerm, ...]:
    """Read the checked-in CSV. Cached; call ``load_blocklist.cache_clear()``
    after editing it in a long-lived process."""
    if not BLOCKLIST_PATH.exists():
        logger.warning("No skill blocklist at %s", BLOCKLIST_PATH)
        return ()
    with BLOCKLIST_PATH.open(encoding="utf-8", newline="") as handle:
        return tuple(
            BlockedTerm(
                term=row["term"].strip(),
                category=row["category"].strip(),
                reason=row["reason"].strip(),
            )
            for row in csv.DictReader(handle)
            if row.get("term", "").strip()
        )


@lru_cache(maxsize=1)
def blocked_keys() -> frozenset[str]:
    """Normalised forms of every blocklisted term, for membership tests."""
    return frozenset(normalize_text(entry.term) for entry in load_blocklist())


def is_disallowed(name: str) -> str | None:
    """Return a reason if ``name`` must not become a skill row, else None.

    Applied at the point of *creation*, so a blocklisted term that reappears in
    a future posting is refused rather than silently re-added under a new id —
    which would defeat the whole exercise, since the `active` flag lives on the
    row rather than on the string.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return "empty"

    key = normalize_text(cleaned)
    if not key:
        return "normalises to nothing"
    if key in blocked_keys():
        return "on the checked-in blocklist"
    if _JOB_TITLE_TAIL.search(cleaned):
        return "reads as a job title, not a skill"
    return None


def apply_blocklist(session: Session) -> tuple[int, list[str]]:
    """Set ``active = false`` on every blocklisted term present in the table.

    Returns ``(rows_deactivated, terms_not_found)``. Terms absent from the table
    are reported rather than ignored: a blocklist entry that matches nothing is
    either a typo or a term whose row was never created, and both are worth
    seeing.
    """
    entries = load_blocklist()
    if not entries:
        return 0, []

    by_key = {normalize_text(entry.term): entry for entry in entries}
    present: dict[str, list] = {}
    for skill_id, name in session.execute(select(Skill.id, Skill.name)):
        key = normalize_text(name)
        if key in by_key:
            present.setdefault(key, []).append(skill_id)

    ids = [skill_id for group in present.values() for skill_id in group]
    if ids:
        session.execute(
            update(Skill).where(Skill.id.in_(ids)).values(active=False)
        )
        session.commit()

    missing = sorted(
        by_key[key].term for key in by_key if key not in present
    )
    logger.info(
        "Blocklist applied: %s row(s) deactivated, %s term(s) not present",
        len(ids), len(missing),
    )
    return len(ids), missing

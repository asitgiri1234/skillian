"""The ``ingestion_runs.status`` and ``ingestion_runs.stage`` vocabularies.

Two vocabularies share one column, and that is worth stating plainly rather than
discovering by grep:

* Day 1's CLI ingestion writes ``running`` -> ``success`` | ``partial`` |
  ``failed``.
* Day 3's search pipeline was specified as ``queued`` -> ``running`` ->
  ``succeeded``, so it writes ``succeeded`` where the CLI writes ``success``.

Renaming either one would either break day 1's verified behaviour or diverge
from the day-3 specification, so instead both are declared here and every
*reader* goes through :func:`is_terminal` / :func:`is_success` rather than
comparing to a string literal. That keeps the wart in one file.
See DECISIONS 18.4.
"""

from __future__ import annotations

from typing import Final

# --- shared -----------------------------------------------------------------
STATUS_RUNNING: Final = "running"
STATUS_PARTIAL: Final = "partial"
STATUS_FAILED: Final = "failed"

# --- day 1, CLI ingestion ---------------------------------------------------
STATUS_SUCCESS: Final = "success"

# --- day 3, search pipeline -------------------------------------------------
STATUS_QUEUED: Final = "queued"
STATUS_SUCCEEDED: Final = "succeeded"

#: Every status that means "this run is over, stop polling".
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {STATUS_SUCCESS, STATUS_SUCCEEDED, STATUS_PARTIAL, STATUS_FAILED}
)

#: Terminal *and* nothing went wrong.
SUCCESS_STATUSES: Final[frozenset[str]] = frozenset({STATUS_SUCCESS, STATUS_SUCCEEDED})


def is_terminal(status: str | None) -> bool:
    """True when the run has finished, whichever vocabulary wrote it."""
    return status in TERMINAL_STATUSES


def is_success(status: str | None) -> bool:
    """True for a clean finish. ``partial`` is terminal but not a success."""
    return status in SUCCESS_STATUSES


# --- stages -----------------------------------------------------------------
# Written to ingestion_runs.stage as the pipeline advances. Ordered, so a UI can
# render "step 4 of 8" from STAGE_ORDER.index(stage) without hardcoding names.
STAGE_QUEUED: Final = "queued"
STAGE_LOADING_RESUME: Final = "loading_resume"
STAGE_BUILDING_QUERIES: Final = "building_queries"
STAGE_FETCHING: Final = "fetching"
STAGE_EXTRACTING_SKILLS: Final = "extracting_skills"
STAGE_EMBEDDING: Final = "embedding"
STAGE_SCORING: Final = "scoring"
STAGE_WRITING_MATCHES: Final = "writing_matches"
STAGE_EXPLAINING: Final = "explaining"
STAGE_DONE: Final = "done"

STAGE_ORDER: Final[tuple[str, ...]] = (
    STAGE_QUEUED,
    STAGE_LOADING_RESUME,
    STAGE_BUILDING_QUERIES,
    STAGE_FETCHING,
    STAGE_EXTRACTING_SKILLS,
    STAGE_EMBEDDING,
    STAGE_SCORING,
    STAGE_WRITING_MATCHES,
    STAGE_EXPLAINING,
    STAGE_DONE,
)

#: Human-readable labels, so the API can hand a UI something printable without
#: every client re-deriving "extracting_skills" -> "Extracting skills".
STAGE_LABELS: Final[dict[str, str]] = {
    STAGE_QUEUED: "Queued",
    STAGE_LOADING_RESUME: "Loading resume",
    STAGE_BUILDING_QUERIES: "Building search queries",
    STAGE_FETCHING: "Fetching jobs",
    STAGE_EXTRACTING_SKILLS: "Extracting job requirements",
    STAGE_EMBEDDING: "Embedding job descriptions",
    STAGE_SCORING: "Scoring matches",
    STAGE_WRITING_MATCHES: "Saving matches",
    STAGE_EXPLAINING: "Writing explanations",
    STAGE_DONE: "Done",
}


def stage_progress(stage: str | None) -> tuple[int, int]:
    """``(step, total)`` for ``stage``, 1-based. ``(0, N)`` for an unknown stage."""
    total = len(STAGE_ORDER)
    if stage in STAGE_ORDER:
        return STAGE_ORDER.index(stage) + 1, total
    return 0, total

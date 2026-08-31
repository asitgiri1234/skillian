"""Turn a parsed resume into the search queries a run will actually fetch.

Referenced by the day-3 brief as an existing module; it was not — this is it.

Deliberately **LLM-free**. Query generation runs once per search, so a model call
here would be affordable, but it would also be a 30-90 second local-model wait
before the first HTTP request goes out, it would be non-deterministic across
runs of the same resume, and job-board keyword search is a bag-of-words matcher
that gains nothing from a fluent phrase. Two job titles and a handful of the
candidate's strongest skills is what the board can actually use.
See DECISIONS 18.1.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.normalize import normalize_text
from app.sources.base import SearchQuery

logger = logging.getLogger(__name__)

#: Distinct queries per run. Each one costs a full paginated fetch against every
#: enabled source, so this is a quota decision as much as a relevance one.
MAX_QUERIES = 3

#: Skills appended to the skill-based query. Beyond a handful, a board's keyword
#: matcher starts returning the union of everything rather than the intersection.
SKILLS_PER_QUERY = 4

# Seniority words are dropped from a title before it becomes a query: boards
# match them literally, so "Senior Python Engineer" misses every posting titled
# "Python Engineer" while the reverse returns both.
_SENIORITY_RE = re.compile(
    r"\b(senior|sr|junior|jr|lead|principal|staff|chief|head|associate|"
    r"entry.level|mid.level|intern|trainee|i{1,3}|iv|v|\d)\b",
    re.IGNORECASE,
)

# Fragments that survive a title strip but carry no search signal.
_TITLE_NOISE_RE = re.compile(r"\b(at|for|of|the|and|with|in|to)\b", re.IGNORECASE)

# Skills too generic to narrow a job search. Every posting mentions them, so
# including one costs a query slot and returns the whole board.
_GENERIC_SKILLS: frozenset[str] = frozenset(
    {
        "communication", "teamwork", "leadership", "problem solving",
        "problem-solving", "time management", "agile", "scrum", "git",
        "english", "microsoft office", "excel", "ms office", "documentation",
        "collaboration", "analytical skills", "teamplayer", "team player",
    }
)


def _clean_title(title: str) -> str:
    """Strip seniority and filler from a job title, leaving the role itself."""
    cleaned = _SENIORITY_RE.sub(" ", title)
    cleaned = _TITLE_NOISE_RE.sub(" ", cleaned)
    # Drop anything after a separator: "Backend Engineer - Remote (Bengaluru)"
    # is one role plus two facets the SearchQuery already carries as fields.
    cleaned = re.split(r"[-–—|(/,]", cleaned)[0]
    return " ".join(cleaned.split()).strip()


def _titles(parsed: dict[str, Any]) -> list[str]:
    """Distinct cleaned role titles, most recent first."""
    seen: set[str] = set()
    titles: list[str] = []
    for entry in parsed.get("experience") or []:
        if not isinstance(entry, dict):
            continue
        # "role" since the schema trim; ExperienceRef replaced ExperienceEntry
        # and renamed this field. See DECISIONS 20.
        raw = (entry.get("role") or "").strip()
        if not raw:
            continue
        cleaned = _clean_title(raw)
        # A one-word remnant ("Engineer") is too broad to be worth a query slot.
        if len(cleaned.split()) < 2:
            continue
        key = normalize_text(cleaned)
        if key and key not in seen:
            seen.add(key)
            titles.append(cleaned)
    return titles


def _skills(parsed: dict[str, Any]) -> list[str]:
    """Specific skills, in resume order.

    Resume order is a real signal: candidates list what they lead with first,
    and ParsedResume preserves the document's ordering.
    """
    result: list[str] = []
    seen: set[str] = set()
    for value in parsed.get("skills") or []:
        skill = str(value).strip()
        key = normalize_text(skill)
        if not key or key in _GENERIC_SKILLS or key in seen:
            continue
        seen.add(key)
        result.append(skill)
    return result


def build_search_queries(
    parsed: dict[str, Any] | None,
    *,
    location: str | None = None,
    remote_only: bool = False,
    max_results: int = 100,
    max_queries: int = MAX_QUERIES,
) -> list[SearchQuery]:
    """Build the queries for one search run.

    Produces, in order of expected precision:

    1. The candidate's most recent role title.
    2. Their top skills as one keyword bag.
    3. Their previous distinct role title, if they have one.

    ``max_results`` is divided across the queries so that raising the query count
    does not multiply the number of rows a single run fetches — the cap is on the
    run, not on each query.

    Raises ValueError when the resume yields no usable keywords at all, which
    means the parse was empty and running the search would fetch noise.
    """
    parsed = parsed or {}
    titles = _titles(parsed)
    skills = _skills(parsed)

    keyword_sets: list[str] = []
    if titles:
        keyword_sets.append(titles[0])
    if skills:
        keyword_sets.append(" ".join(skills[:SKILLS_PER_QUERY]))
    if len(titles) > 1:
        keyword_sets.append(titles[1])
    # Last resort for a resume with no parseable titles and few skills: a broad
    # query beats no search at all.
    if not keyword_sets and skills:
        keyword_sets.append(skills[0])

    keyword_sets = keyword_sets[:max_queries]
    if not keyword_sets:
        raise ValueError(
            "Resume has no usable job titles or skills to search on. "
            "Re-parse it, or add skills via PATCH /resumes/{id}/skills."
        )

    # Integer division with a floor of 1: three queries over max_results=100
    # fetch up to 33 each, not 100 each.
    per_query = max(1, max_results // len(keyword_sets))

    queries = [
        SearchQuery(
            keywords=keywords,
            location=location,
            remote_only=remote_only,
            max_results=per_query,
        )
        for keywords in keyword_sets
    ]
    logger.info("Built %s search queries: %s", len(queries), keyword_sets)
    return queries

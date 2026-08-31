"""Extract a job's skill requirements, and canonicalise skill names to rows.

Two jobs that both want "React" must resolve to the *same* ``skills.id``, or the
skill component of the score is meaningless — set intersection over strings that
differ by case, punctuation or an alias ("JS" vs "JavaScript") silently returns
nothing and every candidate looks unqualified. So every skill name, from a job
or a resume, goes through :class:`SkillCanonicalizer`.

This module is not in the day-3 file list. It exists because step (d) of the
pipeline — "extract + canonicalize skills into job_skills" — is two distinct
concerns (a model call, and a database identity mapping) and neither belongs in
``pipeline.py`` next to the orchestration. See DECISIONS 18.2.
"""

from __future__ import annotations

import logging
import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import Skill
from app.normalize import normalize_text
from app.providers import LLMProvider

logger = logging.getLogger(__name__)

# A job description longer than this is truncated before extraction. Requirements
# are near-universally in the first half of a posting; the tail is benefits and
# legal boilerplate, which contains no skills and costs tokens to read.
MAX_DESCRIPTION_CHARS = 12_000

MAX_SKILLS_PER_JOB = 40

# A canonical skill name is short. "Amazon Web Services" is three words;
# "Experience running services in production, including on-call" is a
# requirement sentence, and storing it as a skill is what breaks matching.
MAX_SKILL_NAME_CHARS = 32
MAX_SKILL_NAME_WORDS = 4

# Junk the extractor emits when a description has no real requirements.
_SKILL_PLACEHOLDERS: frozenset[str] = frozenset(
    {"string", "n/a", "na", "none", "null", "unknown", "skill", "skills", "-", ""}
)

# --- name cleaning ----------------------------------------------------------
#
# The single most important code in this module, and the thing verification
# caught. Asked for a job's skills, qwen2.5 reliably answers in the posting's own
# words — "Strong Python", "Comfortable with Docker", "FastAPI or Django in
# production" — because that is what the document says. Stored verbatim, none of
# those canonicalise onto the resume's "Python" / "Docker" / "FastAPI", so a
# perfectly qualified candidate scores near zero on skill overlap and the whole
# ranking inverts. The prompt asks for bare names; this is the belt to that
# braces, because the prompt alone is not reliable.

# Qualifier prefixes, stripped repeatedly: "strong hands-on Python" -> "Python".
_PREFIX_RE = re.compile(
    r"^(?:"
    r"strong|solid|deep|excellent|good|great|advanced|expert(?:ise)?(?:\s+in|\s+with)?|"
    r"proficien(?:t|cy)(?:\s+in|\s+with)?|profound|significant|substantial|"
    r"experience(?:d)?(?:\s+in|\s+with|\s+using)?|hands[\s-]?on(?:\s+experience)?"
    r"(?:\s+in|\s+with)?|familiar(?:ity)?(?:\s+with)?|comfortable(?:\s+with)?|"
    r"knowledge(?:\s+of)?|working\s+knowledge(?:\s+of)?|understanding(?:\s+of)?|"
    r"background(?:\s+in)?|exposure(?:\s+to)?|ability\s+to|able\s+to|"
    r"demonstrated|proven|track\s+record(?:\s+of)?|some|basic|"
    r"preferably|ideally|especially|particularly|plus\s+points?\s+for|"
    r"a\s+strong|an?"
    r")\b[\s:,-]*",
    re.IGNORECASE,
)

# Qualifier suffixes: "PostgreSQL knowledge" -> "PostgreSQL".
_SUFFIX_RE = re.compile(
    r"[\s,-]*\b(?:"
    r"experience|expertise|knowledge|skills?|proficiency|fundamentals|"
    r"in\s+production|at\s+scale|in\s+a\s+production\s+environment|"
    r"is\s+a\s+plus|a\s+plus|preferred|required"
    r")\b\.?$",
    re.IGNORECASE,
)

# A duration anywhere means this is a requirement, not a skill name.
_DURATION_RE = re.compile(r"\b\d+\s*\+?\s*(?:years?|yrs?|months?)\b", re.IGNORECASE)

# Lists the posting writes as one requirement: "FastAPI or Django", "Go/Rust",
# "SQL and relational modelling", "Kafka, Redis, Celery". Each side is a real
# skill a candidate might hold, so these are split rather than dropped.
# Splitting on "and" occasionally cuts a two-word name in half; that costs one
# alias, whereas not splitting loses every skill after the first.
#
# Two phases, because "/" needs an escape hatch that the others do not: the
# word separators run first, then "/" runs on each piece unless that piece is a
# known compound. One pass would see "CI/CD pipelines, ideally GitHub Actions"
# as a whole and never recognise the CI/CD inside it.
_LIST_RE = re.compile(r"\s+(?:or|and)\s+|\s*[,;]\s*", re.IGNORECASE)
_SLASH_RE = re.compile(r"\s*/\s*")

# "Deep PostgreSQL knowledge: schema design, query tuning" — the head before the
# colon names the skill; everything after it elaborates.
_ELABORATION_RE = re.compile(r"\s*[:—–]\s*")

#: After a duration is stripped, a remainder this long means the requirement was
#: about seniority, not a named skill: "5+ years of professional backend
#: development" is not the skill "professional backend development". "3+ years
#: Python" is still salvageable.
_MAX_WORDS_AFTER_DURATION = 2

# A discipline rather than a skill. Only applied to what is left after a
# duration is stripped — "Data Engineering" as a standalone extracted skill is
# odd but harmless, whereas "3+ years in data engineering" is unambiguously a
# seniority requirement.
_DISCIPLINE_RE = re.compile(
    r"\b(?:engineering|development|design|operations|management|science)\b$",
    re.IGNORECASE,
)

# Compounds that must survive the "/" split. Without this, "CI/CD" becomes the
# two useless skills "CI" and "CD". Kept as an explicit short list rather than a
# rule, because "Go/Rust" genuinely is two skills and no heuristic separates the
# two cases.
_SLASH_COMPOUNDS: tuple[str, ...] = ("ci/cd", "a/b", "tcp/ip", "ui/ux", "i/o")
_COMPOUND_RE = re.compile(
    "|".join(re.escape(compound) for compound in _SLASH_COMPOUNDS), re.IGNORECASE
)

# Phrases that survive every filter but name a discipline, not a skill. Storing
# them costs a skills row and matches nothing useful.
_GENERIC_NON_SKILLS: frozenset[str] = frozenset(
    {
        "backend development", "frontend development", "software engineering",
        "software development", "web development", "programming", "coding",
        "computer science", "development", "engineering", "technology",
        "communication", "teamwork", "collaboration",
    }
)

# "Monitoring with Prometheus" -> "Prometheus". A gerund plus a preposition is
# the posting describing an activity; the skill is what follows.
_ACTIVITY_RE = re.compile(r"^\w+ing\s+(?:with|in|on|using|via)\s+", re.IGNORECASE)

# Sentence markers. If one survives cleaning, the model returned prose.
_PROSE_RE = re.compile(
    r"\b(?:you|your|we|our|the\s+role|this\s+role|including|such\s+as|"
    r"etc|e\.g|i\.e|who|which|that\s+you|will\s+be|must\s+be|should)\b",
    re.IGNORECASE,
)


def clean_skill_name(raw: str) -> list[str]:
    """Reduce a model-reported requirement to zero or more canonical skill names.

    Returns a list because a posting's "FastAPI or Django" names two skills.
    Returns ``[]`` when nothing survives — a requirement like "5+ years of
    professional backend development" describes seniority, not a skill, and
    ``jobs.experience_min_years`` is where that belongs.
    """
    candidate = raw.strip().strip(".;:,-").strip()
    # Checked before any splitting: "n/a" would otherwise become "n" and "a".
    # Both forms, because normalize_text turns "/" into a space (it treats a
    # slash as a word joiner), so "n/a" never equals the "n/a" placeholder.
    if not candidate or {
        candidate.casefold(),
        normalize_text(candidate),
    } & _SKILL_PLACEHOLDERS:
        return []

    # Keep only the head of an elaborated requirement.
    candidate = _ELABORATION_RE.split(candidate)[0].strip()
    if not candidate:
        return []

    # A duration is never part of a skill name, and its presence usually means
    # the whole string is a seniority requirement rather than a skill.
    if _DURATION_RE.search(candidate):
        remainder = _DURATION_RE.sub(" ", candidate)
        remainder = re.sub(
            r"\b(?:of|in|with|professional|commercial|industry)\b",
            " ",
            remainder,
            flags=re.IGNORECASE,
        )
        remainder = " ".join(remainder.split()).strip(".;:,- ")
        # "3+ years Python" -> "Python", kept. "5+ years of professional backend
        # development" -> three words, which is a seniority bar, not a skill.
        if not remainder or len(remainder.split()) > _MAX_WORDS_AFTER_DURATION:
            return []
        # "3+ years in data engineering" leaves two words, but naming a
        # discipline after a duration is still a seniority bar.
        if _DISCIPLINE_RE.search(remainder):
            return []
        candidate = remainder

    parts: list[str] = []
    for piece in _LIST_RE.split(candidate):
        piece = piece.strip()
        if not piece:
            continue
        # Hide the slashes inside known compounds so the split below cannot see
        # them, then restore. Matching the compound as a substring rather than
        # the whole piece is what makes "CI/CD pipelines" work.
        protected = _COMPOUND_RE.sub(lambda m: m.group(0).replace("/", "\x00"), piece)
        parts.extend(part.replace("\x00", "/") for part in _SLASH_RE.split(protected))

    results: list[str] = []
    for part in parts:
        name = _ACTIVITY_RE.sub("", part.strip()).strip()
        # Strip prefixes repeatedly: qualifiers stack ("strong hands-on ...").
        for _ in range(4):
            stripped = _PREFIX_RE.sub("", name).strip()
            if stripped == name:
                break
            name = stripped
        for _ in range(3):
            stripped = _SUFFIX_RE.sub("", name).strip()
            if stripped == name:
                break
            name = stripped

        name = name.strip().strip(".;:,-").strip()
        name = re.sub(r"\s+", " ", name)

        if not name or len(name) > MAX_SKILL_NAME_CHARS:
            continue
        if len(name.split()) > MAX_SKILL_NAME_WORDS:
            continue
        if _PROSE_RE.search(name):
            continue
        normalised = normalize_text(name)
        if normalised in _SKILL_PLACEHOLDERS or normalised in _GENERIC_NON_SKILLS:
            continue
        # A bare verb phrase ("running services", "operating what you built")
        # survives the filters above but is not a skill.
        if name.split()[0].casefold().endswith("ing") and len(name.split()) > 1:
            continue
        results.append(name)

    return results


class ExtractedSkill(BaseModel):
    """One requirement, as the model reports it."""

    name: str | None = Field(
        description=(
            "The bare canonical name of one skill, tool or technology — "
            "'Python', 'PostgreSQL', 'Kubernetes'. Never a phrase, a sentence, "
            "or a duration. Write 'Python', not 'Strong Python' or "
            "'5+ years of Python experience'."
        )
    )
    # Literal, not str: the JSON schema renders this as an enum, so constrained
    # decoding makes any third value unrepresentable. Classification is the whole
    # reason this is an LLM call rather than a keyword scan, so it must not be
    # left to the model's discretion to answer in its own vocabulary.
    requirement: Literal["required", "preferred"] = Field(
        description=(
            "'required' if the posting presents this as a must-have, "
            "'preferred' if it is a nice-to-have, bonus or plus"
        )
    )


class ExtractedSkills(BaseModel):
    """The extraction target for a job description."""

    skills: list[ExtractedSkill] = Field(
        description="Every distinct technical or professional skill the posting asks for"
    )

    @field_validator("skills", mode="after")
    @classmethod
    def _clean(cls, values: list[ExtractedSkill]) -> list[ExtractedSkill]:
        """Reduce each entry to canonical names, then de-duplicate.

        One reported requirement can become zero names ("5+ years of backend
        development") or two ("FastAPI or Django"), which is why this rebuilds
        the list rather than filtering it.

        When the same skill appears twice at different levels the stronger one
        wins: a posting that says "React required" in one place and "React a
        plus" in another is asking for React.
        """
        best: dict[str, ExtractedSkill] = {}
        order: list[str] = []

        for value in values:
            if not value.name:
                continue
            for name in clean_skill_name(value.name):
                key = normalize_text(name)
                if not key or key in _SKILL_PLACEHOLDERS:
                    continue
                cleaned = ExtractedSkill(name=name, requirement=value.requirement)
                existing = best.get(key)
                if existing is None:
                    best[key] = cleaned
                    order.append(key)
                elif (
                    existing.requirement == "preferred"
                    and cleaned.requirement == "required"
                ):
                    best[key] = cleaned

        return [best[key] for key in order][:MAX_SKILLS_PER_JOB]


_SYSTEM_PROMPT = (
    "You read job postings and name the skills they ask for. Give the bare, "
    "canonical name of each skill, tool, technology or named methodology — the "
    "name it would have in a dropdown list, not the words the posting used "
    "around it. Strip every qualifier, every duration and every verb.\n"
    "Write 'Python', not 'Strong Python' or '5+ years of Python'.\n"
    "Write 'Docker', not 'Comfortable with Docker'.\n"
    "Write 'PostgreSQL', not 'Deep PostgreSQL knowledge'.\n"
    "List 'FastAPI' and 'Django' separately, not 'FastAPI or Django'.\n"
    "Skip anything that is not a nameable skill: years of experience, job "
    "titles, company names, degrees, and soft qualities like 'team player'. "
    "Never list a skill the posting does not mention."
)


def _build_prompt(title: str, description: str) -> str:
    return "\n".join(
        [
            f"Job title: {title}",
            "",
            "Name the skills this posting asks for, one entry per skill, and for "
            "each one say whether the posting treats it as required or merely "
            'preferred. Language like "must have", "required", "you have" means '
            'required; "nice to have", "bonus", "a plus", "preferred", '
            '"ideally" means preferred.',
            "",
            "--- JOB DESCRIPTION ---",
            description,
            "--- END JOB DESCRIPTION ---",
        ]
    )


def extract_job_skills(
    title: str, description: str | None, llm: LLMProvider
) -> list[ExtractedSkill]:
    """Pull classified skill requirements out of one job description.

    No retry loop, unlike resume extraction. An empty result here is a legitimate
    answer — plenty of postings genuinely list no concrete requirements — and the
    scorer already handles that case explicitly by returning None from
    ``skill_component``. Retrying would spend a second local-model minute per job
    to argue with a model that was right the first time. See DECISIONS 18.3.

    Returns ``[]`` rather than raising when the description is empty or the model
    reply does not validate: one unreadable posting must not sink a search over
    two hundred.
    """
    if not description or not description.strip():
        return []

    text = description.strip()[:MAX_DESCRIPTION_CHARS]

    raw = llm.complete(
        _build_prompt(title, text), ExtractedSkills, system=_SYSTEM_PROMPT
    )
    try:
        return ExtractedSkills.model_validate(raw).skills
    except Exception:  # noqa: BLE001 - one bad posting must not abort the run
        logger.warning("Skill extraction returned an unusable object for %r", title)
        return []


class SkillCanonicalizer:
    """Maps free-text skill names onto stable ``skills`` rows.

    Loads the whole table once and keeps it in memory. That is the right call at
    this size — the skills table is a controlled vocabulary numbering in the low
    thousands even for a mature deployment, and the alternative is a round-trip
    per skill per job, which for 200 jobs at ~15 skills each is 3,000 queries.

    One instance per pipeline run; it is not thread-safe and is not meant to be
    long-lived, since it will not see rows another process inserts.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        # normalised name or alias -> skills.id
        self._index: dict[str, UUID] = {}
        self._load()

    def _load(self) -> None:
        for skill in self._session.execute(select(Skill)).scalars():
            self._index.setdefault(normalize_text(skill.name), skill.id)
            for alias in skill.aliases or []:
                # setdefault, not assignment: a canonical name always outranks
                # someone else's alias if the two ever collide.
                self._index.setdefault(normalize_text(alias), skill.id)
        logger.debug("Loaded %s skill keys", len(self._index))

    def canonicalize(self, name: str) -> UUID | None:
        """Return the ``skills.id`` for ``name``, creating the row if new.

        Returns None for a name that normalises to nothing (punctuation only).
        """
        key = normalize_text(name)
        if not key or key in _SKILL_PLACEHOLDERS:
            return None

        existing = self._index.get(key)
        if existing is not None:
            return existing

        # ON CONFLICT DO NOTHING + a follow-up SELECT rather than a plain INSERT:
        # two search runs can extract the same new skill concurrently, and losing
        # that race must yield the winner's id, not a unique-violation that
        # aborts the transaction.
        display_name = name.strip()
        stmt = (
            pg_insert(Skill)
            .values(name=display_name, aliases=[])
            .on_conflict_do_nothing(index_elements=[Skill.name])
            .returning(Skill.id)
        )
        skill_id = self._session.execute(stmt).scalar_one_or_none()
        if skill_id is None:
            skill_id = self._session.execute(
                select(Skill.id).where(Skill.name == display_name)
            ).scalar_one()

        self._index[key] = skill_id
        return skill_id

    def canonicalize_all(self, names: list[str]) -> list[UUID]:
        """Canonicalise several names, dropping unusable ones and duplicates.

        Runs :func:`clean_skill_name` first, unlike :meth:`canonicalize`. This is
        the path resume skills take, and both sides of the match have to be
        cleaned the same way or the intersection silently fails: a resume saying
        "Strong Python" and a job saying "Python" must land on one row.
        """
        seen: set[UUID] = set()
        result: list[UUID] = []
        for raw in names:
            for name in clean_skill_name(raw):
                skill_id = self.canonicalize(name)
                if skill_id is not None and skill_id not in seen:
                    seen.add(skill_id)
                    result.append(skill_id)
        return result

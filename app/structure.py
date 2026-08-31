"""Resume text -> structured :class:`ParsedResume`.

The extraction contract has two independent halves, and conflating them is the
usual way this goes wrong:

* **Shape** is guaranteed by the provider. ``ParsedResume.model_json_schema()``
  is passed to the LLM as ``format=``, so decoding is constrained and malformed
  JSON is unrepresentable. The prompt never asks for JSON.
* **Accuracy is not guaranteed by anything.** A constrained decoder will happily
  emit a perfectly-shaped object full of nonsense — most memorably the literal
  string ``"string"``, copied straight out of the schema. That is what the
  validate-and-retry loop below is for, and why the validators here check
  meaning rather than types.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.providers import LLMProvider, get_parse_provider

logger = logging.getLogger(__name__)

# qwen2.5 has a 32k context, but a long resume plus the schema plus retry
# feedback can still crowd it. Truncating the input is better than a silent
# server-side truncation that would drop the education section without saying so.
MAX_RESUME_CHARS = 24_000

# Values a model emits when it has nothing real to say. The first two come
# directly from the JSON schema's own vocabulary, which constrained decoding
# makes *more* likely, not less.
_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "string",
        "str",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "not specified",
        "not provided",
        "not mentioned",
        "example",
        "john doe",
        "jane doe",
        "your name",
        "-",
        "",
    }
)


def _is_placeholder(value: str | None) -> bool:
    return value is None or value.strip().casefold() in _PLACEHOLDERS


def _clean_optional(value: str | None) -> str | None:
    """Collapse placeholder junk to None so downstream code has one empty case."""
    if _is_placeholder(value):
        return None
    assert value is not None  # _is_placeholder covers None
    return value.strip()


def _clean_required(value: str | None) -> str:
    """Placeholder-strip a non-nullable field down to "" rather than None.

    The Ref models below use bare ``str`` where the old ones used ``str | None``,
    because a nullable field lets the grammar emit ``null`` and a null costs
    tokens for no information. "" is the empty case here, and the list
    validators on ParsedResume drop entries that are empty in every field.
    """
    return "" if _is_placeholder(value) else (value or "").strip()


class ProjectRef(BaseModel):
    """A project, as a pointer rather than a description.

    ``description`` is deliberately absent — see the note on ParsedResume.
    """

    title: str = Field(description="Project name, as written")
    tech: list[str] = Field(description="Technologies used, one per item")

    @field_validator("title", mode="after")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        return _clean_required(value)

    @field_validator("tech", mode="after")
    @classmethod
    def _clean_tech(cls, values: list[str]) -> list[str]:
        return _dedupe_labels(values)


class ExperienceRef(BaseModel):
    """One role, as three short labels.

    ``summary`` is gone. ``duration`` replaces the old start/end/is_current
    trio: it is one field instead of three, it stays free text because resumes
    write dates a dozen ways, and it is copied verbatim rather than reasoned
    about.
    """

    company: str = Field(description="Employer name")
    role: str = Field(description="Job title held")
    duration: str = Field(description="Dates as written, e.g. '2021 - Present'")

    @field_validator("company", "role", "duration", mode="after")
    @classmethod
    def _clean(cls, value: str) -> str:
        return _clean_required(value)


class EducationRef(BaseModel):
    institution: str = Field(description="School or university name")
    degree: str = Field(description="Degree awarded, e.g. 'B.E. Computer Science'")
    year: str | None = Field(description="Graduation year as written")

    @field_validator("institution", "degree", mode="after")
    @classmethod
    def _clean(cls, value: str) -> str:
        return _clean_required(value)

    @field_validator("year", mode="after")
    @classmethod
    def _clean_year(cls, value: str | None) -> str | None:
        return _clean_optional(value)


def _dedupe_labels(values: list[str]) -> list[str]:
    """Drop placeholders and case-insensitive duplicates, preserving order.

    Shared by ``skills`` and ``ProjectRef.tech``: JSON schema can express
    "array of strings" but not "no duplicates, no junk".
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        if _is_placeholder(value):
            continue
        label = value.strip()
        # A whole sentence is a description the model mislabelled as a skill.
        if len(label) > 60:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(label)
    return cleaned


class ParsedResume(BaseModel):
    """The extraction target. This class *is* the prompt — its field names and
    descriptions are what the model sees via the JSON schema.

    **Every field here is an identifier or a short label. None of it is prose,
    and that is the point.**

    Extraction was measured at ~217s warm for a 600-word resume. The breakdown
    (scripts/benchmark_llm.py) showed generation runs at a flat ~4 tokens/sec
    and that wall-clock scales with *output token count only* — input size,
    schema nesting and constrained decoding were all ruled out, the last of
    those by an unconstrained run that came out slower (254s) than the
    constrained one. So the only lever is emitting fewer tokens.

    The removed fields — ``ExperienceEntry.summary``, ``ProjectRef.description``,
    the top-level ``summary`` — were the bulk of the ~950 output tokens, and
    every one of them was the model *transcribing prose that already exists
    verbatim in* ``resumes.raw_text``. Anything long needed for display is read
    from there. See DECISIONS 20.
    """

    name: str | None = Field(description="Candidate's full name")
    email: str | None = Field(description="Primary email address")
    phone: str | None = Field(description="Primary phone number")
    total_experience_years: float | None = Field(
        ge=0, le=60, description="Total years of professional experience"
    )

    # Required — deliberately no default. Pydantic only marks a field "required"
    # in the JSON schema when it has no default, and the grammar built from that
    # schema is what forces the model to emit the key at all. With defaults here,
    # qwen2.5:3b returned a valid object containing only name/location/education
    # and skipped the two fields that matter. Requiring them makes omission
    # unrepresentable rather than merely undesirable.
    #
    # skills is kept in full even though it is the longest list: it is the
    # single input to the skill component of every match score.
    skills: list[str] = Field(
        description="Technical and professional skills, one per item",
    )
    projects: list[ProjectRef] = Field(
        description="Named projects, with the technologies each used"
    )
    experience: list[ExperienceRef] = Field(
        description="Work history, most recent first"
    )
    education: list[EducationRef] = Field(
        description="Degrees and qualifications"
    )

    @field_validator("name", "phone", mode="after")
    @classmethod
    def _strip_placeholders(cls, value: str | None) -> str | None:
        return _clean_optional(value)

    @field_validator("projects", mode="after")
    @classmethod
    def _drop_untitled_projects(cls, values: list[ProjectRef]) -> list[ProjectRef]:
        """A project with no title is not a project — it is the model filling
        the array because the grammar allowed it."""
        return [project for project in values if project.title]

    @field_validator("experience", mode="after")
    @classmethod
    def _drop_empty_roles(cls, values: list[ExperienceRef]) -> list[ExperienceRef]:
        """Keep a role if it names either an employer or a title. Dropping only
        on both being empty, since plenty of resumes omit one."""
        return [entry for entry in values if entry.company or entry.role]

    @field_validator("education", mode="after")
    @classmethod
    def _drop_empty_education(cls, values: list[EducationRef]) -> list[EducationRef]:
        return [entry for entry in values if entry.institution or entry.degree]

    @field_validator("email", mode="after")
    @classmethod
    def _valid_email(cls, value: str | None) -> str | None:
        cleaned = _clean_optional(value)
        if cleaned is None:
            return None
        # Not full RFC validation: just enough to reject a hallucinated name or
        # "email@example.com". A missing email is fine; a wrong one is not.
        if "@" not in cleaned or "." not in cleaned.split("@")[-1]:
            return None
        if cleaned.casefold().endswith(("@example.com", "@email.com", "@domain.com")):
            return None
        return cleaned.casefold()

    @field_validator("skills", mode="after")
    @classmethod
    def _clean_skills(cls, values: list[str]) -> list[str]:
        return _dedupe_labels(values)

    @model_validator(mode="after")
    def _must_extract_something(self) -> "ParsedResume":
        """Reject an empty parse.

        The single most valuable check here. A well-formed object with no skills
        and no experience means extraction failed — the model returned the
        schema's skeleton. Without this the pipeline would happily store an empty
        resume and every downstream match would be garbage.
        """
        if not self.skills and not self.experience:
            raise ValueError(
                "Extracted neither skills nor work experience. "
                "Re-read the resume text and populate both."
            )
        return self


def build_resume_embedding_text(parsed: "ParsedResume | dict[str, Any]") -> str:
    """The text a resume's embedding is built from: skills, roles and project tech.

    Deliberately **not** ``raw_text``. A resume's raw text is mostly things that
    say nothing about employability — a postal address, a phone number, hobbies,
    "References available on request", the name of a school. Those tokens are a
    large fraction of the document and they pull every candidate's vector toward
    the same generic-CV centroid, which compresses the range of resume-to-job
    cosines and makes the semantic component nearly constant across jobs.

    Education is excluded for the same reason: a degree title matches every
    posting's boilerplate degree requirement about equally, so it adds distance
    to no one.

    **This text is thinner than it was.** It previously included each role's
    ``summary`` — a sentence or two of what the candidate actually did, which
    was the most job-description-like prose available. Those fields were removed
    from the schema because they were the bulk of a 950-token, 217-second
    extraction (see the ParsedResume docstring). What is left is nouns: skill
    names, job titles, employers, project names and project technologies. Nouns
    embed further from a job description's prose than prose does, so expect the
    resume-to-job cosine band to shift, and re-run
    ``scripts/calibrate_similarity.py`` before trusting COS_LO/COS_HI. This is a
    real trade, made knowingly: extraction is 4x faster and the skill component
    — which carries 60% of the score and is the half a candidate can act on — is
    unaffected, because ``skills`` was kept in full. See DECISIONS 20.4.

    Accepts either a :class:`ParsedResume` or the JSONB dict read back from
    ``resumes.parsed``, so the API can call it without a re-validation round
    trip through pydantic.

    Returns "" when there is nothing usable; the caller must not embed that
    (the provider rejects empty text, deliberately).
    """
    if isinstance(parsed, ParsedResume):
        data = parsed.model_dump()
    else:
        data = parsed or {}

    sections: list[str] = []

    skills = [str(s).strip() for s in (data.get("skills") or []) if str(s).strip()]
    if skills:
        # Comma-joined under a heading rather than one per line: the embedding
        # model reads prose, and a bare newline-separated list of nouns embeds
        # further from a job description's prose than the same list inline.
        sections.append("Skills: " + ", ".join(skills))

    role_lines: list[str] = []
    for entry in data.get("experience") or []:
        if not isinstance(entry, dict):
            continue
        # Role and employer only. `duration` is dropped — "2021 - Present" is
        # noise under cosine, exactly as the old start/end dates were.
        line = " at ".join(
            part
            for part in (entry.get("role"), entry.get("company"))
            if part and str(part).strip()
        )
        if line:
            role_lines.append(line)
    if role_lines:
        sections.append("Roles: " + "; ".join(role_lines))

    # Projects are the closest thing left to evidence of *what the candidate
    # did*, and their tech lists are the closest thing to a second skills
    # section — often naming tools the skills list omits.
    project_lines: list[str] = []
    for entry in data.get("projects") or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        tech = [str(t).strip() for t in (entry.get("tech") or []) if str(t).strip()]
        if not title and not tech:
            continue
        if title and tech:
            project_lines.append(f"{title} ({', '.join(tech)})")
        else:
            project_lines.append(title or ", ".join(tech))
    if project_lines:
        sections.append("Projects: " + "; ".join(project_lines))

    return "\n\n".join(sections).strip()


class StructureError(RuntimeError):
    """Extraction failed validation on every attempt."""

    def __init__(self, message: str, attempts: int, last_errors: str | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_errors = last_errors


# Describes the task only. It deliberately says nothing about JSON, formatting or
# output shape: that is enforced by the grammar, and prompt instructions about it
# would be redundant tokens that also invite the model to editorialise.
_SYSTEM_PROMPT = (
    "You extract structured data from resumes. Copy values verbatim from the "
    "document. If a field is genuinely absent, leave it empty rather than "
    "guessing or inventing a plausible value."
)


def _build_prompt(resume_text: str, previous_errors: str | None) -> str:
    parts = [
        "Extract every field you can from the following resume.",
        "",
        "--- RESUME ---",
        resume_text,
        "--- END RESUME ---",
    ]
    if previous_errors:
        # Feeding the validator's own message back is what makes the retry
        # different from a plain re-roll at temperature 0 — which, being
        # deterministic, would otherwise return the identical bad answer.
        parts += [
            "",
            "Your previous answer was rejected for these reasons:",
            previous_errors,
            "Correct them using only what the resume actually says.",
        ]
    return "\n".join(parts)


def _format_errors(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)


def extract_resume(
    resume_text: str,
    llm: LLMProvider | None = None,
    max_attempts: int | None = None,
) -> ParsedResume:
    """Extract a :class:`ParsedResume` from raw resume text.

    Retries on *validation* failure, feeding the errors back into the prompt.
    Does not retry :class:`LLMUnavailableError` — a stopped daemon will not fix
    itself, and retrying only delays a clear error message.

    Raises:
        StructureError: every attempt failed validation.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text is empty")

    from app.config import get_settings

    # get_parse_provider, not get_llm_provider: resume parsing is the one
    # call per upload that a user actually waits on, so it runs on the
    # hosted provider with a local fallback. See DECISIONS 22.4.
    provider = llm or get_parse_provider()
    attempts = max_attempts or get_settings().extraction_max_attempts

    text = resume_text.strip()
    if len(text) > MAX_RESUME_CHARS:
        logger.warning(
            "Resume truncated from %s to %s chars", len(text), MAX_RESUME_CHARS
        )
        text = text[:MAX_RESUME_CHARS]

    previous_errors: str | None = None

    for attempt in range(1, attempts + 1):
        raw: dict[str, Any] = provider.complete(
            _build_prompt(text, previous_errors),
            ParsedResume,
            system=_SYSTEM_PROMPT,
        )
        try:
            parsed = ParsedResume.model_validate(raw)
        except ValidationError as exc:
            previous_errors = _format_errors(exc)
            logger.warning(
                "Extraction attempt %s/%s failed validation:\n%s",
                attempt, attempts, previous_errors,
            )
            continue

        logger.info(
            "Extraction succeeded on attempt %s/%s: %s skills, %s roles",
            attempt, attempts, len(parsed.skills), len(parsed.experience),
        )
        return parsed

    raise StructureError(
        f"Resume extraction failed validation after {attempts} attempts",
        attempts=attempts,
        last_errors=previous_errors,
    )

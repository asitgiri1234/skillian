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

from app.providers import LLMProvider, get_llm_provider

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


class ExperienceEntry(BaseModel):
    """One role. Dates stay free text — resumes write them a dozen ways
    ("Jan 2020", "2020-01", "Spring 2020") and a wrong ISO date is worse than
    the original string."""

    # All required, all nullable. See the note on ParsedResume.skills: a field
    # with a default is absent from the schema's "required" list, and the model
    # then simply omits it. Required-and-nullable forces the key to be emitted
    # with an explicit null, which is a real answer; a missing key is not.
    company: str | None = Field(description="Employer name")
    title: str | None = Field(description="Job title held")
    start_date: str | None = Field(description="Start date exactly as written")
    end_date: str | None = Field(description="End date as written, or 'Present'")
    is_current: bool = Field(description="True if this is the candidate's current role")
    summary: str | None = Field(description="What the candidate did in this role")

    @field_validator("company", "title", "start_date", "end_date", "summary", mode="after")
    @classmethod
    def _strip_placeholders(cls, value: str | None) -> str | None:
        return _clean_optional(value)


class EducationEntry(BaseModel):
    institution: str | None = Field(description="School or university name")
    degree: str | None = Field(description="Degree awarded, e.g. 'B.E.'")
    field_of_study: str | None = Field(description="Subject studied")
    graduation_year: int | None = Field(description="Four-digit year of graduation")

    @field_validator("institution", "degree", "field_of_study", mode="after")
    @classmethod
    def _strip_placeholders(cls, value: str | None) -> str | None:
        return _clean_optional(value)

    @field_validator("graduation_year", mode="after")
    @classmethod
    def _plausible_year(cls, value: int | None) -> int | None:
        # Outside this range it is a page number or a hallucination, not a year.
        if value is not None and not (1950 <= value <= 2100):
            return None
        return value


class ParsedResume(BaseModel):
    """The extraction target. This class *is* the prompt — its field names and
    descriptions are what the model sees via the JSON schema."""

    name: str | None = Field(description="Candidate's full name")
    email: str | None = Field(description="Primary email address")
    phone: str | None = Field(description="Primary phone number")
    location: str | None = Field(description="City and country of residence")
    summary: str | None = Field(description="Professional summary or objective")

    # Required — deliberately no default. Pydantic only marks a field "required"
    # in the JSON schema when it has no default, and the grammar built from that
    # schema is what forces the model to emit the key at all. With defaults here,
    # qwen2.5:3b returned a valid object containing only name/location/education
    # and skipped the two fields that matter. Requiring them makes omission
    # unrepresentable rather than merely undesirable.
    skills: list[str] = Field(
        description="Technical and professional skills, one per item",
    )
    experience: list[ExperienceEntry] = Field(
        description="Work history, most recent first"
    )
    education: list[EducationEntry] = Field(
        description="Degrees and qualifications"
    )
    total_years_experience: float | None = Field(
        ge=0, le=60, description="Total years of professional experience"
    )

    @field_validator("name", "phone", "location", "summary", mode="after")
    @classmethod
    def _strip_placeholders(cls, value: str | None) -> str | None:
        return _clean_optional(value)

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
        """Drop placeholders and de-duplicate case-insensitively, preserving order.

        JSON schema can express "array of strings" but not "no duplicates, no
        junk" — exactly the gap this loop exists to cover.
        """
        seen: set[str] = set()
        cleaned: list[str] = []
        for value in values:
            if _is_placeholder(value):
                continue
            skill = value.strip()
            # A whole sentence is a description the model mislabelled as a skill.
            if len(skill) > 60:
                continue
            key = skill.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(skill)
        return cleaned

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
    """The text a resume's embedding is built from: skills and experience only.

    Deliberately **not** ``raw_text``. A resume's raw text is mostly things that
    say nothing about employability — a postal address, a phone number, hobbies,
    "References available on request", the name of a school. Those tokens are a
    large fraction of the document and they pull every candidate's vector toward
    the same generic-CV centroid, which compresses the range of resume-to-job
    cosines and makes the semantic component nearly constant across jobs.

    Education is excluded for the same reason: a degree title matches every
    posting's boilerplate degree requirement about equally, so it adds distance
    to no one. Skills and what the candidate actually *did* are the signal.

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
        # Title and company describe the role; the summary describes the work.
        # Dates are dropped — "Jan 2020 - Present" is noise under cosine.
        headline = " at ".join(
            part for part in (entry.get("title"), entry.get("company")) if part
        )
        summary = (entry.get("summary") or "").strip()
        line = ". ".join(part for part in (headline, summary) if part)
        if line:
            role_lines.append(line)
    if role_lines:
        sections.append("Experience:\n" + "\n".join(role_lines))

    # The professional summary is the candidate's own description of their work,
    # so it belongs with experience. Included last: it is often absent, and when
    # present it is the most boilerplate-prone field on the page.
    summary = (data.get("summary") or "").strip()
    if summary:
        sections.append("Summary: " + summary)

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

    provider = llm or get_llm_provider()
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

"""Natural-language explanations for a scored match.

The expensive stage, and the reason the pipeline is ordered the way it is: this
is a local 7b model producing prose, at roughly 5-15 seconds per job on CPU.
Scoring all 200 jobs takes seconds; explaining all 200 would take the better part
of an hour. So scoring is exhaustive and explanation is capped — see
:data:`MAX_EXPLANATIONS` and DECISIONS 16.6.

Free text, not a schema. An explanation is read by a person, and constraining the
decoder to JSON would spend tokens on punctuation and encourage the clipped,
list-shaped register that structured output tends to produce.
"""

from __future__ import annotations

import logging
import re

from app.matching.scorer import ScoreResult
from app.providers import LLMError, LLMProvider

logger = logging.getLogger(__name__)

#: How many of a run's matches get an explanation. Ranked by overall_score, so
#: these are the only ones a user is realistically going to read.
MAX_EXPLANATIONS = 20

#: Enough for three sentences with room to spare; stops a model that decides to
#: write a cover letter.
MAX_EXPLANATION_TOKENS = 220

#: Skills listed in the prompt. The full list of a job's missing skills can run
#: to twenty items, and a model handed twenty will name all twenty.
_SKILLS_IN_PROMPT = 6

# Openers models reach for that waste one of the three sentences saying nothing.
_PREAMBLE_RE = re.compile(
    r"^\s*(sure[,!.]?|certainly[,!.]?|of course[,!.]?|here(?:'s| is)[^.:]*[.:]|"
    r"explanation[.:]|based on the (?:above|provided)[^,]*,)\s*",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You explain to a job seeker why a specific posting was matched to them. "
    "Write two or three sentences of plain prose, addressed to the candidate as "
    "'you'. Name the specific skills involved. Be direct about gaps without "
    "being discouraging. Never invent a skill or a requirement that is not "
    "listed for you."
)


def _build_prompt(
    job_title: str,
    company: str | None,
    result: ScoreResult,
    candidate_years: float | None,
    required_years: float | None,
) -> str:
    matching = result.matching_skills[:_SKILLS_IN_PROMPT]
    missing = result.missing_skills[:_SKILLS_IN_PROMPT]

    lines = [
        f"Job title: {job_title}",
        f"Company: {company or 'not stated'}",
        f"Overall match score: {result.overall_score:.0%}",
    ]

    if result.skills_unparsed:
        # Say so explicitly. Given a blank skill list and no explanation of why,
        # the model reliably invents requirements to fill the gap.
        lines.append(
            "Skill requirements: this posting's requirements could not be "
            "extracted, so the match is based on overall similarity between the "
            "candidate's background and the description. Say this plainly and do "
            "not name any specific requirement of the job."
        )
    else:
        lines.append(
            "Skills the candidate has that this job asks for: "
            + (", ".join(matching) if matching else "none")
        )
        lines.append(
            "Skills this job asks for that the candidate does not list: "
            + (", ".join(missing) if missing else "none")
        )

    if required_years is not None and candidate_years is not None:
        lines.append(
            f"Experience: the posting asks for {required_years:g} years; the "
            f"candidate has about {candidate_years:g}."
        )

    lines += [
        "",
        "Write the explanation now. Two or three sentences, no preamble, no "
        "bullet points, no heading.",
    ]
    return "\n".join(lines)


def _tidy(text: str) -> str:
    """Strip the conversational scaffolding models wrap prose in."""
    cleaned = text.strip()
    # Some models fence prose as if it were code.
    cleaned = re.sub(r"^```[a-z]*\n?|```$", "", cleaned).strip()
    cleaned = _PREAMBLE_RE.sub("", cleaned).strip()
    # Surrounding quotes, which appear when the prompt is read as dictation.
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return " ".join(cleaned.split())


def explain_match(
    job_title: str,
    company: str | None,
    result: ScoreResult,
    llm: LLMProvider,
    *,
    candidate_years: float | None = None,
    required_years: float | None = None,
) -> str | None:
    """Write a 2-3 sentence explanation of one match.

    Returns None instead of raising when the model fails. An explanation is a
    nicety layered on top of a score that is already computed and stored; losing
    one must not fail a search run that has otherwise succeeded, and the match
    row is perfectly usable with ``explanation`` left NULL.

    ``temperature`` is left at the provider default of 0: the same match should
    produce the same explanation, so that a re-run does not silently reword every
    row and make the two runs look different when nothing changed.
    """
    prompt = _build_prompt(job_title, company, result, candidate_years, required_years)
    try:
        raw = llm.complete_text(
            prompt, system=_SYSTEM_PROMPT, max_tokens=MAX_EXPLANATION_TOKENS
        )
    except LLMError as exc:
        logger.warning("Explanation failed for %r: %r", job_title, exc)
        return None

    explanation = _tidy(raw)
    if not explanation:
        logger.warning("Explanation for %r was empty after cleanup", job_title)
        return None
    return explanation

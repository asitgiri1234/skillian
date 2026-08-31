"""Scoring a resume against a job. Pure arithmetic — no I/O of any kind.

This module deliberately knows nothing about SQLAlchemy, HTTP or Ollama. It
takes plain frozen dataclasses in and returns a :class:`ScoreResult`, which is
what makes it (a) exhaustively testable without a database and (b) fast enough
to run over every job in a search: scoring 200 jobs is set arithmetic plus a few
thousand dot products, which is milliseconds. The moment an LLM call enters this
file, a 200-job search becomes a 25-minute one. It must not.

The overall score is::

    overall = (W_SKILL * skill + W_SEMANTIC * semantic) * experience_multiplier

with a documented fallback to ``semantic * multiplier`` when a job's
requirements could not be parsed at all.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple
from uuid import UUID

logger = logging.getLogger(__name__)

# --- tunables ---------------------------------------------------------------

#: What a job's stated requirement level is worth when the candidate has it.
#: A preferred skill counts, but at well under half the weight of a hard
#: requirement — missing a "nice to have" should barely move the score.
SKILL_WEIGHTS: dict[str, float] = {"required": 1.0, "preferred": 0.4}

#: Skill overlap vs. semantic similarity. Skills lead because they are the
#: component a candidate can act on ("you are missing Kubernetes" is advice;
#: "your cosine is 0.61" is not), and because they are checkable — a skill match
#: is either there or it is not, whereas the semantic score is a soft signal
#: derived from a 768-dimensional model with no notion of what a job pays or
#: requires. Semantic keeps a substantial 40% because skill extraction is lossy:
#: it catches named technologies and misses everything phrased as a sentence.
W_SKILL, W_SEMANTIC = 0.6, 0.4

#: Cosine rescaling bounds. **These are placeholders and are known to be wrong.**
#: Embeddings of related English text do not use the [0, 1] range: in practice
#: every resume-to-job-chunk cosine lands in a narrow band, so an unrescaled
#: semantic score barely varies between the best and worst job in a result set
#: and contributes far less spread than its 40% weight implies. Rescaling
#: stretches the observed band back across [0, 1].
#:
#: Run ``python scripts/calibrate_similarity.py --resume-id <id>`` against real
#: ingested jobs and set COS_LO to roughly the 5th percentile and COS_HI to
#: roughly the 95th of the printed distribution. See DECISIONS 16.3.
COS_LO, COS_HI = 0.45, 0.85

#: How many chunks contribute to the semantic score. See semantic_component.
TOP_K_CHUNKS = 3

#: Stamped onto matches.model_version so a scoring change can invalidate old
#: rows selectively instead of truncating the table. Bump it when a weight,
#: a bound, or the formula changes.
SCORER_VERSION = "skillian-scorer-1"

# Free-text requirement labels seen in the wild, folded onto the two levels
# SKILL_WEIGHTS knows about. job_skills.requirement is a free-text column by
# design (models.py), so the scorer must not assume it only ever holds the two
# values the extractor currently writes.
_REQUIREMENT_ALIASES: dict[str, str] = {
    "required": "required",
    "require": "required",
    "must": "required",
    "must_have": "required",
    "essential": "required",
    "mandatory": "required",
    "preferred": "preferred",
    "prefer": "preferred",
    "nice_to_have": "preferred",
    "nice": "preferred",
    "optional": "preferred",
    "desired": "preferred",
    "desirable": "preferred",
    "bonus": "preferred",
    "plus": "preferred",
}


# --- inputs -----------------------------------------------------------------


@dataclass(frozen=True)
class JobSkillRef:
    """One row of ``job_skills``, carrying the skill's name for display."""

    skill_id: UUID
    name: str
    requirement: str | None


@dataclass(frozen=True)
class ResumeProfile:
    """Everything scoring needs from a resume. Built once per search."""

    resume_id: UUID
    #: ``resumes.embedding``, built from skills + experience (see
    #: structure.build_resume_embedding_text). None when not yet embedded.
    embedding: Sequence[float] | None = None
    #: Canonical ``skills.id`` values evidenced by this resume.
    skill_ids: frozenset[UUID] = frozenset()
    #: From ParsedResume.total_years_experience. None when unparseable.
    total_years_experience: float | None = None


@dataclass(frozen=True)
class JobPosting:
    """Everything scoring needs from a job."""

    job_id: UUID
    skills: tuple[JobSkillRef, ...] = ()
    #: One vector per ``job_chunks`` row, in chunk_index order.
    chunk_embeddings: tuple[Sequence[float], ...] = ()
    #: ``jobs.experience_min_years``. None for the large majority of postings.
    required_years: float | None = None


# --- outputs ----------------------------------------------------------------


class SkillMatch(NamedTuple):
    """``(score, matched, missing)`` — a tuple, but a named one."""

    score: float
    matched: list[str]
    missing: list[str]


@dataclass(frozen=True)
class ScoreResult:
    """One scored resume x job pair."""

    overall_score: float
    semantic_score: float
    #: 0.0 when requirements could not be parsed; read ``skills_unparsed`` to
    #: tell that apart from "matched none of the requirements".
    skill_score: float
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    #: True when the job had no ``job_skills`` rows, so the skill component was
    #: dropped and the score is semantic-only. The UI must say so — presenting
    #: an unweighted semantic score as if it were a full match is a lie.
    skills_unparsed: bool = False
    experience_multiplier: float = 1.0


# --- components -------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 for a zero-magnitude vector: cosine is undefined there, and 0.0
    ("no relationship") is the only defensible substitute for a NaN that would
    otherwise poison every downstream mean and comparison.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Cannot compare vectors of different widths: {len(a)} vs {len(b)}"
        )
    if not a:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    # One pass instead of three: this is the innermost loop of the whole
    # pipeline, run once per (job chunk) for every job in the result set.
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _weight_for(requirement: str | None) -> float:
    """SKILL_WEIGHTS lookup that survives an unrecognised requirement label.

    A KeyError here would abort a whole search run over one odd string in a
    free-text column. Unknown labels fold to ``preferred`` — the lower weight —
    because over-crediting an unknown label inflates scores, and an inflated
    match is worse than a slightly conservative one. A NULL requirement reads as
    ``required``: a skill listed with no qualifier is being asked for.
    """
    if requirement is None:
        return SKILL_WEIGHTS["required"]
    key = requirement.strip().casefold().replace("-", "_").replace(" ", "_")
    level = _REQUIREMENT_ALIASES.get(key)
    if level is None:
        logger.warning(
            "Unrecognised job_skills.requirement %r; treating as 'preferred'",
            requirement,
        )
        level = "preferred"
    return SKILL_WEIGHTS[level]


def skill_component(
    resume_skill_ids: frozenset[UUID] | set[UUID],
    job_skills: Sequence[JobSkillRef],
) -> SkillMatch | None:
    """Weighted recall of a job's requirements against a resume's skills.

    ``earned / possible``, where each requirement contributes its SKILL_WEIGHTS
    value. This is recall, not Jaccard: a candidate is not penalised for knowing
    things the job did not ask for. Under Jaccard, a senior engineer with forty
    skills would score *worse* against a five-skill job than a junior with
    exactly those five, which is the opposite of useful. See DECISIONS 16.4.

    Returns **None**, not 0.0, when the job has no parsed requirements. Those two
    cases mean completely different things — "we could not read this posting"
    versus "this candidate matches nothing in it" — and collapsing them would
    bury every job with a thin description at the bottom of the results as though
    the candidate had been rejected on merit.
    """
    if not job_skills:
        return None

    resume_skill_ids = frozenset(resume_skill_ids)
    earned = 0.0
    possible = 0.0
    matched: list[str] = []
    missing: list[str] = []

    for job_skill in job_skills:
        weight = _weight_for(job_skill.requirement)
        possible += weight
        if job_skill.skill_id in resume_skill_ids:
            earned += weight
            matched.append(job_skill.name)
        else:
            missing.append(job_skill.name)

    # Defensive: reachable only if every requirement weighed 0.0, which today's
    # SKILL_WEIGHTS cannot produce but a future tuning pass could.
    if possible <= 0.0:
        return None

    return SkillMatch(score=earned / possible, matched=matched, missing=missing)


def semantic_component(
    resume_vec: Sequence[float] | None,
    chunk_vecs: Sequence[Sequence[float]],
) -> float:
    """Rescaled similarity between a resume and the best passages of a job.

    Mean of the top :data:`TOP_K_CHUNKS` cosines (or all of them when the job has
    fewer), then linearly rescaled from the ``[COS_LO, COS_HI]`` band onto
    ``[0, 1]`` and clamped.

    Top-3 mean rather than max: max rewards a single lucky chunk, and every job
    description contains at least one paragraph of generic engineering prose
    that any technical resume scores well against. Requiring three good passages
    means the *posting* matches, not one sentence of it. Mean-over-all is the
    other failure — it dilutes a genuinely strong requirements section with the
    benefits boilerplate sitting next to it, which is exactly the problem
    chunking was introduced to solve. See DECISIONS 16.2.

    Returns 0.0 when either side has no vectors; there is no similarity to
    measure and 0.0 is the neutral floor the clamp would produce anyway.
    """
    if resume_vec is None or not chunk_vecs:
        return 0.0

    similarities = sorted(
        (cosine_similarity(resume_vec, chunk) for chunk in chunk_vecs),
        reverse=True,
    )
    top = similarities[:TOP_K_CHUNKS]
    mean = sum(top) / len(top)

    # COS_HI == COS_LO would divide by zero; treat a degenerate band as a step.
    if COS_HI <= COS_LO:
        logger.warning("COS_HI (%s) <= COS_LO (%s); falling back to a step", COS_HI, COS_LO)
        return 1.0 if mean >= COS_HI else 0.0

    rescaled = (mean - COS_LO) / (COS_HI - COS_LO)
    return min(1.0, max(0.0, rescaled))


def experience_multiplier(
    required_years: float | None, candidate_years: float | None
) -> float:
    """A soft penalty for being under a posting's stated experience bar.

    A multiplier, not a subtracted term: being three years short should discount
    an otherwise-excellent match, not flatten it.

    ``None`` on *either* side returns 1.0. Most job descriptions state no
    parseable experience requirement, and resumes frequently do not state a
    total either — treating unknown as zero would penalise the majority of pairs
    for a gap in our extraction rather than a gap in the candidate. Missing data
    is not evidence of a shortfall. See DECISIONS 16.5.

    The bands are deliberately gentle: a candidate one year short of a "5+ years"
    posting is, in practice, a fine applicant, and the largest possible penalty
    is 30%.
    """
    if required_years is None or candidate_years is None:
        return 1.0

    gap = required_years - candidate_years
    if gap <= 0:
        return 1.0
    if gap <= 1:
        return 0.95
    if gap <= 3:
        return 0.85
    return 0.70


# --- the whole score --------------------------------------------------------


def score(resume: ResumeProfile, job: JobPosting) -> ScoreResult:
    """Score one resume against one job.

    Pure: same inputs, same output, no side effects, no I/O.
    """
    semantic = semantic_component(resume.embedding, job.chunk_embeddings)
    multiplier = experience_multiplier(
        job.required_years, resume.total_years_experience
    )
    skills = skill_component(resume.skill_ids, job.skills)

    if skills is None:
        # No requirements parsed: fall back to semantic alone rather than
        # blending in a 0.0 skill score, which would halve the score of every
        # thin posting for a reason that has nothing to do with the candidate.
        # Note this is *not* renormalised by W_SEMANTIC — the semantic score is
        # carrying the full weight here, and dividing it by 0.4 would push it
        # above every properly-scored job instead.
        overall = semantic * multiplier
        return ScoreResult(
            overall_score=overall,
            semantic_score=semantic,
            skill_score=0.0,
            matching_skills=[],
            missing_skills=[],
            skills_unparsed=True,
            experience_multiplier=multiplier,
        )

    overall = (W_SKILL * skills.score + W_SEMANTIC * semantic) * multiplier
    return ScoreResult(
        overall_score=overall,
        semantic_score=semantic,
        skill_score=skills.score,
        matching_skills=skills.matched,
        missing_skills=skills.missing,
        skills_unparsed=False,
        experience_multiplier=multiplier,
    )

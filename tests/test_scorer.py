"""Scorer arithmetic. Pure — no database, no network, no model."""

from __future__ import annotations

import math
import uuid

import pytest

from app.matching import scorer
from app.matching.scorer import (
    COS_HI,
    COS_LO,
    W_SEMANTIC,
    W_SKILL,
    JobPosting,
    JobSkillRef,
    ResumeProfile,
    cosine_similarity,
    experience_multiplier,
    score,
    semantic_component,
    skill_component,
)


def _skill(name: str, requirement: str | None = "required") -> JobSkillRef:
    return JobSkillRef(skill_id=uuid.uuid4(), name=name, requirement=requirement)


# --- cosine_similarity ------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self) -> None:
        vector = [0.3, 0.4, 0.5, 0.7]
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_opposite_vectors_are_minus_one(self) -> None:
        assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)

    def test_magnitude_does_not_matter(self) -> None:
        """Cosine is scale-invariant — the whole reason it is used here."""
        assert cosine_similarity([1.0, 1.0], [50.0, 50.0]) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        # 45 degrees between (1,0) and (1,1).
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(
            1 / math.sqrt(2)
        )

    def test_zero_vector_is_zero_not_nan(self) -> None:
        """Undefined mathematically; 0.0 keeps NaN out of every downstream mean."""
        result = cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert result == 0.0
        assert not math.isnan(result)

    def test_mismatched_widths_raise(self) -> None:
        with pytest.raises(ValueError, match="different widths"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_empty_vectors_are_zero(self) -> None:
        assert cosine_similarity([], []) == 0.0


# --- skill_component --------------------------------------------------------


class TestSkillComponent:
    def test_returns_none_when_job_has_no_skills(self) -> None:
        """The headline case: None, never 0.0.

        "We could not parse this posting's requirements" and "this candidate
        matches none of them" must stay distinguishable, or every thin posting
        is buried as though the candidate had been rejected on merit.
        """
        assert skill_component({uuid.uuid4()}, []) is None

    def test_none_is_not_falsy_confusable_with_zero(self) -> None:
        """`if not result:` would treat both the same. Callers must check
        `is None`, and the type makes that the only thing that works."""
        empty = skill_component(set(), [])
        assert empty is None
        assert not isinstance(empty, tuple), "a 0.0-scored tuple would be wrong"

    def test_zero_score_when_nothing_matches(self) -> None:
        """Contrast with the above: requirements parsed, none held."""
        result = skill_component(set(), [_skill("Rust"), _skill("Go")])
        assert result is not None
        assert result.score == 0.0
        assert result.matched == []
        assert result.missing == ["Rust", "Go"]

    def test_all_required_matched_is_one(self) -> None:
        python, sql = _skill("Python"), _skill("SQL")
        result = skill_component({python.skill_id, sql.skill_id}, [python, sql])
        assert result is not None
        assert result.score == pytest.approx(1.0)
        assert result.matched == ["Python", "SQL"]
        assert result.missing == []

    def test_preferred_weighs_less_than_required(self) -> None:
        required = _skill("Python", "required")
        preferred = _skill("Kubernetes", "preferred")
        skills = [required, preferred]

        # Holding only the required one: 1.0 / 1.4
        has_required = skill_component({required.skill_id}, skills)
        # Holding only the preferred one: 0.4 / 1.4
        has_preferred = skill_component({preferred.skill_id}, skills)

        assert has_required is not None and has_preferred is not None
        assert has_required.score == pytest.approx(1.0 / 1.4)
        assert has_preferred.score == pytest.approx(0.4 / 1.4)
        assert has_required.score > has_preferred.score

    def test_is_recall_not_jaccard(self) -> None:
        """Extra skills the job never asked for must not dilute the score.

        Under Jaccard a senior with forty skills would score worse against a
        two-skill job than a junior with exactly those two.
        """
        python, sql = _skill("Python"), _skill("SQL")
        exact = skill_component({python.skill_id, sql.skill_id}, [python, sql])
        generalist = skill_component(
            {python.skill_id, sql.skill_id, *(uuid.uuid4() for _ in range(38))},
            [python, sql],
        )
        assert exact is not None and generalist is not None
        assert generalist.score == exact.score == pytest.approx(1.0)

    def test_null_requirement_counts_as_required(self) -> None:
        skill = _skill("Python", None)
        result = skill_component({skill.skill_id}, [skill])
        assert result is not None and result.score == pytest.approx(1.0)

    def test_unknown_requirement_label_does_not_raise(self) -> None:
        """job_skills.requirement is free text; a surprise value must not abort
        an entire search run with a KeyError."""
        odd = _skill("Python", "would-be-lovely")
        result = skill_component({odd.skill_id}, [odd])
        assert result is not None
        assert result.score == pytest.approx(1.0)

    def test_requirement_aliases_are_folded(self) -> None:
        nice = _skill("Kubernetes", "nice_to_have")
        must = _skill("Python", "must_have")
        result = skill_component({nice.skill_id}, [nice, must])
        assert result is not None
        # 0.4 earned out of 0.4 + 1.0 possible.
        assert result.score == pytest.approx(0.4 / 1.4)

    def test_accepts_a_frozenset_or_a_set(self) -> None:
        skill = _skill("Python")
        assert skill_component(frozenset({skill.skill_id}), [skill]) is not None
        assert skill_component({skill.skill_id}, [skill]) is not None


# --- semantic_component -----------------------------------------------------


class TestSemanticComponent:
    def test_fewer_than_three_chunks_uses_all_of_them(self) -> None:
        """A short JD produces one or two chunks; averaging over a fixed 3 would
        divide by chunks that do not exist."""
        resume = [1.0, 0.0]
        one_chunk = semantic_component(resume, [[1.0, 0.0]])
        two_chunks = semantic_component(resume, [[1.0, 0.0], [1.0, 0.0]])
        # Both are cosine 1.0 throughout, so both rescale to the same 1.0.
        assert one_chunk == pytest.approx(1.0)
        assert two_chunks == pytest.approx(1.0)

    def test_single_chunk_is_just_that_chunk(self) -> None:
        resume = [1.0, 0.0]
        # cos = 1/sqrt(2) ~= 0.7071, inside [COS_LO, COS_HI].
        expected = (1 / math.sqrt(2) - COS_LO) / (COS_HI - COS_LO)
        assert semantic_component(resume, [[1.0, 1.0]]) == pytest.approx(expected)

    def test_two_chunks_average_both(self) -> None:
        resume = [1.0, 0.0]
        mean = (1.0 + 1 / math.sqrt(2)) / 2
        expected = min(1.0, max(0.0, (mean - COS_LO) / (COS_HI - COS_LO)))
        assert semantic_component(resume, [[1.0, 0.0], [1.0, 1.0]]) == pytest.approx(
            expected
        )

    def test_takes_the_top_three_not_all(self) -> None:
        """A long tail of irrelevant chunks — benefits, legal boilerplate — must
        not drag down a job whose requirements section is a strong match."""
        resume = [1.0, 0.0]
        strong = [[1.0, 0.0]] * 3
        weak = [[0.0, 1.0]] * 20
        assert semantic_component(resume, strong + weak) == pytest.approx(1.0)

    def test_top_three_is_not_max(self) -> None:
        """One lucky chunk must not carry the job. Every JD has a paragraph of
        generic engineering prose that any technical resume scores well on."""
        resume = [1.0, 0.0]
        one_good = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        all_good = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        assert semantic_component(resume, one_good) < semantic_component(resume, all_good)

    def test_no_chunks_is_zero(self) -> None:
        assert semantic_component([1.0, 0.0], []) == 0.0

    def test_no_resume_vector_is_zero(self) -> None:
        assert semantic_component(None, [[1.0, 0.0]]) == 0.0

    def test_rescaling_clamps_below_and_above(self) -> None:
        resume = [1.0, 0.0]
        # cos 0.0 is far below COS_LO -> clamps to 0.0.
        assert semantic_component(resume, [[0.0, 1.0]]) == 0.0
        # cos 1.0 is above COS_HI -> clamps to 1.0.
        assert semantic_component(resume, [[1.0, 0.0]]) == 1.0

    def test_rescaling_stretches_the_observed_band(self) -> None:
        """The reason rescaling exists: a small raw gap becomes a usable one."""
        resume = [1.0, 0.0]
        # Two vectors whose raw cosines differ by ~0.09 within the band.
        near = [1.0, 0.55]   # cos ~= 0.8761
        far = [1.0, 0.75]    # cos ~= 0.8
        raw_gap = cosine_similarity(resume, near) - cosine_similarity(resume, far)
        scaled_gap = semantic_component(resume, [near]) - semantic_component(resume, [far])
        assert scaled_gap > raw_gap


# --- experience_multiplier --------------------------------------------------


class TestExperienceMultiplier:
    def test_none_required_is_one(self) -> None:
        """Most JDs state no parseable requirement. Missing data is not a gap."""
        assert experience_multiplier(None, 3.0) == 1.0
        assert experience_multiplier(None, None) == 1.0
        assert experience_multiplier(None, 0.0) == 1.0

    def test_none_candidate_is_also_one(self) -> None:
        """Same principle in the other direction: an unparsed resume total is
        our failure, not the candidate's."""
        assert experience_multiplier(5.0, None) == 1.0

    def test_meeting_or_exceeding_is_one(self) -> None:
        assert experience_multiplier(5.0, 5.0) == 1.0
        assert experience_multiplier(5.0, 12.0) == 1.0

    @pytest.mark.parametrize(
        ("required", "candidate", "expected"),
        [
            (5.0, 4.5, 0.95),   # half a year short
            (5.0, 4.0, 0.95),   # exactly one year, inclusive
            (5.0, 3.9, 0.85),   # just over one
            (5.0, 2.0, 0.85),   # exactly three, inclusive
            (5.0, 1.9, 0.70),   # over three
            (10.0, 0.0, 0.70),  # the floor
        ],
    )
    def test_bands(self, required: float, candidate: float, expected: float) -> None:
        assert experience_multiplier(required, candidate) == pytest.approx(expected)

    def test_penalty_is_bounded(self) -> None:
        """A multiplier, not a subtraction: the worst case is a 30% discount, not
        a zeroed score."""
        assert experience_multiplier(40.0, 0.0) == 0.70


# --- score ------------------------------------------------------------------


class TestScore:
    def test_combines_components_with_the_stated_weights(self) -> None:
        python = _skill("Python")
        resume = ResumeProfile(
            resume_id=uuid.uuid4(),
            embedding=[1.0, 0.0],
            skill_ids=frozenset({python.skill_id}),
        )
        job = JobPosting(
            job_id=uuid.uuid4(), skills=(python,), chunk_embeddings=([1.0, 0.0],)
        )
        result = score(resume, job)

        assert result.skill_score == pytest.approx(1.0)
        assert result.semantic_score == pytest.approx(1.0)
        assert result.overall_score == pytest.approx(W_SKILL * 1.0 + W_SEMANTIC * 1.0)
        assert result.skills_unparsed is False

    def test_falls_back_to_semantic_when_skills_unparsed(self) -> None:
        resume = ResumeProfile(resume_id=uuid.uuid4(), embedding=[1.0, 0.0])
        job = JobPosting(
            job_id=uuid.uuid4(), skills=(), chunk_embeddings=([1.0, 0.0],)
        )
        result = score(resume, job)

        assert result.skills_unparsed is True
        assert result.semantic_score == pytest.approx(1.0)
        # Semantic alone, NOT semantic * W_SEMANTIC — blending in a 0.0 skill
        # score would halve every thin posting for no reason to do with the
        # candidate.
        assert result.overall_score == pytest.approx(1.0)
        assert result.skill_score == 0.0
        assert result.matching_skills == []

    def test_unparsed_flag_distinguishes_from_a_genuine_zero(self) -> None:
        resume = ResumeProfile(resume_id=uuid.uuid4(), embedding=[1.0, 0.0])
        unparsed = score(resume, JobPosting(job_id=uuid.uuid4(), skills=()))
        matched_none = score(
            resume, JobPosting(job_id=uuid.uuid4(), skills=(_skill("Rust"),))
        )
        assert unparsed.skill_score == matched_none.skill_score == 0.0
        assert unparsed.skills_unparsed is True
        assert matched_none.skills_unparsed is False
        assert matched_none.missing_skills == ["Rust"]

    def test_experience_multiplier_is_applied(self) -> None:
        python = _skill("Python")
        resume = ResumeProfile(
            resume_id=uuid.uuid4(),
            embedding=[1.0, 0.0],
            skill_ids=frozenset({python.skill_id}),
            total_years_experience=1.0,
        )
        job = JobPosting(
            job_id=uuid.uuid4(),
            skills=(python,),
            chunk_embeddings=([1.0, 0.0],),
            required_years=5.0,
        )
        result = score(resume, job)
        assert result.experience_multiplier == pytest.approx(0.70)
        assert result.overall_score == pytest.approx(
            (W_SKILL * 1.0 + W_SEMANTIC * 1.0) * 0.70
        )

    def test_result_stays_within_zero_and_one(self) -> None:
        python = _skill("Python")
        resume = ResumeProfile(
            resume_id=uuid.uuid4(),
            embedding=[1.0, 0.0],
            skill_ids=frozenset({python.skill_id}),
        )
        job = JobPosting(
            job_id=uuid.uuid4(), skills=(python,), chunk_embeddings=([1.0, 0.0],)
        )
        result = score(resume, job)
        # Numeric(5,4) cannot hold anything above 9.9999, and a score above 1.0
        # would be meaningless anyway.
        assert 0.0 <= result.overall_score <= 1.0

    def test_no_embedding_degrades_to_skills_only(self) -> None:
        python = _skill("Python")
        resume = ResumeProfile(
            resume_id=uuid.uuid4(),
            embedding=None,
            skill_ids=frozenset({python.skill_id}),
        )
        result = score(resume, JobPosting(job_id=uuid.uuid4(), skills=(python,)))
        assert result.semantic_score == 0.0
        assert result.overall_score == pytest.approx(W_SKILL * 1.0)

    def test_is_pure(self) -> None:
        """Same inputs, same output, twice."""
        python = _skill("Python")
        resume = ResumeProfile(
            resume_id=uuid.uuid4(),
            embedding=[0.3, 0.9],
            skill_ids=frozenset({python.skill_id}),
        )
        job = JobPosting(
            job_id=uuid.uuid4(), skills=(python,), chunk_embeddings=([0.5, 0.5],)
        )
        assert score(resume, job) == score(resume, job)


def test_weights_sum_to_one() -> None:
    """If they ever do not, overall_score silently leaves [0, 1] and no longer
    fits matches.overall_score's Numeric(5,4)."""
    assert W_SKILL + W_SEMANTIC == pytest.approx(1.0)


def test_calibration_bounds_are_ordered() -> None:
    assert scorer.COS_LO < scorer.COS_HI

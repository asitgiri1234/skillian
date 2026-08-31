"""Templated match explanations. Pure — no database, no model."""

from __future__ import annotations

import pytest

from app.matching.explain import (
    MAX_LISTED_SKILLS,
    THIN_EVIDENCE_MAX,
    _join,
    render_explanation,
)


def render(**kwargs) -> str:
    base = dict(tier="strong", matching_skills=["Python"], missing_skills=[], parsed_count=5)
    return render_explanation(**{**base, **kwargs})


class TestJoin:
    def test_single(self) -> None:
        assert _join(["Python"]) == "Python"

    def test_two(self) -> None:
        assert _join(["Python", "Go"]) == "Python and Go"

    def test_three(self) -> None:
        assert _join(["Python", "Go", "Rust"]) == "Python, Go and Rust"

    def test_elides_past_the_cap(self) -> None:
        """Beyond five, a card becomes a wall of nouns."""
        names = [f"S{i}" for i in range(9)]
        out = _join(names)
        assert out.endswith("and 4 more")
        assert out.count(",") == MAX_LISTED_SKILLS - 1

    def test_empty(self) -> None:
        assert _join([]) == ""


class TestTiers:
    def test_strong_names_what_you_have(self) -> None:
        text = render(tier="strong", matching_skills=["Python", "Django"],
                      missing_skills=["Flask"])
        assert text.startswith("Strong match.")
        assert "Python and Django" in text
        assert "Flask" in text

    def test_strong_with_nothing_missing(self) -> None:
        text = render(tier="strong", matching_skills=["Python"], missing_skills=[])
        assert "nothing it asks for is missing" in text

    def test_moderate_frames_the_gap_as_actionable(self) -> None:
        text = render(tier="moderate", matching_skills=["Python"],
                      missing_skills=["Kubernetes"])
        assert text.startswith("Decent match.")
        assert "would strengthen your fit" in text

    def test_weak_leads_with_what_is_needed(self) -> None:
        text = render(tier="weak", matching_skills=[], missing_skills=["Salesforce"])
        assert text.startswith("Weak fit.")
        assert "Salesforce" in text

    def test_weak_number_agreement(self) -> None:
        assert "which isn't" in render(tier="weak", matching_skills=[],
                                       missing_skills=["SEO"])
        assert "which aren't" in render(tier="weak", matching_skills=[],
                                        missing_skills=["SEO", "Salesforce"])

    def test_never_empty(self) -> None:
        """Every card needs a reason; a blank string is not one."""
        for tier in ("strong", "moderate", "weak", None):
            assert render(tier=tier, matching_skills=[], missing_skills=[]).strip()


class TestThinEvidence:
    """The honest surfacing of the asymmetry in DECISIONS 30.1 — the reason
    skill_confidence is stored rather than discarded."""

    @pytest.mark.parametrize("n", [1, 2])
    def test_discloses_thin_evidence(self, n: int) -> None:
        text = render(parsed_count=n)
        assert f"only {n} stated requirement" in text
        assert "rough signal" in text

    def test_singular_and_plural(self) -> None:
        assert "1 stated requirement," in render(parsed_count=1)
        assert "2 stated requirements," in render(parsed_count=2)

    @pytest.mark.parametrize("n", [3, 6, 20])
    def test_stays_silent_when_evidence_is_adequate(self, n: int) -> None:
        assert "rough signal" not in render(parsed_count=n)

    def test_a_strong_match_on_one_requirement_still_discloses(self) -> None:
        """The exact case that made 'Account Executive matched Git' rank 7th."""
        text = render(tier="strong", matching_skills=["Git"], missing_skills=[],
                      parsed_count=1)
        assert text.startswith("Strong match.")
        assert "only 1 stated requirement" in text

    def test_threshold_constant_is_respected(self) -> None:
        assert "rough signal" in render(parsed_count=THIN_EVIDENCE_MAX)
        assert "rough signal" not in render(parsed_count=THIN_EVIDENCE_MAX + 1)


class TestUnparsed:
    def test_states_that_requirements_were_not_read(self) -> None:
        text = render_explanation(tier=None, matching_skills=[], missing_skills=[],
                                  parsed_count=0, skills_unparsed=True)
        assert "could not be read" in text
        assert "overall similarity" in text

    def test_claims_no_skill_match(self) -> None:
        """The top of the unparsed bucket scores 0.999 on semantics alone. The
        text must not imply anything was checked against the resume."""
        text = render_explanation(tier=None, matching_skills=[], missing_skills=[],
                                  parsed_count=0, skills_unparsed=True)
        assert "nothing was checked" in text
        for word in ("Strong match", "Decent match", "Weak fit"):
            assert word not in text

    def test_ignores_any_stale_skill_lists(self) -> None:
        text = render_explanation(tier=None, matching_skills=["Python"],
                                  missing_skills=["Go"], parsed_count=0,
                                  skills_unparsed=True)
        assert "Python" not in text and "Go" not in text


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        """A re-run must not silently reword every card."""
        kwargs = dict(tier="moderate", matching_skills=["Python", "Go"],
                      missing_skills=["Rust"], parsed_count=4)
        assert render_explanation(**kwargs) == render_explanation(**kwargs)

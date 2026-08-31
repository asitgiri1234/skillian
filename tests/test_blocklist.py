"""The skill blocklist and the job-title guard. Pure — no database."""

from __future__ import annotations

import pytest

from app.matching.blocklist import (
    blocked_keys,
    is_disallowed,
    load_blocklist,
)


class TestBlocklistData:
    def test_loads_from_the_checked_in_csv(self) -> None:
        entries = load_blocklist()
        assert len(entries) >= 30

    def test_every_entry_has_a_reason_and_category(self) -> None:
        """The blocklist is a set of judgements; an entry without a stated
        reason is unreviewable."""
        for entry in load_blocklist():
            assert entry.term, "blank term"
            assert entry.category, f"{entry.term} has no category"
            assert len(entry.reason) > 15, f"{entry.term} has a thin reason"

    def test_no_duplicate_terms(self) -> None:
        terms = [e.term.casefold() for e in load_blocklist()]
        assert len(terms) == len(set(terms))

    def test_the_measured_offenders_are_present(self) -> None:
        """Backend Engineer was the most-matched row in the corpus at 25 of 80
        jobs; these are the ones the survey identified."""
        keys = blocked_keys()
        from app.normalize import normalize_text

        for term in ("Backend Engineer", "AI", "architecture", "Databases",
                     "APIs", "Architect", "Fintech", "Payments", "Scalability"):
            assert normalize_text(term) in keys, term


class TestIsDisallowed:
    @pytest.mark.parametrize(
        "term", ["Backend Engineer", "AI", "architecture", "exp", "Promises"]
    )
    def test_blocklisted_terms_are_refused(self, term: str) -> None:
        assert is_disallowed(term) is not None

    @pytest.mark.parametrize(
        "term",
        [
            "Data Platform Engineer",
            "Senior Developer",
            "Solutions Architect",
            "Engineering Manager",
            "Business Analyst",
            "Integration Specialist",
            "Site Reliability Engineering",
        ],
    )
    def test_job_title_shapes_are_refused(self, term: str) -> None:
        """The recurrence guard. Without it a blocklisted term reappears under a
        fresh id on the next run and the pruning is silently undone."""
        assert is_disallowed(term) == "reads as a job title, not a skill"

    @pytest.mark.parametrize(
        "term",
        ["Python", "Kubernetes", "PostgreSQL", "UPI", "QR", "P2P",
         "Machine Learning", "CI/CD", "React", "Data Structures"],
    )
    def test_real_skills_are_allowed(self, term: str) -> None:
        assert is_disallowed(term) is None

    def test_the_four_borderline_terms_survive(self) -> None:
        """distributed systems, Data Structures, Security and Agile were judged
        genuine screenable competencies and deliberately kept."""
        for term in ("distributed systems", "Data Structures", "Security", "Agile"):
            assert is_disallowed(term) is None, term

    def test_a_trailing_title_word_only_counts_at_the_end(self) -> None:
        """'Engineering Productivity Tooling' is a real area; only a *trailing*
        title word indicates a job title."""
        assert is_disallowed("Engineering Productivity Tooling") is None

    @pytest.mark.parametrize("term", ["", "   ", "!!!"])
    def test_empty_and_junk(self, term: str) -> None:
        assert is_disallowed(term) is not None

    def test_case_insensitive(self) -> None:
        assert is_disallowed("BACKEND ENGINEER") is not None
        assert is_disallowed("backend engineer") is not None

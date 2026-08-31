"""Skill-name cleaning. Pure — no database, no network, no model.

Every case in :class:`TestRealExtractorOutput` is a string `qwen2.5:7b` actually
returned during day-3 verification. Stored verbatim, they broke matching
completely: a candidate with Python, FastAPI, Docker, Kubernetes and Kafka scored
`skill=0.065` against a senior backend posting, because "Strong Python" and
"Comfortable with Docker" do not canonicalise onto "Python" and "Docker". These
are the regression tests for that fix.
"""

from __future__ import annotations

import pytest

from app.matching.skills import ExtractedSkills, clean_skill_name


class TestRealExtractorOutput:
    """Strings the model actually produced, and what they must become."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Strong Python", ["Python"]),
            ("Comfortable with Docker", ["Docker"]),
            ("Familiarity with Docker", ["Docker"]),
            ("Terraform experience", ["Terraform"]),
            ("AWS experience", ["AWS"]),
            ("hands-on experience with Kafka", ["Kafka"]),
            ("proficiency in TypeScript", ["TypeScript"]),
            ("3+ years Python", ["Python"]),
            ("FastAPI or Django in production", ["FastAPI", "Django"]),
            ("Expert React and TypeScript", ["React", "TypeScript"]),
            ("Airflow or Dagster", ["Airflow", "Dagster"]),
            ("Go/Rust", ["Go", "Rust"]),
            ("Deep PostgreSQL knowledge: schema design", ["PostgreSQL"]),
            ("Monitoring with Prometheus and Grafana", ["Prometheus", "Grafana"]),
        ],
    )
    def test_reduces_to_canonical_names(self, raw: str, expected: list[str]) -> None:
        assert clean_skill_name(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "5+ years of professional backend development",
            "8+ years of software engineering",
            "Experience running services in production, including on-call",
            "Operating systems or services you built",
            "3+ years in data engineering",
        ],
    )
    def test_drops_requirements_that_are_not_skills(self, raw: str) -> None:
        """A seniority bar belongs in jobs.experience_min_years, not skills."""
        assert clean_skill_name(raw) == []


class TestAlreadyClean:
    @pytest.mark.parametrize(
        "name",
        [
            "Python", "PostgreSQL", "Kubernetes", "dbt", "PyTorch",
            "REST APIs", "Amazon Web Services", "GitHub Actions",
        ],
    )
    def test_passes_through_unchanged(self, name: str) -> None:
        """The common case must be a no-op, not a mangling."""
        assert clean_skill_name(name) == [name]


class TestSlashCompounds:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("CI/CD", ["CI/CD"]),
            ("CI/CD pipelines", ["CI/CD pipelines"]),
            ("UI/UX design", ["UI/UX design"]),
            ("TCP/IP networking", ["TCP/IP networking"]),
        ],
    )
    def test_known_compounds_survive_the_split(self, raw: str, expected: list[str]) -> None:
        """Without the escape hatch, "CI/CD" becomes the two useless skills
        "CI" and "CD"."""
        assert clean_skill_name(raw) == expected

    def test_an_unknown_compound_still_splits(self) -> None:
        """"Go/Rust" genuinely is two skills; no heuristic separates the cases,
        which is why the compound list is explicit."""
        assert clean_skill_name("Go/Rust") == ["Go", "Rust"]


class TestEdgeCases:
    @pytest.mark.parametrize("raw", ["", "   ", ".", "-", "n/a"])
    def test_empty_and_junk_yield_nothing(self, raw: str) -> None:
        assert clean_skill_name(raw) == []

    def test_generic_disciplines_are_dropped(self) -> None:
        assert clean_skill_name("software engineering") == []
        assert clean_skill_name("communication") == []

    def test_a_long_phrase_is_dropped(self) -> None:
        assert clean_skill_name(
            "the ability to work independently in a fast paced startup"
        ) == []

    def test_stacked_qualifiers_are_all_stripped(self) -> None:
        assert clean_skill_name("strong hands-on experience with Kubernetes") == [
            "Kubernetes"
        ]

    def test_case_is_preserved(self) -> None:
        """Canonicalisation matches case-insensitively, but the stored display
        name should read the way a human wrote it."""
        assert clean_skill_name("Strong PostgreSQL") == ["PostgreSQL"]


class TestExtractedSkillsValidator:
    def test_applies_cleaning_and_deduplicates(self) -> None:
        parsed = ExtractedSkills.model_validate(
            {
                "skills": [
                    {"name": "Strong Python", "requirement": "required"},
                    {"name": "Python experience", "requirement": "preferred"},
                    {"name": "5+ years of backend development", "requirement": "required"},
                    {"name": "FastAPI or Django", "requirement": "preferred"},
                ]
            }
        )
        names = [skill.name for skill in parsed.skills]
        assert names == ["Python", "FastAPI", "Django"]

    def test_the_stronger_requirement_level_wins(self) -> None:
        """A posting saying "React required" in one place and "React a plus" in
        another is asking for React."""
        parsed = ExtractedSkills.model_validate(
            {
                "skills": [
                    {"name": "React a plus", "requirement": "preferred"},
                    {"name": "Strong React", "requirement": "required"},
                ]
            }
        )
        assert len(parsed.skills) == 1
        assert parsed.skills[0].requirement == "required"

    def test_one_entry_can_become_two(self) -> None:
        parsed = ExtractedSkills.model_validate(
            {"skills": [{"name": "Go/Rust", "requirement": "required"}]}
        )
        assert [s.name for s in parsed.skills] == ["Go", "Rust"]
        assert all(s.requirement == "required" for s in parsed.skills)

    def test_one_entry_can_become_none(self) -> None:
        parsed = ExtractedSkills.model_validate(
            {"skills": [{"name": "8+ years of experience", "requirement": "required"}]}
        )
        assert parsed.skills == []

    def test_an_empty_extraction_is_valid(self) -> None:
        """Unlike a resume parse, an empty result here is a legitimate answer —
        plenty of postings list no concrete requirements."""
        assert ExtractedSkills.model_validate({"skills": []}).skills == []

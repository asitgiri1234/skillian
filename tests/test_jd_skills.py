"""Dictionary skill extraction. Pure — the index is built in memory, no database.

The two cases named in the spec — "Go to our website" and "R&D team" — have
their own tests, because a short-name false positive is the failure mode that
would quietly poison every match score: a job wrongly tagged with Go matches
every Go developer.
"""

from __future__ import annotations

import re
import uuid

import pytest

from app.matching.jd_skills import (
    PREFERRED,
    REQUIRED,
    SkillIndex,
    _boundary,
    extract_skills,
    split_sections,
)

# --- a hand-built index, so these tests need no Postgres --------------------

VOCAB = {
    "Python": ["py"],
    "JavaScript": ["js"],
    "Kubernetes": ["k8s"],
    "PostgreSQL": ["postgres"],
    "Machine Learning": ["ml"],
    "Learning": [],
    "Go": ["golang"],
    "R": [],
    "C": [],
    "C++": ["cpp"],
    "C#": ["csharp"],
    "D": [],
    "Docker": [],
    "Kafka": [],
    "Terraform": [],
    ".NET": ["dotnet"],
}


@pytest.fixture(scope="module")
def index() -> SkillIndex:
    lookup: dict[str, tuple[uuid.UUID, str]] = {}
    for name, aliases in VOCAB.items():
        skill_id = uuid.uuid5(uuid.NAMESPACE_DNS, name)
        for form in (name, *aliases):
            lookup.setdefault(form.casefold(), (skill_id, name))
    forms = sorted(lookup, key=len, reverse=True)
    pattern = re.compile("|".join(_boundary(f) for f in forms), re.IGNORECASE)
    return SkillIndex(pattern=pattern, lookup=lookup)


def names(hits) -> set[str]:
    return {h.name for h in hits}


def level(hits, name: str) -> str | None:
    return next((h.requirement for h in hits if h.name == name), None)


# --- section splitting ------------------------------------------------------


class TestSplitSections:
    def test_text_before_any_marker_is_required(self) -> None:
        """A posting that opens with a bare technology list is stating
        requirements; defaulting to preferred would under-weight every one."""
        spans = split_sections("We use Python and Docker daily.")
        assert spans == [(REQUIRED, "We use Python and Docker daily.")]

    def test_requirements_marker_is_required(self) -> None:
        spans = split_sections("Intro text\n\nRequirements:\n- Python\n")
        assert spans[0][0] == REQUIRED
        assert spans[-1][0] == REQUIRED
        assert "Python" in spans[-1][1]

    @pytest.mark.parametrize(
        "marker", ["Nice to have", "Bonus", "Preferred", "Good to have", "Desirable"]
    )
    def test_preferred_markers(self, marker: str) -> None:
        spans = split_sections(f"Requirements:\n- Python\n\n{marker}:\n- Kubernetes\n")
        assert spans[-1][0] == PREFERRED
        assert "Kubernetes" in spans[-1][1]

    @pytest.mark.parametrize(
        "marker", ["Requirements", "Must have", "Qualifications", "Essential"]
    )
    def test_required_markers(self, marker: str) -> None:
        spans = split_sections(f"Intro\n\n{marker}:\n- Python\n")
        assert spans[-1][0] == REQUIRED

    def test_longest_marker_wins(self) -> None:
        """"Preferred qualifications" must not be read as "qualifications" —
        that error promotes a nice-to-have into a hard requirement."""
        spans = split_sections("Preferred qualifications:\n- Kubernetes\n")
        assert spans[-1][0] == PREFERRED

    def test_a_marker_word_mid_sentence_does_not_split(self) -> None:
        text = "We have no formal requirements beyond curiosity and Python."
        assert len(split_sections(text)) == 1

    def test_bulleted_marker_is_recognised(self) -> None:
        spans = split_sections("Requirements:\n- Python\n\n* Nice to have:\n- Kafka\n")
        assert spans[-1][0] == PREFERRED

    def test_empty_input(self) -> None:
        assert split_sections("") == []
        assert split_sections(None) == []


# --- extraction -------------------------------------------------------------


class TestExtractSkills:
    def test_finds_skills_and_tags_them_by_section(self, index) -> None:
        jd = """
About the role
We build data services.

Requirements:
- Python
- PostgreSQL

Nice to have:
- Kubernetes
- Kafka
"""
        hits = extract_skills(jd, index)
        assert names(hits) == {"Python", "PostgreSQL", "Kubernetes", "Kafka"}
        assert level(hits, "Python") == REQUIRED
        assert level(hits, "PostgreSQL") == REQUIRED
        assert level(hits, "Kubernetes") == PREFERRED
        assert level(hits, "Kafka") == PREFERRED

    def test_required_wins_when_a_skill_is_in_both_sections(self, index) -> None:
        jd = "Requirements:\n- Python\n\nNice to have:\n- Python\n"
        hits = extract_skills(jd, index)
        assert level(hits, "Python") == REQUIRED

    def test_required_wins_regardless_of_order(self, index) -> None:
        jd = "Nice to have:\n- Python\n\nRequirements:\n- Python\n"
        assert level(extract_skills(jd, index), "Python") == REQUIRED

    def test_aliases_resolve_to_the_canonical_name(self, index) -> None:
        hits = extract_skills("Requirements:\n- k8s, postgres, py\n", index)
        assert names(hits) == {"Kubernetes", "PostgreSQL", "Python"}

    def test_case_insensitive(self, index) -> None:
        assert names(extract_skills("Requirements:\n- PYTHON, docker\n", index)) == {
            "Python",
            "Docker",
        }

    def test_longest_form_wins(self, index) -> None:
        """"machine learning" must beat "learning", or the vocabulary's more
        specific entries are unreachable."""
        hits = extract_skills("Requirements:\n- Machine Learning\n", index)
        assert names(hits) == {"Machine Learning"}
        assert "Learning" not in names(hits)

    def test_deduplicates_repeats(self, index) -> None:
        jd = "Requirements:\n- Python\n- Python again\n- More Python\n"
        assert len([h for h in extract_skills(jd, index) if h.name == "Python"]) == 1

    def test_word_boundaries(self, index) -> None:
        """Substring matching would find Python inside "Pythonic"."""
        assert names(extract_skills("Requirements:\n- Pythonic style\n", index)) == set()

    def test_punctuation_heavy_names(self, index) -> None:
        hits = extract_skills("Requirements:\n- C++, C#, .NET\n", index)
        assert names(hits) == {"C++", "C#", ".NET"}

    def test_cpp_does_not_register_as_c(self, index) -> None:
        """Longest-first plus the lookahead: "C++" must not also yield "C"."""
        assert names(extract_skills("Requirements:\n- C++\n", index)) == {"C++"}

    def test_empty_description(self, index) -> None:
        assert extract_skills("", index) == []
        assert extract_skills(None, index) == []

    def test_unknown_skill_yields_nothing(self, index) -> None:
        """The known cost of a dictionary: it cannot find what it does not know,
        and reports nothing rather than guessing."""
        assert extract_skills("Requirements:\n- Zorblang\n", index) == []


# --- the short-name guard ---------------------------------------------------


class TestShortNameGuard:
    def test_go_to_our_website_does_not_match_go(self, index) -> None:
        """Named in the spec. Start-of-string must not count as a list
        boundary, or every sentence opening with "Go" tags the job."""
        assert "Go" not in names(extract_skills("Go to our website to apply.", index))

    def test_r_and_d_team_does_not_match_r(self, index) -> None:
        """Named in the spec. "&" is not list punctuation."""
        assert "R" not in names(extract_skills("Join our R&D team today.", index))

    @pytest.mark.parametrize(
        "text",
        [
            "Requirements:\n- Python, Go, Rust\n",
            "Requirements:\n- Go, Python\n",
            "Requirements:\n- Python/Go\n",
            "Requirements:\n- Go (Golang)\n",
            "Requirements:\n- Go programming language\n",
        ],
    )
    def test_go_matches_in_a_technology_list(self, text: str, index) -> None:
        assert "Go" in names(extract_skills(text, index))

    @pytest.mark.parametrize(
        "text",
        [
            "We want you to go far in your career.",
            "Please go through the onboarding.",
            "This is a good place to grow.",
        ],
    )
    def test_go_does_not_match_in_prose(self, text: str, index) -> None:
        assert "Go" not in names(extract_skills(text, index))

    def test_r_matches_in_a_list(self, index) -> None:
        assert "R" in names(extract_skills("Requirements:\n- Python, R, SQL\n", index))

    @pytest.mark.parametrize(
        "text",
        [
            "You will be a R&D engineer.",
            "Reporting to the R and D lead.",
        ],
    )
    def test_r_does_not_match_in_prose(self, text: str, index) -> None:
        assert "R" not in names(extract_skills(text, index))

    def test_c_matches_in_a_list_but_not_in_prose(self, index) -> None:
        assert "C" in names(extract_skills("Requirements:\n- C, C++, Rust\n", index))
        assert "C" not in names(extract_skills("A grade C or above is fine.", index))

    def test_d_does_not_match_a_bare_letter_in_prose(self, index) -> None:
        assert "D" not in names(extract_skills("Option D is also available.", index))

    def test_long_names_are_unaffected_by_the_guard(self, index) -> None:
        """The guard applies only to the short list; Python in prose still
        counts, because there is no English word "Python"."""
        assert "Python" in names(extract_skills("We write a lot of Python here.", index))


class TestIndexBoundaryPattern:
    def test_boundary_rejects_a_leading_word_character(self) -> None:
        pattern = re.compile(_boundary("Go"), re.IGNORECASE)
        assert pattern.search("Go,") is not None
        assert pattern.search("Django") is None
        assert pattern.search("Going") is None

    def test_boundary_handles_plus_and_hash(self) -> None:
        assert re.compile(_boundary("C++")).search("C++,") is not None
        assert re.compile(_boundary("C")).search("C++") is None

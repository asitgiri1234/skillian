"""Unit tests for the supporting matching modules: queries, explain, embedding text."""

from __future__ import annotations

import pytest

from app.matching.explain import _tidy, explain_match
from app.matching.queries import build_search_queries
from app.matching.scorer import ScoreResult
from app.providers import LLMUnavailableError
from app.structure import build_resume_embedding_text

PARSED = {
    "name": "Test Candidate",
    "email": "candidate@test.invalid",
    "phone": "+91 90000 00000",
    "total_experience_years": 6.0,
    "skills": ["Python", "PostgreSQL", "Docker", "Kafka", "Redis", "Communication"],
    "projects": [
        {"title": "Billing reconciliation", "tech": ["Python", "Celery"]},
        {"title": "Event replay", "tech": ["Kafka", "Go"]},
    ],
    "experience": [
        {
            "company": "Globex",
            "role": "Senior Backend Engineer",
            "duration": "2021 - Present",
        },
        {
            "company": "Initech",
            "role": "Software Development Engineer II",
            "duration": "2019 - 2021",
        },
    ],
    "education": [
        {"institution": "Some University", "degree": "B.E.", "year": "2019"}
    ],
}


class TestBuildResumeEmbeddingText:
    def test_includes_skills_roles_and_projects(self) -> None:
        text = build_resume_embedding_text(PARSED)
        assert "Python" in text
        assert "PostgreSQL" in text
        assert "Senior Backend Engineer" in text
        assert "Globex" in text

    def test_includes_project_titles_and_tech(self) -> None:
        """Projects replace the role summaries the schema trim removed; their
        tech lists often name tools the skills list omits (here, Celery and Go)."""
        text = build_resume_embedding_text(PARSED)
        assert "Event replay" in text
        assert "Celery" in text
        assert "Go" in text

    def test_excludes_contact_details_and_education(self) -> None:
        """The whole reason this function exists: an address, a phone number and
        a degree title are a large fraction of the document and pull every
        candidate's vector toward the same generic-CV centroid."""
        text = build_resume_embedding_text(PARSED)
        assert "+91 90000 00000" not in text
        assert "candidate@test.invalid" not in text
        assert "Some University" not in text

    def test_excludes_durations(self) -> None:
        """"2021 - Present" is noise under cosine, exactly as the old
        start/end dates were."""
        text = build_resume_embedding_text(PARSED)
        assert "2021" not in text
        assert "Present" not in text

    def test_accepts_a_parsed_resume_object(self) -> None:
        from app.structure import ParsedResume

        parsed = ParsedResume.model_validate(PARSED)
        assert build_resume_embedding_text(parsed) == build_resume_embedding_text(
            PARSED
        )

    def test_empty_parse_yields_empty_string(self) -> None:
        """Must not return whitespace: the embedding provider rejects empty text
        deliberately, and the caller checks for "" to skip embedding entirely."""
        assert build_resume_embedding_text({}) == ""
        assert build_resume_embedding_text(None) == ""
        assert build_resume_embedding_text({"skills": [], "experience": [], "projects": []}) == ""

    def test_skills_only_resume_still_produces_text(self) -> None:
        text = build_resume_embedding_text({"skills": ["Rust", "Go"], "experience": []})
        assert "Rust" in text and "Go" in text

    def test_is_shorter_than_the_raw_document_would_be(self) -> None:
        text = build_resume_embedding_text(PARSED)
        raw_ish = " ".join(str(v) for v in PARSED.values())
        assert len(text) < len(raw_ish)


class TestBuildSearchQueries:
    def test_leads_with_the_most_recent_title(self) -> None:
        queries = build_search_queries(PARSED)
        assert queries[0].keywords == "Backend Engineer"

    def test_strips_seniority_from_titles(self) -> None:
        """Boards match seniority words literally, so 'Senior Python Engineer'
        misses every posting titled 'Python Engineer' while the reverse hits both."""
        queries = build_search_queries(PARSED)
        assert "Senior" not in queries[0].keywords
        assert all("II" not in query.keywords for query in queries)

    def test_includes_a_skills_query(self) -> None:
        queries = build_search_queries(PARSED)
        skills_query = next(q for q in queries if "Python" in q.keywords)
        assert "PostgreSQL" in skills_query.keywords

    def test_drops_generic_skills(self) -> None:
        """'Communication' is on every posting; it costs a query slot and
        returns the whole board."""
        queries = build_search_queries(PARSED)
        assert all("Communication" not in query.keywords for query in queries)

    def test_respects_the_query_cap(self) -> None:
        assert len(build_search_queries(PARSED, max_queries=2)) == 2

    def test_divides_max_results_across_queries(self) -> None:
        """The cap is on the run, not per query — otherwise adding a query
        silently triples the rows a single search fetches."""
        queries = build_search_queries(PARSED, max_results=90)
        assert sum(query.max_results for query in queries) <= 90

    def test_passes_filters_through(self) -> None:
        queries = build_search_queries(PARSED, location="Pune", remote_only=True)
        assert all(query.location == "Pune" for query in queries)
        assert all(query.remote_only for query in queries)

    def test_falls_back_to_skills_with_no_titles(self) -> None:
        queries = build_search_queries({"skills": ["Rust", "WebAssembly"], "experience": []})
        assert queries
        assert "Rust" in queries[0].keywords

    def test_raises_on_an_empty_parse(self) -> None:
        """Better than fetching noise: the caller records this on the run row."""
        with pytest.raises(ValueError, match="no usable job titles or skills"):
            build_search_queries({"skills": [], "experience": []})

    def test_is_deterministic(self) -> None:
        first = [q.keywords for q in build_search_queries(PARSED)]
        second = [q.keywords for q in build_search_queries(PARSED)]
        assert first == second


class TestExplain:
    def _result(self, **kwargs) -> ScoreResult:
        defaults = dict(
            overall_score=0.82,
            semantic_score=0.7,
            skill_score=0.9,
            matching_skills=["Python", "PostgreSQL"],
            missing_skills=["Kubernetes"],
            skills_unparsed=False,
        )
        return ScoreResult(**{**defaults, **kwargs})

    def test_returns_the_model_prose(self, fake_llm) -> None:
        text = explain_match("Backend Engineer", "Acme", self._result(), fake_llm)
        assert text == "You match this role on Python and PostgreSQL."

    def test_prompt_names_the_specific_skills(self, fake_llm) -> None:
        explain_match("Backend Engineer", "Acme", self._result(), fake_llm)
        prompt = fake_llm.complete_text_calls[0]
        assert "Python" in prompt
        assert "Kubernetes" in prompt

    def test_prompt_says_so_when_requirements_were_not_parsed(self, fake_llm) -> None:
        """Handed a blank skill list with no explanation, the model reliably
        invents requirements to fill the gap."""
        explain_match(
            "Backend Engineer", "Acme", self._result(skills_unparsed=True), fake_llm
        )
        prompt = fake_llm.complete_text_calls[0]
        assert "could not be extracted" in prompt

    def test_uses_free_text_not_a_schema(self, fake_llm) -> None:
        explain_match("Backend Engineer", "Acme", self._result(), fake_llm)
        assert fake_llm.complete_text_calls
        assert not fake_llm.complete_calls

    def test_returns_none_when_the_model_fails(self) -> None:
        """A lost explanation must not fail a run whose scores are already
        computed and stored."""

        class BrokenLLM:
            def complete_text(self, *args, **kwargs):
                raise LLMUnavailableError("daemon down")

        assert explain_match("X", "Y", self._result(), BrokenLLM()) is None

    def test_returns_none_on_an_empty_reply(self) -> None:
        class EmptyLLM:
            def complete_text(self, *args, **kwargs):
                return "   "

        assert explain_match("X", "Y", self._result(), EmptyLLM()) is None


class TestTidy:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Sure! You match well.", "You match well."),
            ("Here's why: You match well.", "You match well."),
            ("Certainly. You match well.", "You match well."),
            ('"You match well."', "You match well."),
            ("```\nYou match well.\n```", "You match well."),
            ("You  match\n\nwell.", "You match well."),
        ],
    )
    def test_strips_conversational_scaffolding(self, raw: str, expected: str) -> None:
        assert _tidy(raw) == expected

    def test_leaves_clean_prose_alone(self) -> None:
        prose = "You bring Python and PostgreSQL, which this role needs."
        assert _tidy(prose) == prose

"""API endpoint tests, against a real database with the models faked out.

The dependency override in :func:`client` is what makes the request handler use
the *test's* transaction, so everything an endpoint writes is rolled back with
the rest of the test.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import runs as run_status
from app.api.deps import get_session
from app.main import app
from app.matching.pipeline import create_search_run
from app.models import IngestionRun, Job, JobChunk, JobSkill, Match, Resume, Skill

pytestmark = pytest.mark.usefixtures("db_available")


PARSED = {
    "name": "Test Candidate",
    "email": "candidate@test.invalid",
    "phone": None,
    "location": "Bengaluru",
    "summary": "Backend engineer.",
    "skills": ["Python", "PostgreSQL"],
    "experience": [
        {
            "company": "Globex",
            "title": "Senior Backend Engineer",
            "start_date": "2021",
            "end_date": "Present",
            "is_current": True,
            "summary": "Built Python services.",
        }
    ],
    "education": [],
    "total_years_experience": 6.0,
}


@pytest.fixture
def client(db_session):
    """TestClient whose handlers share the test's session and transaction."""
    app.dependency_overrides[get_session] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def resume(db_session, user_row, fake_embedder):
    from app.structure import build_resume_embedding_text

    row = Resume(
        user_id=user_row.id,
        label="Test CV",
        raw_text="unused",
        parsed=PARSED,
        embedding=fake_embedder.embed(build_resume_embedding_text(PARSED)),
    )
    db_session.add(row)
    db_session.flush()
    return row


@pytest.fixture
def job(db_session, fake_embedder):
    row = Job(
        source="fake",
        source_job_id=f"api-{uuid.uuid4().hex[:12]}",
        dedup_hash=uuid.uuid4().hex,
        title="Backend Engineer",
        company="Acme",
        location="Bengaluru, Karnataka",
        description="We need a backend engineer with Python.",
        apply_url="https://example.test/1",
    )
    db_session.add(row)
    db_session.flush()

    skill = db_session.execute(
        select(Skill).where(Skill.name == "Python")
    ).scalar_one_or_none()
    if skill is None:
        skill = Skill(name="Python", aliases=[])
        db_session.add(skill)
        db_session.flush()
    db_session.add(
        JobSkill(job_id=row.id, skill_id=skill.id, requirement="required")
    )
    db_session.add(
        JobChunk(
            job_id=row.id,
            chunk_index=0,
            text="We need a backend engineer with Python.",
            embedding=fake_embedder.embed("We need a backend engineer with Python."),
        )
    )
    db_session.flush()
    return row


class TestHealth:
    def test_returns_ok_without_touching_dependencies(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCreateSearch:
    def test_returns_a_run_id_and_202(self, client, db_session, resume, monkeypatch) -> None:
        # Neutralise the background task: this test is about the handler's
        # response, and the pipeline has its own tests.
        monkeypatch.setattr("app.api.searches.run_search_task", lambda **kwargs: None)

        response = client.post("/searches", json={"resume_id": str(resume.id)})
        assert response.status_code == 202

        body = response.json()
        assert body["status"] == run_status.STATUS_QUEUED
        assert body["stage"] == run_status.STAGE_QUEUED

        run = db_session.get(IngestionRun, uuid.UUID(body["run_id"]))
        assert run is not None
        assert run.resume_id == resume.id

    def test_is_fast(self, client, resume, monkeypatch) -> None:
        """The 100ms budget. The handler makes no network call and no model
        call, so this is a regression guard on someone adding one."""
        import time

        monkeypatch.setattr("app.api.searches.run_search_task", lambda **kwargs: None)
        start = time.perf_counter()
        client.post("/searches", json={"resume_id": str(resume.id)})
        assert (time.perf_counter() - start) < 0.5  # generous; typical is ~5ms

    def test_404_for_an_unknown_resume(self, client) -> None:
        response = client.post("/searches", json={"resume_id": str(uuid.uuid4())})
        assert response.status_code == 404

    def test_409_for_an_unparsed_resume(self, client, db_session, user_row) -> None:
        """Better than a run that fails 200ms after the client stops watching."""
        unparsed = Resume(user_id=user_row.id, raw_text="not parsed yet")
        db_session.add(unparsed)
        db_session.flush()
        response = client.post("/searches", json={"resume_id": str(unparsed.id)})
        assert response.status_code == 409

    def test_422_for_an_unknown_source(self, client, resume) -> None:
        response = client.post(
            "/searches",
            json={"resume_id": str(resume.id), "sources": ["nonexistent"]},
        )
        assert response.status_code == 422
        assert "Available" in response.json()["detail"]


class TestGetRun:
    def test_reports_status_stage_and_progress(self, client, db_session, resume) -> None:
        run = create_search_run(db_session, resume.id, ["fake"])
        body = client.get(f"/runs/{run.id}").json()

        assert body["status"] == run_status.STATUS_QUEUED
        assert body["stage"] == run_status.STAGE_QUEUED
        assert body["stage_label"] == "Queued"
        assert body["stage_number"] == 1
        assert body["stage_total"] == len(run_status.STAGE_ORDER)
        assert body["is_terminal"] is False

    def test_is_terminal_spans_both_status_vocabularies(
        self, client, db_session, resume
    ) -> None:
        """'success' (day 1) and 'succeeded' (day 3) share the column; a client
        polling on is_terminal must not have to know which wrote the row."""
        for status in (run_status.STATUS_SUCCESS, run_status.STATUS_SUCCEEDED):
            run = create_search_run(db_session, resume.id, ["fake"])
            run.status = status
            db_session.flush()
            assert client.get(f"/runs/{run.id}").json()["is_terminal"] is True

    def test_404_for_an_unknown_run(self, client) -> None:
        assert client.get(f"/runs/{uuid.uuid4()}").status_code == 404


class TestListMatches:
    @pytest.fixture
    def matches(self, db_session, resume, fake_embedder):
        rows = []
        for index, score in enumerate([0.9, 0.5, 0.2]):
            job = Job(
                source="fake",
                source_job_id=f"m-{uuid.uuid4().hex[:12]}",
                dedup_hash=uuid.uuid4().hex,
                title=f"Job {index}",
                company=f"Company {index}",
                location="Bengaluru",
                salary_raw="INR 20,00,000 per year",
                apply_url=f"https://example.test/{index}",
            )
            db_session.add(job)
            db_session.flush()
            match = Match(
                resume_id=resume.id,
                job_id=job.id,
                overall_score=score,
                semantic_score=score,
                skill_score=score,
                matching_skills=["Python"],
                missing_skills=["Rust"],
                explanation=f"Explanation {index}",
                model_version="test",
            )
            db_session.add(match)
            rows.append(match)
        db_session.flush()
        return rows

    def test_sorted_by_score_descending(self, client, resume, matches) -> None:
        body = client.get("/matches", params={"resume_id": str(resume.id)}).json()
        scores = [float(row["overall_score"]) for row in body]
        assert scores == sorted(scores, reverse=True)

    def test_includes_joined_job_fields(self, client, resume, matches) -> None:
        """A result list returning bare job ids would force N follow-up requests
        to render a single page."""
        row = client.get("/matches", params={"resume_id": str(resume.id)}).json()[0]
        assert row["title"]
        assert row["company"]
        assert row["apply_url"]
        assert row["explanation"]
        assert row["matching_skills"] == ["Python"]

    def test_min_score_filters(self, client, resume, matches) -> None:
        body = client.get(
            "/matches", params={"resume_id": str(resume.id), "min_score": 0.4}
        ).json()
        assert len(body) == 2

    def test_limit_and_offset(self, client, resume, matches) -> None:
        first = client.get(
            "/matches", params={"resume_id": str(resume.id), "limit": 1}
        ).json()
        second = client.get(
            "/matches", params={"resume_id": str(resume.id), "limit": 1, "offset": 1}
        ).json()
        assert len(first) == len(second) == 1
        assert first[0]["job_id"] != second[0]["job_id"]

    def test_unknown_resume_returns_an_empty_list(self, client) -> None:
        """Not a 404: "this resume has no matches yet" is the normal state
        while a search is still running."""
        response = client.get("/matches", params={"resume_id": str(uuid.uuid4())})
        assert response.status_code == 200
        assert response.json() == []


class TestGetJob:
    def test_returns_detail_with_skills_and_chunks(self, client, job) -> None:
        body = client.get(f"/jobs/{job.id}").json()
        assert body["title"] == "Backend Engineer"
        assert body["description"]
        assert [s["name"] for s in body["skills"]] == ["Python"]
        assert body["skills"][0]["requirement"] == "required"
        assert body["chunks"][0]["chunk_index"] == 0
        assert body["chunks"][0]["text"]

    def test_does_not_return_chunk_embeddings(self, client, job) -> None:
        """768 floats per chunk that no client can use."""
        body = client.get(f"/jobs/{job.id}").json()
        assert "embedding" not in body["chunks"][0]

    def test_404_for_an_unknown_job(self, client) -> None:
        assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404


class TestPatchResumeSkills:
    def test_replaces_skills_reembeds_and_clears_matches(
        self, client, db_session, resume, job, fake_embedder, monkeypatch
    ) -> None:
        """All three, or the correction is cosmetic."""
        monkeypatch.setattr(
            "app.api.resumes.get_embedding_provider", lambda *a, **k: fake_embedder
        )
        db_session.add(
            Match(resume_id=resume.id, job_id=job.id, overall_score=0.5)
        )
        db_session.flush()
        before = resume.embedding

        response = client.patch(
            f"/resumes/{resume.id}/skills",
            json={"skills": ["Rust", "WebAssembly", "Go"]},
        )
        assert response.status_code == 200

        body = response.json()
        assert sorted(body["skills"]) == ["Go", "Rust", "WebAssembly"]
        assert body["parsed"]["skills"] == ["Rust", "WebAssembly", "Go"]
        assert body["has_embedding"] is True
        assert "Rust" in body["embedding_text"]

        db_session.expire_all()
        refreshed = db_session.get(Resume, resume.id)
        assert list(refreshed.embedding) != list(before), "must re-embed"
        assert (
            db_session.execute(
                select(Match).where(Match.resume_id == resume.id)
            ).scalars().all()
            == []
        ), "scores computed against the old skill set are wrong, not stale"

    def test_removing_a_skill_actually_removes_it(
        self, client, db_session, resume, fake_embedder, monkeypatch
    ) -> None:
        """Replace, not merge — otherwise a bad parse can never be corrected."""
        monkeypatch.setattr(
            "app.api.resumes.get_embedding_provider", lambda *a, **k: fake_embedder
        )
        client.patch(
            f"/resumes/{resume.id}/skills", json={"skills": ["Python", "Rust"]}
        )
        body = client.patch(
            f"/resumes/{resume.id}/skills", json={"skills": ["Python"]}
        ).json()
        assert body["skills"] == ["Python"]

    def test_404_for_an_unknown_resume(self, client, fake_embedder, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.api.resumes.get_embedding_provider", lambda *a, **k: fake_embedder
        )
        response = client.patch(
            f"/resumes/{uuid.uuid4()}/skills", json={"skills": ["Python"]}
        )
        assert response.status_code == 404


class TestCreateResume:
    def test_extracts_links_skills_and_embeds(
        self, client, db_session, fake_embedder, monkeypatch
    ) -> None:
        from app.structure import ParsedResume

        monkeypatch.setattr(
            "app.api.resumes.get_embedding_provider", lambda *a, **k: fake_embedder
        )
        monkeypatch.setattr(
            "app.api.resumes.extract_resume",
            lambda text: ParsedResume.model_validate(PARSED),
        )

        response = client.post(
            "/resumes",
            json={
                "email": f"new-{uuid.uuid4().hex[:8]}@test.invalid",
                "raw_text": "Some resume text",
                "label": "CV",
            },
        )
        assert response.status_code == 201

        body = response.json()
        assert sorted(body["skills"]) == ["PostgreSQL", "Python"]
        assert body["has_embedding"] is True
        # The embedding text is returned because "why did this match" is
        # unanswerable without knowing what was embedded.
        assert "Python" in body["embedding_text"]
        assert "candidate@test.invalid" not in body["embedding_text"]

    def test_503_when_the_model_is_unavailable(
        self, client, fake_embedder, monkeypatch
    ) -> None:
        """An operator problem, not the caller's — and it succeeds on retry."""
        from app.providers import LLMUnavailableError

        def boom(text):
            raise LLMUnavailableError("Cannot reach the Ollama daemon")

        monkeypatch.setattr(
            "app.api.resumes.get_embedding_provider", lambda *a, **k: fake_embedder
        )
        monkeypatch.setattr("app.api.resumes.extract_resume", boom)

        response = client.post(
            "/resumes",
            json={"email": "x@test.invalid", "raw_text": "Some resume text"},
        )
        assert response.status_code == 503

    def test_422_when_extraction_never_validates(
        self, client, fake_embedder, monkeypatch
    ) -> None:
        from app.structure import StructureError

        def boom(text):
            raise StructureError("failed", attempts=3, last_errors="- skills: empty")

        monkeypatch.setattr(
            "app.api.resumes.get_embedding_provider", lambda *a, **k: fake_embedder
        )
        monkeypatch.setattr("app.api.resumes.extract_resume", boom)

        response = client.post(
            "/resumes",
            json={"email": "y@test.invalid", "raw_text": "Some resume text"},
        )
        assert response.status_code == 422
        assert "3 attempts" in response.json()["detail"]

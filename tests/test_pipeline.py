"""End-to-end pipeline tests against a real Postgres, with fake source and models.

Real database, because the parts worth testing here are Postgres behaviours: the
``ON CONFLICT`` upserts, the ``xmax = 0`` new-row detection, pgvector columns and
JSONB round-trips. A SQLite stand-in would fake all four rather than verify them.

Fake source, LLM and embedder, because those are already covered by their own
contracts and a suite that called a 7b model would take an hour to run.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select

from app import runs as run_status
from app.matching.pipeline import (
    SearchFilters,
    create_search_run,
    run_search,
)
from app.models import IngestionRun, Job, JobChunk, JobSkill, Match, Resume, ResumeSkill, Skill
from app.sources.base import JobSource, NormalizedJob, SearchQuery

pytestmark = pytest.mark.usefixtures("db_available")


DESCRIPTION = """
About the role

We are looking for a backend engineer to join our platform team. You will design
and operate services that handle millions of requests a day, working closely with
product and infrastructure. This is a hands-on role with real ownership.

Requirements:
- Strong Python, with production experience
- PostgreSQL and relational data modelling
- Comfortable operating services you have built
- Excellent written communication

Nice to have:
- Kubernetes
- Experience with vector databases

Responsibilities:
You will own services end to end, from design review through to production
operation. That means writing the code, writing the tests, instrumenting what
you ship, and carrying a pager for it one week in six. You will review other
people's designs and be reviewed in turn. You will work with product managers to
turn a rough requirement into something shippable, and you will be expected to
push back when the requirement does not make sense. Much of the work is
incremental improvement of systems that already exist and already have users, so
comfort with legacy code and careful migrations matters more than greenfield
enthusiasm. We expect you to leave the codebase better than you found it and to
explain your reasoning in writing so that the next person can follow it.

Benefits:
We offer competitive compensation, health cover for you and your dependants,
a learning budget, and flexible working hours. We support remote work for up to
three days a week and provide a stipend towards a home office setup. There is a
generous parental leave policy, an annual conference budget, and a sabbatical
after five years. We are an equal opportunity employer and welcome applicants
from every background, and we make reasonable accommodations throughout the
hiring process for anyone who needs them.
"""

#: Distinct role titles, so that jobs differ by dedup_hash. Identical
#: company + title + city is *by design* one job — see the dedup test below —
#: so a fixture that wants N jobs has to vary one of the three.
TITLES = [
    "Backend Engineer", "Platform Engineer", "Data Engineer", "API Engineer",
    "Infrastructure Engineer", "Services Engineer", "Systems Engineer",
    "Reliability Engineer", "Integration Engineer", "Storage Engineer",
    "Search Engineer", "Payments Engineer", "Identity Engineer",
    "Streaming Engineer", "Batch Engineer", "Ingestion Engineer",
    "Analytics Engineer", "Tooling Engineer", "Developer Experience Engineer",
    "Security Engineer", "Networking Engineer", "Database Engineer",
    "Observability Engineer", "Workflow Engineer", "Scheduling Engineer",
]


class FakeSource(JobSource):
    """A job source with no network behind it.

    Records the run row's status and stage each time it is asked to fetch, which
    is how the transition test observes ``running`` from *inside* the run rather
    than inferring it.
    """

    name = "fake"

    def __init__(self, jobs: list[NormalizedJob], observer=None) -> None:
        self._jobs = jobs
        self._observer = observer
        self.queries: list[SearchQuery] = []

    def fetch(self, query: SearchQuery) -> list[NormalizedJob]:
        self.queries.append(query)
        if self._observer is not None:
            self._observer()
        return list(self._jobs)


class FailingSource(JobSource):
    name = "failing"

    def fetch(self, query: SearchQuery) -> list[NormalizedJob]:
        raise RuntimeError("board is down")


def _job(
    index: int, *, company: str = "Acme", title: str | None = None
) -> NormalizedJob:
    return NormalizedJob(
        source="fake",
        source_job_id=f"fake-{uuid.uuid4().hex[:12]}-{index}",
        title=title if title is not None else TITLES[index % len(TITLES)],
        company=company,
        location="Bengaluru, Karnataka",
        description=DESCRIPTION,
        apply_url=f"https://example.test/jobs/{index}",
        experience_min_years=Decimal("5.0"),
    )


PARSED = {
    "name": "Test Candidate",
    "email": "candidate@test.invalid",
    "phone": None,
    "total_experience_years": 6.0,
    "skills": ["Python", "PostgreSQL", "Docker"],
    "projects": [
        {"title": "Ledger reconciliation", "tech": ["Python", "PostgreSQL"]}
    ],
    "experience": [
        {
            "company": "Globex",
            "role": "Senior Backend Engineer",
            "duration": "2021 - Present",
        }
    ],
    "education": [],
}


@pytest.fixture
def resume_row(db_session, user_row, fake_embedder):
    from app.structure import build_resume_embedding_text

    resume = Resume(
        user_id=user_row.id,
        label="Test CV",
        raw_text="irrelevant, the pipeline reads `parsed`",
        parsed=PARSED,
        embedding=fake_embedder.embed(build_resume_embedding_text(PARSED)),
    )
    db_session.add(resume)
    db_session.flush()

    # Give the resume real canonical skills, so skill_component has something to
    # intersect with the ones the fake LLM reports for each job.
    for name in PARSED["skills"]:
        skill = db_session.execute(
            select(Skill).where(Skill.name == name)
        ).scalar_one_or_none()
        if skill is None:
            skill = Skill(name=name, aliases=[])
            db_session.add(skill)
            db_session.flush()
        db_session.add(
            ResumeSkill(resume_id=resume.id, skill_id=skill.id, source="test")
        )
    db_session.flush()
    return resume


class TestStatusTransitions:
    def test_queued_then_running_then_succeeded(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """The specified lifecycle, each state observed where it actually holds."""
        observed: list[tuple[str, str | None]] = []

        run = create_search_run(db_session, resume_row.id, ["fake"])
        # 1. queued — set before any work starts, so POST /searches can return a
        #    run_id the client can poll immediately.
        assert run.status == run_status.STATUS_QUEUED
        assert run.stage == run_status.STAGE_QUEUED

        def observe() -> None:
            db_session.expire_all()
            current = db_session.get(IngestionRun, run.id)
            observed.append((current.status, current.stage))

        source = FakeSource([_job(i) for i in range(3)], observer=observe)

        outcome = run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(location="Bengaluru"),
            sources=[source],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )

        # 2. running — visible from inside the run, on a separate read of the
        #    row. This is the whole point of committing each stage on its own.
        assert observed, "the source was never asked to fetch"
        assert all(status == run_status.STATUS_RUNNING for status, _ in observed)
        assert all(stage == run_status.STAGE_FETCHING for _, stage in observed)

        # 3. succeeded
        assert outcome.status == run_status.STATUS_SUCCEEDED
        db_session.expire_all()
        final = db_session.get(IngestionRun, run.id)
        assert final.status == run_status.STATUS_SUCCEEDED
        assert final.stage == run_status.STAGE_DONE
        assert final.finished_at is not None
        assert final.error is None
        assert final.jobs_found == 3

    def test_stage_advances_through_the_declared_order(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """Stages must be real names from STAGE_ORDER, or the progress bar a
        client builds from stage_progress() is nonsense."""
        run = create_search_run(db_session, resume_row.id, ["fake"])
        seen: list[str] = []

        def observe() -> None:
            db_session.expire_all()
            seen.append(db_session.get(IngestionRun, run.id).stage)

        run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource([_job(0)], observer=observe)],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        assert all(stage in run_status.STAGE_ORDER for stage in seen)
        db_session.expire_all()
        assert db_session.get(IngestionRun, run.id).stage == run_status.STAGE_DONE


class TestPipelineWork:
    def test_stores_jobs_skills_chunks_and_matches(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        run = create_search_run(db_session, resume_row.id, ["fake"])
        outcome = run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource([_job(i) for i in range(3)])],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )

        assert outcome.jobs_found == 3
        assert outcome.new_jobs == 3
        assert outcome.matches_written == 3
        assert outcome.chunks_written > 3, "the sample JD should chunk into several"

        matches = list(
            db_session.execute(
                select(Match).where(Match.resume_id == resume_row.id)
            ).scalars()
        )
        assert len(matches) == 3
        for match in matches:
            assert match.overall_score is not None
            assert 0 <= match.overall_score <= 1
            assert match.model_version is not None
            # The fake LLM reports Python + PostgreSQL (required) and Kubernetes
            # (preferred); the resume holds the first two.
            assert "Python" in (match.matching_skills or [])
            assert "Kubernetes" in (match.missing_skills or [])

    def test_chunks_are_indexed_from_zero_and_contiguous(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        run = create_search_run(db_session, resume_row.id, ["fake"])
        run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource([_job(0)])],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        job_id = db_session.execute(select(Job.id)).scalars().first()
        indexes = list(
            db_session.execute(
                select(JobChunk.chunk_index)
                .where(JobChunk.job_id == job_id)
                .order_by(JobChunk.chunk_index)
            ).scalars()
        )
        assert indexes == list(range(len(indexes)))

    def test_scoring_makes_no_model_calls(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """The pipeline's central performance claim, asserted rather than
        commented: one skill-extraction call per new job, one explanation call
        per top match, and nothing at all for scoring."""
        jobs = [_job(i) for i in range(5)]
        run = create_search_run(db_session, resume_row.id, ["fake"])
        run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource(jobs)],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        # Exactly one extraction per job — not one per job plus one per score.
        assert len(fake_llm.complete_calls) == 5
        assert len(fake_llm.complete_text_calls) == 5

    def test_explanations_are_capped_at_twenty(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """25 jobs in, 25 scored, 20 explained. The cap is what keeps a large
        search from spending half an hour in the local model."""
        jobs = [_job(i, company=f"Company {i}") for i in range(25)]
        run = create_search_run(db_session, resume_row.id, ["fake"])
        outcome = run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource(jobs)],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        assert outcome.matches_written == 25
        assert outcome.explanations_written == 20
        assert len(fake_llm.complete_text_calls) == 20

        explained = db_session.execute(
            select(Match)
            .where(Match.resume_id == resume_row.id, Match.explanation.isnot(None))
        ).scalars().all()
        assert len(explained) == 20

    def test_explanations_go_to_the_highest_scoring_matches(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        jobs = [_job(i, company=f"Company {i}") for i in range(25)]
        run = create_search_run(db_session, resume_row.id, ["fake"])
        run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource(jobs)],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        rows = db_session.execute(
            select(Match.overall_score, Match.explanation)
            .where(Match.resume_id == resume_row.id)
            .order_by(Match.overall_score.desc())
        ).all()
        # Every explained row outranks every unexplained one.
        explained_scores = [score for score, text in rows if text is not None]
        unexplained_scores = [score for score, text in rows if text is None]
        if unexplained_scores:
            assert min(explained_scores) >= max(unexplained_scores)

    def test_dedupes_on_dedup_hash_within_a_run(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """Same company + title + city from two postings is one row this run —
        and, more importantly, one local-model skill extraction instead of two."""
        duplicate_pair = [
            _job(0, company="Acme Pvt Ltd", title="Backend Engineer"),
            _job(1, company="Acme Private Limited", title="Backend Engineer"),
            _job(2, company="Globex", title="Data Engineer"),
        ]
        run = create_search_run(db_session, resume_row.id, ["fake"])
        outcome = run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource(duplicate_pair)],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        assert outcome.jobs_found == 2

    def test_rerunning_is_idempotent(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """The second run re-scores the same jobs without re-extracting or
        re-chunking them, and does not duplicate any row."""
        jobs = [_job(i) for i in range(3)]
        source = FakeSource(jobs)

        first_run = create_search_run(db_session, resume_row.id, ["fake"])
        run_search(
            run_id=first_run.id, resume_id=resume_row.id, filters=SearchFilters(),
            sources=[source], llm=fake_llm, embedder=fake_embedder,
            session_factory=session_factory,
        )
        # Scoped to this test's own postings throughout — the database may hold
        # rows from anywhere else.
        our_job_ids = list(
            db_session.execute(
                select(Job.id).where(
                    Job.source_job_id.in_([job.source_job_id for job in jobs])
                )
            ).scalars()
        )
        assert len(our_job_ids) == 3
        chunks_after_first = db_session.execute(
            select(JobChunk).where(JobChunk.job_id.in_(our_job_ids))
        ).scalars().all()
        extractions_after_first = len(fake_llm.complete_calls)

        second_run = create_search_run(db_session, resume_row.id, ["fake"])
        outcome = run_search(
            run_id=second_run.id, resume_id=resume_row.id, filters=SearchFilters(),
            sources=[FakeSource(jobs)], llm=fake_llm, embedder=fake_embedder,
            session_factory=session_factory,
        )

        assert outcome.jobs_found == 3
        assert outcome.new_jobs == 0, "the same source_job_ids must update, not insert"
        assert outcome.chunks_written == 0, "already-chunked jobs are skipped"
        assert len(fake_llm.complete_calls) == extractions_after_first, (
            "already-extracted jobs must not be re-read by the model"
        )
        assert len(
            db_session.execute(
                select(JobChunk).where(JobChunk.job_id.in_(our_job_ids))
            ).scalars().all()
        ) == len(chunks_after_first)
        assert (
            len(
                db_session.execute(
                    select(Match).where(Match.resume_id == resume_row.id)
                ).scalars().all()
            )
            == 3
        )


class TestFailureHandling:
    def test_a_failing_source_yields_partial_not_failed(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        run = create_search_run(db_session, resume_row.id, ["fake", "failing"])
        outcome = run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource([_job(0)]), FailingSource()],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        assert outcome.status == run_status.STATUS_PARTIAL
        assert "failing" in outcome.source_errors
        db_session.expire_all()
        final = db_session.get(IngestionRun, run.id)
        assert final.status == run_status.STATUS_PARTIAL
        assert "board is down" in (final.error or "")
        assert final.finished_at is not None

    def test_every_source_failing_yields_failed(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        run = create_search_run(db_session, resume_row.id, ["failing"])
        outcome = run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FailingSource()],
            llm=fake_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        assert outcome.status == run_status.STATUS_FAILED

    def test_a_missing_resume_records_the_failure_and_reraises(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        """Nothing is swallowed: the exception escapes *and* the row records it."""
        run = create_search_run(db_session, resume_row.id, ["fake"])
        with pytest.raises(LookupError):
            run_search(
                run_id=run.id,
                resume_id=uuid.uuid4(),
                filters=SearchFilters(),
                sources=[FakeSource([])],
                llm=fake_llm,
                embedder=fake_embedder,
                session_factory=session_factory,
            )
        db_session.expire_all()
        final = db_session.get(IngestionRun, run.id)
        assert final.status == run_status.STATUS_FAILED
        assert "LookupError" in (final.error or "")
        assert final.finished_at is not None

    def test_no_run_is_left_in_a_non_terminal_state(
        self, db_session, session_factory, resume_row, fake_llm, fake_embedder
    ):
        for sources in ([FakeSource([_job(0)])], [FailingSource()]):
            run = create_search_run(db_session, resume_row.id, ["x"])
            try:
                run_search(
                    run_id=run.id, resume_id=resume_row.id, filters=SearchFilters(),
                    sources=sources, llm=fake_llm, embedder=fake_embedder,
                    session_factory=session_factory,
                )
            except Exception:  # noqa: BLE001 - the point is the row, not the raise
                pass
            db_session.expire_all()
            assert run_status.is_terminal(db_session.get(IngestionRun, run.id).status)


class TestUnparsedRequirements:
    def test_a_job_with_no_extractable_skills_scores_semantically(
        self, db_session, session_factory, resume_row, fake_embedder
    ):
        """No job_skills rows -> skills_unparsed -> semantic-only, and the match
        still exists rather than being dropped or zeroed."""
        from tests.conftest import FakeLLMProvider

        silent_llm = FakeLLMProvider(skills_response={"skills": []})
        posting = _job(0)
        run = create_search_run(db_session, resume_row.id, ["fake"])
        run_search(
            run_id=run.id,
            resume_id=resume_row.id,
            filters=SearchFilters(),
            sources=[FakeSource([posting])],
            llm=silent_llm,
            embedder=fake_embedder,
            session_factory=session_factory,
        )
        # Scoped to this test's own posting. A bare `select(Job.id).first()`
        # would pick up whatever else happens to be in the database.
        job_id = db_session.execute(
            select(Job.id).where(Job.source_job_id == posting.source_job_id)
        ).scalar_one()
        assert (
            db_session.execute(
                select(JobSkill).where(JobSkill.job_id == job_id)
            ).scalars().all()
            == []
        )
        match = db_session.execute(
            select(Match).where(Match.resume_id == resume_row.id)
        ).scalar_one()
        assert match.overall_score is not None
        assert match.skill_score == 0
        assert match.matching_skills == []

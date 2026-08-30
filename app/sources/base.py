"""The job-source contract.

The whole point of this module: adding a second provider means writing one new
file in ``app/sources/`` that subclasses :class:`JobSource`. Nothing in
``ingest.py``, ``models.py`` or the CLI needs to change, because they only ever
speak in terms of :class:`SearchQuery` and :class:`NormalizedJob`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.normalize import compute_dedup_hash, extract_city


class SearchQuery(BaseModel):
    """What the caller is looking for, in provider-neutral terms.

    Frozen so a query can be safely shared across several sources without one of
    them mutating it for the others.
    """

    model_config = ConfigDict(frozen=True)

    keywords: str = Field(min_length=1, description='Free text, e.g. "python backend engineer"')
    location: str | None = Field(default=None, description='Free text, e.g. "Bengaluru"')
    remote_only: bool = False
    # A ceiling on rows, not on pages: sources page until they hit it. Keeps a
    # broad query from burning an entire API quota in one run.
    max_results: int = Field(default=100, ge=1, le=1000)


class NormalizedJob(BaseModel):
    """One posting, in the shape of the ``jobs`` table.

    Sources are responsible for producing this and nothing else; they never touch
    the ORM or the database. That keeps a source unit-testable against a recorded
    HTTP response with no Postgres running.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    source_job_id: str
    title: str
    company: str | None = None
    location: str | None = None
    is_remote: bool = False
    description: str | None = None
    apply_url: str | None = None

    salary_raw: str | None = None
    salary_min: Decimal | None = None
    salary_max: Decimal | None = None
    salary_currency: str | None = None
    salary_period: str | None = None

    experience_raw: str | None = None
    experience_min_years: Decimal | None = None

    posted_date: date | None = None

    @property
    def dedup_hash(self) -> str:
        """Computed here, not by the source, so every provider agrees on it."""
        return compute_dedup_hash(self.company, self.title, extract_city(self.location))


class JobSource(ABC):
    """Abstract base for a job provider."""

    #: Stable short identifier, stored in ``jobs.source``. Part of the row's
    #: identity via UNIQUE(source, source_job_id), so it must never be renamed
    #: casually — doing so would re-insert every job as new.
    name: str

    @abstractmethod
    def fetch(self, query: SearchQuery) -> list[NormalizedJob]:
        """Return postings matching ``query``.

        Implementations own their own pagination, retries and field parsing, and
        should return the postings they *did* manage to parse rather than failing
        the whole batch over one malformed record. Raise only when the run cannot
        continue at all (bad credentials, exhausted retries).
        """
        raise NotImplementedError

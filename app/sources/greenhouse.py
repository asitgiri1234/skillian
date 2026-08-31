"""Greenhouse job boards. Public JSON, no key, **full descriptions**.

Why this exists: Adzuna truncates every description to 500 characters, and that
truncation was measured to break matching outright — six of the test resume's
skills (Kubernetes, Kafka, Terraform, Celery, Linux, pytest) matched zero jobs,
and all 246 extracted skills came back ``required`` with zero ``preferred``,
because requirements sections are cut off before they appear. The skill
component was scoring on each posting's opening paragraph.

Greenhouse returns the whole posting.

Shape difference from Adzuna, and the reason ``fetch`` ignores the query:
Greenhouse is **per company, not a search endpoint**. There is no ``what=`` or
``where=``. This source returns a company's entire board and lets dedup and
scoring filter, which is the right division of labour — the scorer already
ranks, and a keyword pre-filter here would only discard postings before they
could be scored.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.sources.base import JobSource, NormalizedJob, SearchQuery
from app.sources.company_boards import fetch_boards_concurrently, interleave, GREENHOUSE_COMPANIES

logger = logging.getLogger(__name__)

BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

# Greenhouse `content` is HTML *entity-encoded twice over*: the field contains
# escaped markup, so it arrives as "&lt;p&gt;We are&lt;/p&gt;". Unescape first,
# then strip tags — doing it the other way round leaves visible tags behind.
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKS_RE = re.compile(r"\n{3,}")

# Tags that imply a line break once the markup is gone; without this the whole
# posting collapses to one paragraph and the section markers in jd_skills
# (which anchor to line starts) never fire.
_BLOCK_RE = re.compile(
    r"</(?:p|div|li|ul|ol|h[1-6]|tr|table|section)>|<br\s*/?>|<li[^>]*>",
    re.IGNORECASE,
)


def html_to_text(raw: str | None) -> str:
    """Decode entities and strip tags, preserving paragraph and list breaks."""
    if not raw:
        return ""
    text = html.unescape(raw)
    # A second pass: some boards double-encode, and one unescape leaves &amp;.
    if "&lt;" in text or "&amp;" in text:
        text = html.unescape(text)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS_RE.sub("\n\n", text).strip()


class GreenhouseSource(JobSource):
    """Every open posting across the configured Greenhouse boards."""

    name = "greenhouse"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        companies: tuple[str, ...] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._companies = companies if companies is not None else GREENHOUSE_COMPANIES
        self._client = client
        self._owns_client = client is None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._settings.http_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "skillian/0.3 (+job matcher)"},
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def fetch_company(self, company: str) -> list[NormalizedJob]:
        """One board. Raises nothing: a dead slug must not sink the batch."""
        url = f"{BASE_URL}/{company}/jobs"
        try:
            response = self._http().get(url, params={"content": "true"})
        except httpx.HTTPError as exc:
            logger.warning("greenhouse %s: %s", company, exc)
            return []

        if response.status_code == 404:
            # A board that has been renamed or taken down. Logged, not raised —
            # the slug list is checked in and will drift.
            logger.warning("greenhouse %s: 404 (board not found)", company)
            return []
        if response.status_code != 200:
            logger.warning("greenhouse %s: HTTP %s", company, response.status_code)
            return []

        try:
            payload = response.json()
        except ValueError:
            logger.warning("greenhouse %s: non-JSON response", company)
            return []

        jobs: list[NormalizedJob] = []
        for record in payload.get("jobs") or []:
            parsed = self._parse(company, record)
            if parsed is not None:
                jobs.append(parsed)
        return jobs

    def _parse(self, company: str, record: dict[str, Any]) -> NormalizedJob | None:
        """One posting. Returns None rather than raising on a malformed record."""
        try:
            job_id = record.get("id")
            title = (record.get("title") or "").strip()
            if not job_id or not title:
                return None

            location = (record.get("location") or {}).get("name")
            description = html_to_text(record.get("content"))

            posted = None
            raw_date = record.get("updated_at") or record.get("first_published")
            if raw_date:
                try:
                    posted = datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    posted = None

            haystack = f"{title} {location or ''} {description[:2000]}".lower()
            return NormalizedJob(
                source=self.name,
                # Namespaced by company: Greenhouse ids are unique per board,
                # not globally, so a bare id would collide across companies and
                # the (source, source_job_id) unique constraint would silently
                # merge two different jobs.
                source_job_id=f"{company}:{job_id}",
                title=title,
                # The board slug is the only company identifier Greenhouse
                # returns; it is lowercase and unpunctuated, which is exactly
                # what normalize_company would produce anyway.
                company=company,
                location=location,
                is_remote="remote" in haystack,
                description=description or None,
                apply_url=record.get("absolute_url"),
                posted_date=posted,
            )
        except Exception:  # noqa: BLE001 - one bad record must not lose the board
            logger.exception("greenhouse %s: unparseable record", company)
            return None

    def fetch(self, query: SearchQuery) -> list[NormalizedJob]:
        """Every posting on every configured board, capped at max_results.

        ``query`` keywords are deliberately ignored — this is a board dump, not
        a search (see the module docstring). ``remote_only`` and ``max_results``
        are honoured, because both are caps rather than search terms.

        Boards are fetched concurrently: {N} sequential HTTPS round trips at
        ~1s each is a minute of waiting for work that is entirely I/O-bound.
        """
        by_company = fetch_boards_concurrently(self._companies, self.fetch_company)
        if query.remote_only:
            by_company = [(c, [j for j in jobs if j.is_remote]) for c, jobs in by_company]

        total = sum(len(jobs) for _, jobs in by_company)
        jobs = interleave(by_company, query.max_results)
        logger.info(
            "%s: %s job(s) from %s board(s), returning %s after the max_results cap",
            self.name, total, len(self._companies), len(jobs),
        )
        return jobs

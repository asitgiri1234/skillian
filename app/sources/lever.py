"""Lever job boards. Public JSON, no key, full descriptions.

Same rationale and same shape as :mod:`app.sources.greenhouse` — per company,
not a search endpoint, so ``fetch`` returns the whole board and lets dedup and
scoring filter.

Lever differs from Greenhouse in one useful way: alongside the HTML
``description`` it returns ``lists``, an already-structured array of
``{text, content}`` sections — "Requirements", "Nice to have", "What you'll do".
Those headings are exactly what ``jd_skills.split_sections`` looks for, so they
are rendered back as headed blocks rather than flattened, which is what lets the
required/preferred split actually fire.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.sources.base import JobSource, NormalizedJob, SearchQuery
from app.sources.company_boards import fetch_boards_concurrently, interleave, LEVER_COMPANIES
from app.sources.greenhouse import html_to_text

logger = logging.getLogger(__name__)

BASE_URL = "https://api.lever.co/v0/postings"


class LeverSource(JobSource):
    """Every open posting across the configured Lever boards."""

    name = "lever"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        companies: tuple[str, ...] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._companies = companies if companies is not None else LEVER_COMPANIES
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
        url = f"{BASE_URL}/{company}"
        try:
            response = self._http().get(url, params={"mode": "json"})
        except httpx.HTTPError as exc:
            logger.warning("lever %s: %s", company, exc)
            return []

        if response.status_code == 404:
            logger.warning("lever %s: 404 (board not found)", company)
            return []
        if response.status_code != 200:
            logger.warning("lever %s: HTTP %s", company, response.status_code)
            return []

        try:
            payload = response.json()
        except ValueError:
            logger.warning("lever %s: non-JSON response", company)
            return []
        if not isinstance(payload, list):
            logger.warning("lever %s: expected a list, got %s", company, type(payload))
            return []

        jobs: list[NormalizedJob] = []
        for record in payload:
            parsed = self._parse(company, record)
            if parsed is not None:
                jobs.append(parsed)
        return jobs

    def _describe(self, record: dict[str, Any]) -> str:
        """Rebuild the posting as headed text.

        The ``lists`` array is the valuable part: Lever already knows which
        block is "Requirements" and which is "Nice to have". Emitting each
        heading on its own line preserves that structure for
        ``jd_skills.split_sections``, which anchors its markers to line starts.
        Flattening would throw away the one source that hands us the section
        split for free.
        """
        parts: list[str] = []
        opening = html_to_text(record.get("description"))
        if opening:
            parts.append(opening)

        for block in record.get("lists") or []:
            if not isinstance(block, dict):
                continue
            heading = (block.get("text") or "").strip()
            body = html_to_text(block.get("content"))
            if heading and body:
                parts.append(f"{heading}:\n{body}")
            elif body:
                parts.append(body)

        closing = html_to_text(record.get("additional"))
        if closing:
            parts.append(closing)
        return "\n\n".join(parts).strip()

    def _parse(self, company: str, record: dict[str, Any]) -> NormalizedJob | None:
        try:
            job_id = record.get("id")
            title = (record.get("text") or "").strip()
            if not job_id or not title:
                return None

            categories = record.get("categories") or {}
            location = categories.get("location")
            description = self._describe(record)

            posted = None
            created = record.get("createdAt")
            if isinstance(created, (int, float)):
                # Lever sends epoch milliseconds.
                posted = datetime.fromtimestamp(created / 1000, timezone.utc).date()

            workplace = (record.get("workplaceType") or "").lower()
            haystack = f"{title} {location or ''} {workplace}".lower()
            return NormalizedJob(
                source=self.name,
                source_job_id=f"{company}:{job_id}",
                company=company,
                title=title,
                location=location,
                is_remote=workplace == "remote" or "remote" in haystack,
                description=description or None,
                apply_url=record.get("hostedUrl") or record.get("applyUrl"),
                posted_date=posted,
            )
        except Exception:  # noqa: BLE001 - one bad record must not lose the board
            logger.exception("lever %s: unparseable record", company)
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

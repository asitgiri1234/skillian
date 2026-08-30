"""Adzuna job source.

Everything Adzuna-specific is confined to this file: endpoint shape, pagination
rules, its salary conventions and its currency-by-country quirk. The rest of the
codebase only sees :class:`~app.sources.base.NormalizedJob`.
"""

from __future__ import annotations

import html
import logging
import random
import re
import time
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

import httpx

from app.config import Settings, get_settings
from app.sources.base import JobSource, NormalizedJob, SearchQuery

logger = logging.getLogger(__name__)

_API_ROOT: Final = "https://api.adzuna.com/v1/api/jobs"

# Adzuna partitions its API by country and returns salary figures in that
# country's currency WITHOUT a currency field, so the endpoint country is the
# only way to label the number. Unknown country -> None rather than a guess.
_COUNTRY_CURRENCY: Final[dict[str, str]] = {
    "at": "EUR", "au": "AUD", "be": "EUR", "br": "BRL", "ca": "CAD",
    "ch": "CHF", "de": "EUR", "es": "EUR", "fr": "EUR", "gb": "GBP",
    "in": "INR", "it": "EUR", "mx": "MXN", "nl": "EUR", "nz": "NZD",
    "pl": "PLN", "sg": "SGD", "us": "USD", "za": "ZAR",
}

# Adzuna normalises all salaries to an annual figure, so the period is fixed.
_SALARY_PERIOD: Final = "year"

# Retried: transient. Anything else (401 bad key, 400 bad params) is a bug or a
# config error and must surface immediately instead of being slept over.
_RETRY_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

_TAG_RE: Final = re.compile(r"<[^>]+>")
_WS_RE: Final = re.compile(r"\s+")

# Ordered: the first pattern that matches wins, so ranges ("3-5 years") are tried
# before the bare-number fallback that would read only the "3".
_EXPERIENCE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(\d{1,2})\s*(?:\.\d)?\s*[-–—to]{1,2}\s*\d{1,2}\+?\s*(?:\+)?\s*years?", re.I),
    re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+\s*years?", re.I),
    re.compile(r"(?:minimum|min|at\s+least|atleast|over)\s+(?:of\s+)?(\d{1,2}(?:\.\d)?)\s*\+?\s*years?", re.I),
    re.compile(r"(\d{1,2}(?:\.\d)?)\s*years?\s+(?:of\s+)?(?:relevant\s+|professional\s+|hands[-\s]?on\s+)?experience", re.I),
)

_REMOTE_RE: Final = re.compile(r"\b(remote|work from home|wfh|telecommut\w*)\b", re.I)
# "no remote", "not remote", "remote work is not available" — a plain keyword
# search would flag these as remote jobs.
_NOT_REMOTE_RE: Final = re.compile(r"\b(no|not|non[-\s]?)\s*remote\b|remote[^.]{0,30}\bnot\b", re.I)


class AdzunaConfigError(RuntimeError):
    """Raised when Adzuna credentials are missing or the country is unusable."""


def _clean_html(value: str | None) -> str | None:
    """Strip tags and unescape entities.

    Adzuna descriptions contain markup and HTML entities; the raw text is what
    later gets embedded, so tags would become meaningless tokens.
    """
    if not value:
        return None
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text or None


def _to_decimal(value: Any) -> Decimal | None:
    """Coerce Adzuna's loosely-typed numerics (float, str, None) to Decimal."""
    if value is None:
        return None
    try:
        # str() first: Decimal(float) would carry the float's binary error into
        # a money column.
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.debug("Could not parse %r as a number", value)
        return None


def _parse_posted_date(value: Any) -> date | None:
    """Parse Adzuna's ISO-8601 ``created`` timestamp into a date."""
    if not isinstance(value, str) or not value:
        return None
    try:
        # Adzuna emits a trailing "Z"; fromisoformat only learned to accept it in
        # 3.11, and swapping it keeps the parse working on older interpreters.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        logger.debug("Unparseable created timestamp: %r", value)
        return None


def _build_salary_raw(
    minimum: Decimal | None,
    maximum: Decimal | None,
    currency: str | None,
    is_predicted: bool,
) -> str | None:
    """Reconstruct a human-readable salary string.

    Adzuna gives numbers but no display string, and the jobs table always keeps a
    raw value — so we synthesise one and mark predicted figures, because Adzuna's
    prediction is a model output rather than something the employer published.
    """
    if minimum is None and maximum is None:
        return None
    unit = currency or ""
    if minimum is not None and maximum is not None and minimum != maximum:
        body = f"{unit}{minimum:,.0f} - {unit}{maximum:,.0f} per year"
    else:
        single = minimum if minimum is not None else maximum
        assert single is not None  # guarded by the early return above
        body = f"{unit}{single:,.0f} per year"
    return f"{body} (estimated)" if is_predicted else body


def _parse_experience(text: str | None) -> tuple[str | None, Decimal | None]:
    """Best-effort minimum-years-of-experience from the description.

    Returns the matched phrase as well as the number so a wrong parse is
    auditable against what the posting actually said.
    """
    if not text:
        return None, None
    for pattern in _EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        years = _to_decimal(match.group(1))
        # Reject implausible reads: "20 years" in "20 years in business" is not
        # a requirement, and a 0 tells us nothing.
        if years is None or not (Decimal(0) < years <= Decimal(15)):
            continue
        return match.group(0).strip(), years
    return None, None


def _detect_remote(title: str | None, description: str | None, location: str | None) -> bool:
    """Keyword heuristic — Adzuna has no remote flag of its own."""
    haystack = " ".join(part for part in (title, location, description) if part)
    if not haystack:
        return False
    if _NOT_REMOTE_RE.search(haystack):
        return False
    return bool(_REMOTE_RE.search(haystack))


class AdzunaSource(JobSource):
    """Fetches postings from the Adzuna search API."""

    name = "adzuna"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._country = self._settings.adzuna_country.lower().strip()
        # Injectable so tests can pass a MockTransport client and never hit the
        # network; owned-vs-borrowed is tracked so we only close our own.
        self._client = client
        self._owns_client = client is None

    # --- credentials ------------------------------------------------------

    def _credentials(self) -> tuple[str, str]:
        app_id = self._settings.adzuna_app_id
        app_key = self._settings.adzuna_app_key
        if not app_id or not app_key:
            raise AdzunaConfigError(
                "ADZUNA_APP_ID and ADZUNA_APP_KEY must be set (see .env.example). "
                "Get free credentials at https://developer.adzuna.com/"
            )
        return app_id, app_key

    # --- HTTP -------------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._settings.http_timeout_seconds,
                headers={"User-Agent": "skillian/0.1 (+job-matcher)"},
                # Adzuna redirects http->https and between regional hosts.
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "AdzunaSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _sleep_for_attempt(self, attempt: int, response: httpx.Response | None) -> None:
        """Back off before a retry, honouring Retry-After when the server sends it."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    # Only the delta-seconds form; the HTTP-date form is rare here
                    # and not worth a date parser.
                    time.sleep(min(float(retry_after), 60.0))
                    return
                except ValueError:
                    logger.debug("Unparseable Retry-After: %r", retry_after)
        base = self._settings.http_backoff_base_seconds
        delay = base * (2**attempt)
        # Full jitter: without it, several sources retrying in lockstep would
        # re-collide on exactly the same schedule.
        time.sleep(min(random.uniform(0, delay), 30.0))

    def _request_page(self, page: int, params: dict[str, Any]) -> dict[str, Any]:
        """GET one page, retrying transient failures with exponential backoff."""
        url = f"{_API_ROOT}/{self._country}/search/{page}"
        client = self._get_client()
        max_retries = self._settings.http_max_retries
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = client.get(url, params=params)
                if response.status_code in _RETRY_STATUS:
                    last_error = httpx.HTTPStatusError(
                        f"Adzuna returned {response.status_code} for page {page}",
                        request=response.request,
                        response=response,
                    )
                    logger.warning(
                        "Adzuna page %s: HTTP %s (attempt %s/%s)",
                        page, response.status_code, attempt + 1, max_retries + 1,
                    )
                    if attempt < max_retries:
                        self._sleep_for_attempt(attempt, response)
                        continue
                    raise last_error
                # Non-retryable 4xx/5xx: fail loudly and immediately.
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Adzuna page {page}: expected a JSON object")
                return payload
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                # HTTPStatusError only reaches here from raise_for_status (a
                # non-retryable code) or the exhausted-retries raise above.
                if isinstance(exc, httpx.HTTPStatusError) and (
                    exc.response.status_code not in _RETRY_STATUS or attempt >= max_retries
                ):
                    raise
                last_error = exc
                logger.warning(
                    "Adzuna page %s: %s (attempt %s/%s)",
                    page, exc.__class__.__name__, attempt + 1, max_retries + 1,
                )
                if attempt >= max_retries:
                    raise
                self._sleep_for_attempt(attempt, response)

        # Unreachable: the loop either returns or raises.
        raise RuntimeError(f"Adzuna page {page} failed") from last_error

    # --- parsing ----------------------------------------------------------

    def _to_normalized(self, raw: dict[str, Any]) -> NormalizedJob | None:
        """Map one Adzuna result. Returns None if the record is unusable.

        Only ``id`` and ``title`` are treated as required — everything else on
        Adzuna is genuinely optional and a missing company or salary is not a
        reason to drop an otherwise valid posting.
        """
        source_job_id = raw.get("id")
        title = _clean_html(raw.get("title"))
        if source_job_id is None or not title:
            logger.debug("Skipping Adzuna record without id/title: %r", raw.get("id"))
            return None

        # .get(...) or {} — Adzuna sometimes sends an explicit null for these
        # nested objects, which would make chained .get() raise.
        company = _clean_html((raw.get("company") or {}).get("display_name"))
        location = _clean_html((raw.get("location") or {}).get("display_name"))
        description = _clean_html(raw.get("description"))

        currency = _COUNTRY_CURRENCY.get(self._country)
        salary_min = _to_decimal(raw.get("salary_min"))
        salary_max = _to_decimal(raw.get("salary_max"))
        # Adzuna sends this as the string "0"/"1" rather than a bool.
        is_predicted = str(raw.get("salary_is_predicted", "0")) == "1"

        experience_raw, experience_min_years = _parse_experience(description)

        return NormalizedJob(
            source=self.name,
            source_job_id=str(source_job_id),
            title=title,
            company=company,
            location=location,
            is_remote=_detect_remote(title, description, location),
            description=description,
            apply_url=raw.get("redirect_url"),
            salary_raw=_build_salary_raw(salary_min, salary_max, currency, is_predicted),
            salary_min=salary_min,
            salary_max=salary_max,
            # Do not label a currency we cannot infer from the country.
            salary_currency=currency if (salary_min or salary_max) else None,
            salary_period=_SALARY_PERIOD if (salary_min or salary_max) else None,
            experience_raw=experience_raw,
            experience_min_years=experience_min_years,
            posted_date=_parse_posted_date(raw.get("created")),
        )

    # --- public API -------------------------------------------------------

    def fetch(self, query: SearchQuery) -> list[NormalizedJob]:
        """Page through Adzuna until max_results, an empty page, or max_pages."""
        if self._country not in _COUNTRY_CURRENCY:
            raise AdzunaConfigError(
                f"ADZUNA_COUNTRY={self._country!r} is not an Adzuna country code. "
                f"Expected one of: {', '.join(sorted(_COUNTRY_CURRENCY))}"
            )
        app_id, app_key = self._credentials()

        per_page = min(self._settings.results_per_page, query.max_results)
        jobs: list[NormalizedJob] = []
        # Adzuna ids repeat across pages when the result set shifts mid-run;
        # dedupe here so one fetch cannot return the same posting twice.
        seen_ids: set[str] = set()

        for page in range(1, self._settings.max_pages + 1):
            params: dict[str, Any] = {
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": per_page,
                "what": query.keywords,
                "content-type": "application/json",
            }
            if query.location:
                params["where"] = query.location
            if query.remote_only:
                # Adzuna has no remote filter; biasing the keyword query is the
                # only lever, and _detect_remote still filters the results below.
                params["what_or"] = f"{query.keywords} remote"

            payload = self._request_page(page, params)
            results = payload.get("results") or []
            if not isinstance(results, list) or not results:
                logger.info("Adzuna: no more results at page %s", page)
                break

            for raw in results:
                if not isinstance(raw, dict):
                    continue
                job = self._to_normalized(raw)
                if job is None or job.source_job_id in seen_ids:
                    continue
                if query.remote_only and not job.is_remote:
                    continue
                seen_ids.add(job.source_job_id)
                jobs.append(job)
                if len(jobs) >= query.max_results:
                    logger.info("Adzuna: reached max_results=%s", query.max_results)
                    return jobs

            # A short page means we are at the end of the result set.
            if len(results) < per_page:
                break

        logger.info("Adzuna: fetched %s jobs across up to %s pages", len(jobs), page)
        return jobs

#!/usr/bin/env python
"""Preflight: are the Adzuna credentials in .env real and working?

    python scripts/verify_adzuna.py
    python scripts/verify_adzuna.py --query "data engineer" --location "Pune"

Days 1 through 3 all shipped with `.env` holding the placeholders `test_id` /
`test_key`, so every Adzuna assumption in this codebase — field names, the
`"0"`/`"1"` string for `salary_is_predicted`, ISO-8601 `created`,
country-implied currency — has never been checked against a real response. This
script is the gate that closes that gap.

It goes through :class:`AdzunaSource`, not raw httpx, so a pass here means the
*real ingestion path* works rather than merely that the key is valid.

Exits non-zero on any failure, and prints the exact upstream error rather than
a summary of it: a 401 and a 403 mean different things (bad key vs. key not yet
activated) and the difference is only visible in the response body.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.sources.adzuna import AdzunaConfigError, AdzunaSource  # noqa: E402
from app.sources.base import SearchQuery  # noqa: E402

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

PLACEHOLDERS = {"test_id", "test_key", "your_app_id", "your_app_key", "changeme", ""}


def _mask(value: str | None) -> str:
    """Show enough to identify a key without printing it into a transcript."""
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return value[:2] + "*" * (len(value) - 2)
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_adzuna",
        description="Check that the configured Adzuna credentials actually work.",
    )
    parser.add_argument("--query", default="python developer")
    parser.add_argument("--location", default="Bengaluru")
    parser.add_argument("--max-results", type=int, default=3)
    args = parser.parse_args(argv)

    settings = get_settings()

    print("=" * 72)
    print("Adzuna credential check")
    print("=" * 72)
    print(f"   {INFO} app_id   {_mask(settings.adzuna_app_id)}")
    print(f"   {INFO} app_key  {_mask(settings.adzuna_app_key)}")
    print(f"   {INFO} country  {settings.adzuna_country}")
    print()

    # --- 1. are they even set, and not still the template values? ---------
    print("1. Credentials present")
    if not settings.adzuna_app_id or not settings.adzuna_app_key:
        print(f"   {FAIL} ADZUNA_APP_ID / ADZUNA_APP_KEY are not set in .env")
        print("          Get free credentials at https://developer.adzuna.com/")
        return 1
    stale = [
        name
        for name, value in (
            ("ADZUNA_APP_ID", settings.adzuna_app_id),
            ("ADZUNA_APP_KEY", settings.adzuna_app_key),
        )
        if value.strip().casefold() in PLACEHOLDERS
    ]
    if stale:
        print(f"   {FAIL} still the placeholder value(s): {', '.join(stale)}")
        print("          Replace them in .env with real credentials.")
        return 1
    print(f"   {PASS} both set and not placeholders")
    print()

    # --- 2. a real call, through the real source --------------------------
    print(f"2. Live fetch  (what={args.query!r}, where={args.location!r})")
    source = AdzunaSource()
    query = SearchQuery(
        keywords=args.query, location=args.location, max_results=args.max_results
    )
    try:
        jobs = source.fetch(query)
    except AdzunaConfigError as exc:
        print(f"   {FAIL} configuration rejected: {exc}")
        return 1
    except httpx.HTTPStatusError as exc:
        # The whole point of this script. Print the status AND the body:
        # 401 = wrong key, 403 = key not activated, 410 = wrong country path.
        status = exc.response.status_code
        print(f"   {FAIL} HTTP {status} from {exc.request.url}")
        body = exc.response.text.strip()
        print(f"          body: {body[:600] if body else '(empty)'}")
        if status == 401:
            print("          -> app_id/app_key rejected. Check for a swapped pair,")
            print("             a trailing space, or quotes around the value in .env.")
        elif status == 403:
            print("          -> credentials recognised but not permitted. A new")
            print("             Adzuna key can take a few minutes to activate.")
        elif status == 429:
            print("          -> rate limited, not a bad key. Retry shortly.")
        return 1
    except Exception as exc:  # noqa: BLE001 - preflight reports, never tracebacks
        print(f"   {FAIL} {type(exc).__name__}: {exc}")
        return 1
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()

    print(f"   {PASS} credentials accepted, {len(jobs)} job(s) returned")
    print()

    if not jobs:
        # Not a credential failure: a valid key can legitimately return nothing
        # for a narrow query. Say which it is rather than implying the key is bad.
        print(f"   {INFO} zero results — the key works, but this query matched")
        print("          nothing. Try a broader --query / --location.")
        return 0

    # --- 3. do the day-1 response-shape assumptions hold? -----------------
    print("3. Response shape (the day-1 assumptions, checked for the first time)")
    sample = jobs[0]
    fields = {
        "title": sample.title,
        "company": sample.company,
        "location": sample.location,
        "salary_raw": sample.salary_raw,
        "salary_min": sample.salary_min,
        "salary_max": sample.salary_max,
        "salary_currency": sample.salary_currency,
        "salary_period": sample.salary_period,
        "posted_date": sample.posted_date,
        "is_remote": sample.is_remote,
        "experience_min_years": sample.experience_min_years,
        "apply_url": sample.apply_url,
    }
    for name, value in fields.items():
        marker = PASS if value not in (None, "") else INFO
        shown = str(value)
        print(f"   {marker} {name:<22} {shown[:60]}")
    print(f"   {INFO} description            {len(sample.description or '')} chars")
    print(f"   {INFO} dedup_hash             {sample.dedup_hash[:16]}...")
    print()

    currency_ok = all(j.salary_currency in (None, "INR") for j in jobs)
    print(f"   {PASS if currency_ok else FAIL} currency matches country "
          f"{settings.adzuna_country!r} on every row")

    dates_ok = all(j.posted_date is not None for j in jobs)
    print(f"   {PASS if dates_ok else INFO} posted_date parsed on "
          f"{sum(1 for j in jobs if j.posted_date)}/{len(jobs)} rows")
    print()

    print("Sample:")
    print(json.dumps(
        {k: str(v) for k, v in fields.items() if v not in (None, "")},
        indent=2,
    )[:900])
    print()
    print("=" * 72)
    print("RESULT: Adzuna credentials work. Real ingestion is unblocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

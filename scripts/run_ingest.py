#!/usr/bin/env python
"""CLI entry point for a single ingestion run.

    python scripts/run_ingest.py --query "python backend" --location "Bengaluru"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `import app...` work when the script is run directly from the repo root
# (python scripts/run_ingest.py) without requiring an editable install first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.ingest import IngestionResult, run_ingestion  # noqa: E402
from app.sources.adzuna import AdzunaSource  # noqa: E402
from app.sources.base import JobSource, SearchQuery  # noqa: E402

# The only place source classes are named. A new source is added here and in its
# own file — nothing else changes.
SOURCE_REGISTRY: dict[str, type[JobSource]] = {
    AdzunaSource.name: AdzunaSource,
}

_COLUMNS: tuple[tuple[str, str, int], ...] = (
    # (header, StoredJob attribute, max display width)
    ("TITLE", "title", 42),
    ("COMPANY", "company", 26),
    ("LOCATION", "location", 24),
    ("SALARY", "salary_raw", 38),
    ("STATUS", "_status", 9),
)


def _truncate(value: str, width: int) -> str:
    return value if len(value) <= width else value[: width - 1] + "…"


def _render_table(result: IngestionResult) -> str:
    """Format stored jobs as a fixed-width table.

    Hand-rolled instead of pulling in tabulate/rich: one screen of formatting is
    cheaper to own than a dependency in a project that has no other use for it.
    """
    if not result.stored:
        return "(no jobs stored)"

    rows: list[list[str]] = []
    for job in result.stored:
        rows.append(
            [
                _truncate(job.title or "-", _COLUMNS[0][2]),
                _truncate(job.company or "-", _COLUMNS[1][2]),
                _truncate(job.location or "-", _COLUMNS[2][2]),
                _truncate(job.salary_raw or "-", _COLUMNS[3][2]),
                "NEW" if job.is_new else "duplicate",
            ]
        )

    headers = [column[0] for column in _COLUMNS]
    # Size each column to its widest actual value so short result sets stay tight.
    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def _build_sources(names: list[str]) -> list[JobSource]:
    unknown = [name for name in names if name not in SOURCE_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown source(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(SOURCE_REGISTRY))}"
        )
    return [SOURCE_REGISTRY[name]() for name in names]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ingest",
        description="Fetch jobs from configured sources and store them.",
    )
    parser.add_argument("--query", required=True, help='Search keywords, e.g. "python backend"')
    parser.add_argument("--location", default=None, help='Location filter, e.g. "Bengaluru"')
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=sorted(SOURCE_REGISTRY),
        help="Source to fetch from; repeatable. Defaults to all registered sources.",
    )
    parser.add_argument("--max-results", type=int, default=100, help="Cap on jobs fetched (default: 100)")
    parser.add_argument("--remote-only", action="store_true", help="Keep only postings detected as remote")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    query = SearchQuery(
        keywords=args.query,
        location=args.location,
        remote_only=args.remote_only,
        max_results=args.max_results,
    )
    sources = _build_sources(args.sources or sorted(SOURCE_REGISTRY))

    try:
        result = run_ingestion(query=query, sources=sources)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, do not traceback
        # run_ingestion has already written the reason to ingestion_runs.
        print(f"\nIngestion failed: {exc!r}", file=sys.stderr)
        return 1
    finally:
        for source in sources:
            close = getattr(source, "close", None)
            if callable(close):
                close()

    print()
    print(_render_table(result))
    print()
    print(
        f"run={result.run_id}  status={result.status}  "
        f"stored={result.jobs_found}  new={result.new_count}  "
        f"duplicate={result.duplicate_count}"
    )
    for name, error in result.source_errors.items():
        print(f"  source {name} failed: {error}", file=sys.stderr)

    # Non-zero on partial/failed so the command is usable in a scheduled job.
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())

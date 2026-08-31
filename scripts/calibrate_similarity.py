#!/usr/bin/env python
"""Measure the real resume-to-chunk cosine distribution, to set COS_LO/COS_HI.

    python scripts/calibrate_similarity.py --resume-id <uuid>

Run this **after** an ingestion has stored real jobs and chunked them.

Why it exists. ``scorer.COS_LO`` and ``COS_HI`` are guesses, and guesses of a
particular kind: cosine similarity between two pieces of related English text
does not use the [0, 1] range. Everything a resume is compared against is a job
description — same language, same register, same subject matter — so the scores
cluster in a narrow band, often something like [0.45, 0.75]. Feed that straight
into the weighted sum and the semantic term is nearly constant across every job
in the result set: it moves the *absolute* score but does almost nothing to the
*ranking*, so its nominal 40% weight buys 40% of nothing.

Rescaling the observed band onto [0, 1] is what restores the spread. This script
tells you what the band actually is for your resume, your embedding model and
your corpus — all three of which move it.

Reading the output: set COS_LO to roughly p5 and COS_HI to roughly p95. Not min
and max: the extremes are one weird posting each, and anchoring to them puts
every real job back in a narrow middle. Clamping deliberately sacrifices the
bottom and top 5% — those jobs are already unambiguously bad or good, and their
exact ordering matters far less than resolution across the middle 90%.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.matching.scorer import (  # noqa: E402
    COS_HI,
    COS_LO,
    TOP_K_CHUNKS,
    cosine_similarity,
)
from app.models import Job, JobChunk, Resume  # noqa: E402

PERCENTILES = (5, 25, 50, 75, 95)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. No numpy for one function."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _describe(label: str, values: list[float]) -> str:
    if not values:
        return f"{label:<26} (no data)"
    stats = [
        f"n={len(values)}",
        f"min={min(values):.4f}",
        f"mean={sum(values) / len(values):.4f}",
        f"max={max(values):.4f}",
    ]
    percentiles = "  ".join(
        f"p{p}={percentile(values, p):.4f}" for p in PERCENTILES
    )
    return f"{label}\n  {'  '.join(stats)}\n  {percentiles}"


def _suggestion(values: list[float]) -> str:
    if len(values) < 20:
        return (
            "Not enough data to calibrate confidently (fewer than 20 samples). "
            "Ingest more jobs and re-run."
        )
    low = percentile(values, 5)
    high = percentile(values, 95)
    spread = high - low
    lines = [
        "Suggested values for app/matching/scorer.py:",
        "",
        f"    COS_LO, COS_HI = {low:.2f}, {high:.2f}",
        "",
        f"Currently set to {COS_LO}, {COS_HI}.",
    ]
    if spread < 0.05:
        lines += [
            "",
            "WARNING: the observed band is under 0.05 wide. Rescaling it will "
            "amplify noise as much as signal — every job looks alike to the "
            "embedding model. That usually means the resume embedding text is "
            "too generic; check what build_resume_embedding_text produced.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate_similarity",
        description="Print the resume-to-job-chunk cosine distribution.",
    )
    parser.add_argument("--resume-id", required=True, type=UUID)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap on jobs examined (default: all)"
    )
    parser.add_argument(
        "--show-extremes",
        type=int,
        default=5,
        help="Print the N best and worst jobs by top-3 mean (default: 5)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level="WARNING")

    with SessionLocal() as session:
        resume = session.get(Resume, args.resume_id)
        if resume is None:
            print(f"No resume with id {args.resume_id}", file=sys.stderr)
            return 1
        if resume.embedding is None:
            print(
                "Resume has no embedding. POST it again, or PATCH its skills, "
                "so build_resume_embedding_text runs.",
                file=sys.stderr,
            )
            return 1

        resume_vec = list(resume.embedding)

        stmt = (
            select(Job.id, Job.title, Job.company)
            .join(JobChunk, JobChunk.job_id == Job.id)
            .group_by(Job.id, Job.title, Job.company)
            .order_by(func.count(JobChunk.id).desc())
        )
        if args.limit:
            stmt = stmt.limit(args.limit)
        jobs = list(session.execute(stmt))

        if not jobs:
            print(
                "No chunked jobs in the database. Run a search (or "
                "scripts/run_ingest.py plus a search) first.",
                file=sys.stderr,
            )
            return 1

        all_chunk_scores: list[float] = []
        per_job_top: list[tuple[float, str, str | None, int]] = []

        for job_id, title, company in jobs:
            vectors = [
                list(embedding)
                for (embedding,) in session.execute(
                    select(JobChunk.embedding)
                    .where(JobChunk.job_id == job_id)
                    .order_by(JobChunk.chunk_index)
                )
            ]
            if not vectors:
                continue
            scores = [cosine_similarity(resume_vec, vector) for vector in vectors]
            all_chunk_scores.extend(scores)
            top = sorted(scores, reverse=True)[:TOP_K_CHUNKS]
            per_job_top.append(
                (sum(top) / len(top), title, company, len(vectors))
            )

    print()
    print("=" * 72)
    print(f"Resume {args.resume_id}")
    print(f"{len(per_job_top)} job(s), {len(all_chunk_scores)} chunk(s)")
    print("=" * 72)
    print()
    # Both distributions matter. The per-chunk one shows the raw model behaviour;
    # the per-job top-3 mean is what semantic_component actually rescales, so it
    # is the one COS_LO/COS_HI should be read from.
    print(_describe("Per-chunk cosine (raw)", all_chunk_scores))
    print()
    print(
        _describe(
            f"Per-job top-{TOP_K_CHUNKS} mean  <-- calibrate on this",
            [entry[0] for entry in per_job_top],
        )
    )
    print()

    if args.show_extremes and per_job_top:
        ranked = sorted(per_job_top, reverse=True)
        n = min(args.show_extremes, len(ranked))
        print(f"Best {n}:")
        for score, title, company, chunks in ranked[:n]:
            print(f"  {score:.4f}  {title[:44]:<44}  {(company or '-')[:22]}  ({chunks} chunks)")
        print(f"Worst {n}:")
        for score, title, company, chunks in ranked[-n:]:
            print(f"  {score:.4f}  {title[:44]:<44}  {(company or '-')[:22]}  ({chunks} chunks)")
        print()

    print(_suggestion([entry[0] for entry in per_job_top]))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

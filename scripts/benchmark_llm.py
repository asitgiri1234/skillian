#!/usr/bin/env python
"""Diagnostic: is the ~170s resume extraction slow hardware, or a large schema?

    python scripts/benchmark_llm.py
    python scripts/benchmark_llm.py --repeats 3 --skip-full

The two causes have different fixes — a smaller model / hosted provider for slow
hardware, a smaller schema for constrained-decoding overhead — so this measures
them apart instead of guessing.

The ladder is the point. Each step adds exactly one variable:

    2. no schema at all      -> raw token generation speed on this machine
    3. a 2-field schema      -> the fixed cost of constrained decoding
    4. the real ParsedResume -> what the full schema actually costs
    5. two half-schemas      -> whether splitting recovers any of it

Everything goes through ``app.providers`` rather than a fresh ``ollama.Client``,
so the numbers include whatever the real code path costs — settings loading,
client construction, the provider's own error handling. Model names come from
config for the same reason.

**This script only measures. It changes no application code and writes nothing
to the database.**
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.providers import (  # noqa: E402
    EmbeddingError,
    LLMError,
    get_embedding_provider,
    get_llm_provider,
)
from app.structure import (  # noqa: E402
    EducationRef,
    ExperienceRef,
    ParsedResume,
    build_resume_embedding_text,
)

# --- fixtures ---------------------------------------------------------------

SHORT_SENTENCE = "Priya Raman is a data engineer based in Chennai, India."

# ~600 words. Written inline so the script needs no uploaded file.
RESUME_TEXT = """
DEVIKA RAGHAVAN
Bengaluru, Karnataka, India
devika.raghavan@example.invalid | +91 98450 77120 | github.com/devikar

PROFESSIONAL SUMMARY
Backend and platform engineer with seven years of experience designing,
shipping and operating distributed services in Python and Go. Comfortable
owning a system end to end, from the design document through to the pager.
Particular interest in data-intensive systems and in making other engineers
faster.

EXPERIENCE

Senior Platform Engineer, Nimbus Freight Technologies, Bengaluru
August 2021 - Present
- Led the redesign of the shipment tracking service, moving it from a single
  Django monolith to three Python services communicating over Kafka. Cut p99
  latency on the tracking endpoint from 3.4 seconds to 310 milliseconds.
- Designed the PostgreSQL partitioning scheme for the events table, which had
  grown to 1.2 billion rows. Wrote the online migration that moved it without
  downtime over a nine-day window.
- Built an internal deployment tool in Go that reduced the median time from
  merge to production from forty minutes to six.
- Ran the on-call rotation for a team of nine, wrote the incident review
  template still in use, and mentored three junior engineers.
- Introduced structured logging and distributed tracing with OpenTelemetry,
  which made a class of cross-service timeout bugs diagnosable for the first
  time.

Backend Engineer, Harbourline Analytics, Pune
June 2019 - July 2021
- Built the ingestion pipeline that pulled clickstream data from twelve client
  systems into a warehouse, handling roughly 40 million events a day.
- Owned the REST API used by the customer dashboard, written in FastAPI and
  backed by PostgreSQL and Redis.
- Replaced a nightly batch job with an incremental Celery-based pipeline,
  reducing data freshness from eighteen hours to under ten minutes.
- Wrote the company's first integration test suite, taking coverage of the
  ingestion path from nothing to roughly seventy percent.

Software Engineer, Castille Systems, Pune
July 2018 - May 2019
- Maintained a legacy Java billing service and wrote its replacement in Python.
- Automated a manual reconciliation process that had taken two people three
  days each month.

Junior Developer, Trellis Interactive, Nagpur
July 2017 - June 2018
- Wrote internal reporting scripts in Python and maintained a set of scheduled
  jobs that produced the monthly figures the finance team relied on.
- Fixed defects across a PHP content management system and helped migrate its
  templates during a rebrand.
- Built a small internal tool for tracking asset licences, which the design
  team used for several years afterwards.

SELECTED PROJECTS

Shipment event replay (Nimbus, 2023)
Designed and built a replay mechanism that could reconstruct the state of any
shipment at any past timestamp from the Kafka event log. Used during three
customer disputes and during the partitioning migration to verify that no
events had been lost. Written in Python with a small Go component for the
high-throughput scan path.

Query advisor (Harbourline, 2020)
An internal service that watched pg_stat_statements, identified queries whose
plans had regressed after a deploy, and posted them to the team channel with a
suggested index. Reduced the number of performance incidents reaching on-call
by roughly half over two quarters.

TECHNICAL SKILLS
Languages: Python, Go, SQL, JavaScript, Bash
Frameworks: FastAPI, Django, Flask, Celery
Data: PostgreSQL, Redis, Kafka, ClickHouse, Elasticsearch
Infrastructure: Docker, Kubernetes, Terraform, AWS, GitHub Actions, Prometheus,
Grafana, OpenTelemetry
Practices: distributed systems design, database performance tuning, incident
response, code review, technical writing

EDUCATION
Bachelor of Engineering, Computer Science
College of Engineering Pune, 2018
First class with distinction. Final year project on distributed consensus.

CERTIFICATIONS
Certified Kubernetes Application Developer, 2022
AWS Certified Solutions Architect - Associate, 2020

OTHER
Maintainer of a small open-source library for Postgres connection pooling,
around 900 stars. Occasional conference speaker; gave a talk at PyCon India
2023 on partitioning large tables without downtime. Reads a great deal of
science fiction and runs long distances slowly.
""".strip()

# One chunk of roughly 300 words, for the single-embedding measurement.
EMBED_CHUNK = " ".join(RESUME_TEXT.split()[:300])


class TinyExtract(BaseModel):
    """The smallest useful constrained schema: two required string fields."""

    name: str = Field(description="The person's full name")
    city: str = Field(description="The city they are based in")


class ContactAndSkills(BaseModel):
    """Half of ParsedResume: the flat fields plus the skill list.

    Kept for the step-5 split measurement, which was run and **rejected** — the
    two halves emit the same total tokens as the whole, so the split bought
    nothing. Retained so the number can be reproduced rather than taken on
    trust.
    """

    name: str | None = Field(description="Candidate's full name")
    email: str | None = Field(description="Primary email address")
    phone: str | None = Field(description="Primary phone number")
    location: str | None = Field(description="City and country of residence")
    summary: str | None = Field(description="Professional summary or objective")
    skills: list[str] = Field(description="Technical and professional skills")
    total_years_experience: float | None = Field(
        ge=0, le=60, description="Total years of professional experience"
    )


class History(BaseModel):
    """The other half: the two nested list-of-object fields."""

    experience: list[ExperienceRef] = Field(
        description="Work history, most recent first"
    )
    education: list[EducationRef] = Field(description="Degrees and qualifications")


_SYSTEM = (
    "You extract structured data from resumes. Copy values verbatim from the "
    "document. If a field is genuinely absent, leave it empty rather than "
    "guessing or inventing a plausible value."
)


def _prompt(text: str) -> str:
    return (
        "Extract every field you can from the following resume.\n\n"
        f"--- RESUME ---\n{text}\n--- END RESUME ---"
    )


# --- timing -----------------------------------------------------------------


@dataclass
class Result:
    """One benchmark step across several repeats."""

    key: str
    #: Step number as printed above, e.g. "5a". Carried explicitly rather than
    #: derived from list position, because step 5 occupies two rows.
    number: str
    label: str
    runs: list[float] = field(default_factory=list)
    #: Indices (1-based) whose model was not yet resident in memory.
    cold_runs: list[int] = field(default_factory=list)
    detail: str = ""
    error: str | None = None
    skipped: bool = False

    @property
    def warm(self) -> list[float]:
        """Runs excluding cold ones — the honest steady-state figure."""
        return [t for i, t in enumerate(self.runs, 1) if i not in self.cold_runs]

    def stat(self, fn: Callable[[list[float]], float]) -> float | None:
        values = self.warm or self.runs
        return fn(values) if values else None


def _schema_size(model: type[BaseModel]) -> tuple[int, int]:
    """(characters, top-level properties) of a model's JSON schema."""
    import json

    schema = model.model_json_schema()
    return len(json.dumps(schema)), len(schema.get("properties", {}))


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def run_repeats(
    result: Result, fn: Callable[[], Any], repeats: int, cold_first: bool = False
) -> Result:
    """Execute ``fn`` ``repeats`` times, recording wall-clock seconds."""
    if cold_first:
        result.cold_runs.append(1)
    for index in range(1, repeats + 1):
        try:
            elapsed, _ = timed(fn)
        except (LLMError, EmbeddingError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"    run {index}: FAILED — {result.error}")
            return result
        result.runs.append(elapsed)
        tag = "  (cold)" if index in result.cold_runs else ""
        print(f"    run {index}: {elapsed:8.2f}s{tag}")
    return result


# --- the steps --------------------------------------------------------------


def step_1_daemon(settings: Any) -> list[str]:
    """Confirm the daemon answers and report which models are pulled.

    Uses ollama.Client directly, not the provider: the provider ABC has no
    "list models" call, and this is a precondition check rather than one of the
    timed measurements.
    """
    print("1. Daemon reachable")
    client = ollama.Client(host=settings.ollama_host, timeout=10)
    try:
        elapsed, listing = timed(client.list)
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal
        print(f"   [FAIL] Cannot reach the Ollama daemon at {settings.ollama_host}")
        print(f"          {type(exc).__name__}: {exc}")
        print("          Start it with: ollama serve")
        raise SystemExit(1) from exc

    names = sorted(
        getattr(model, "model", None) or getattr(model, "name", "?")
        for model in listing.models
    )
    print(f"   [PASS] {settings.ollama_host} responded in {elapsed:.3f}s")
    print(f"          {len(names)} model(s): {', '.join(names)}")

    missing = [
        wanted
        for wanted in (settings.ollama_llm_model, settings.ollama_embed_model)
        if not any(name.split(":")[0] == wanted.split(":")[0] for name in names)
    ]
    if missing:
        print(f"   [FAIL] configured model(s) not pulled: {', '.join(missing)}")
        print(f"          Run: ollama pull {missing[0]}")
        raise SystemExit(1)
    print(f"   [PASS] configured LLM       {settings.ollama_llm_model}")
    print(f"   [PASS] configured embedding {settings.ollama_embed_model}")
    return names


def step_5_concurrent(text: str) -> float:
    """Both halves at once, each on its own provider instance.

    Separate providers rather than one shared client, so the measurement is of
    Ollama's own concurrency rather than of contention inside a single httpx
    connection pool.
    """
    def contact() -> Any:
        return get_llm_provider().complete(_prompt(text), ContactAndSkills, system=_SYSTEM)

    def history() -> Any:
        return get_llm_provider().complete(_prompt(text), History, system=_SYSTEM)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(contact), pool.submit(history)]
        for future in futures:
            future.result()
    return 0.0


# --- verify-trim ------------------------------------------------------------

#: The pre-trim figures from the diagnostic run, printed alongside the new ones
#: so the comparison never depends on remembering them. Same resume, same
#: machine, same model, warm.
BASELINE_TOKENS = 941
BASELINE_SECONDS = 217.1

#: Skills that appear unambiguously in RESUME_TEXT. Extraction has to find
#: these, or the speedup was bought by losing the thing matching runs on.
EXPECTED_SKILLS = (
    "python", "go", "sql", "postgresql", "redis", "kafka",
    "docker", "kubernetes", "terraform", "aws", "fastapi", "django",
)

#: Field names removed from the schema. If one comes back, the model is being
#: asked for prose again and the token count will have crept back up with it.
REMOVED_FIELDS = (
    "summary", "description", "start_date", "end_date", "is_current",
    "location", "field_of_study", "graduation_year", "total_years_experience",
)


def _capture_tokens(llm: Any) -> dict[str, Any]:
    """Wrap the provider's client to record Ollama's own token counters.

    In-memory only, on this one instance. The provider ABC discards these
    deliberately (they are an Ollama detail), but the whole question here is
    output volume, so the measurement needs them.
    """
    stats: dict[str, Any] = {}
    original = llm._client.chat

    def chat(**kwargs: Any) -> Any:
        response = original(**kwargs)
        stats["in"] = getattr(response, "prompt_eval_count", None)
        stats["out"] = getattr(response, "eval_count", None)
        return response

    llm._client.chat = chat
    return stats


def verify_trim() -> int:
    """Measure the trimmed schema and assert it did not lose anything."""
    settings = get_settings()
    print("=" * 78)
    print("VERIFY TRIM: trimmed ParsedResume vs the pre-trim baseline")
    print("=" * 78)
    print(f"  model           {settings.ollama_llm_model}")
    print(f"  resume fixture  {len(RESUME_TEXT.split())} words (same as the diagnostic)")
    chars, props = _schema_size(ParsedResume)
    print(f"  schema          {chars} chars, {props} top-level fields")
    print()

    step_1_daemon(settings)
    print()

    llm = get_llm_provider()
    stats = _capture_tokens(llm)

    print("Warming the model (throwaway call)...")
    warm_seconds, _ = timed(lambda: llm.complete_text("Reply with exactly one word: hello"))
    print(f"  warm-up {warm_seconds:.2f}s")
    print()

    print("1. Trimmed ParsedResume extraction")
    elapsed, raw = timed(
        lambda: llm.complete(_prompt(RESUME_TEXT), ParsedResume, system=_SYSTEM)
    )
    out_tokens = stats.get("out")
    in_tokens = stats.get("in")
    print(f"   wall clock      {elapsed:.2f}s")
    print(f"   input tokens    {in_tokens}")
    print(f"   output tokens   {out_tokens}")
    print()

    print("2. Throughput")
    if out_tokens:
        rate = out_tokens / elapsed
        baseline_rate = BASELINE_TOKENS / BASELINE_SECONDS
        print(f"   trimmed         {rate:.2f} tok/s")
        print(f"   baseline        {baseline_rate:.2f} tok/s")
        # If tok/s moved, the machine was in a different state and the speedup
        # cannot be attributed to the token reduction. Saying so is the point of
        # measuring it.
        if abs(rate - baseline_rate) < 1.0:
            print("   -> UNCHANGED: the win came from emitting fewer tokens")
        else:
            print("   -> CHANGED: machine state differs, speedup not attributable")
    print()

    print("3. Against the baseline")
    print(f"   baseline        {BASELINE_TOKENS} tokens / {BASELINE_SECONDS:.1f}s")
    if out_tokens:
        print(
            f"   trimmed         {out_tokens} tokens / {elapsed:.1f}s"
            f"   ({out_tokens / BASELINE_TOKENS:.0%} of tokens, "
            f"{elapsed / BASELINE_SECONDS:.0%} of time)"
        )
        print(f"   speedup         {BASELINE_SECONDS / elapsed:.2f}x")
    print()

    # --- correctness -------------------------------------------------------
    print("4. Correctness (a fast extraction that loses skills is worse than a slow one)")
    failures: list[str] = []

    try:
        parsed = ParsedResume.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print(f"   [FAIL] output did not validate: {exc}")
        return 1

    if parsed.skills:
        print(f"   [PASS] skills non-empty ({len(parsed.skills)}): "
              f"{', '.join(parsed.skills[:12])}")
    else:
        failures.append("skills list is empty")
        print("   [FAIL] skills list is empty")

    found = {s.casefold() for s in parsed.skills}
    missed = [s for s in EXPECTED_SKILLS if not any(s in f for f in found)]
    if missed:
        failures.append(f"skills present in the resume but not extracted: {missed}")
        print(f"   [FAIL] missed {len(missed)}/{len(EXPECTED_SKILLS)} expected: "
              f"{', '.join(missed)}")
    else:
        print(f"   [PASS] all {len(EXPECTED_SKILLS)} expected skills found")

    years = parsed.total_experience_years
    if years is None:
        failures.append("total_experience_years is null")
        print("   [FAIL] total_experience_years is null")
    elif not (3 <= years <= 12):
        failures.append(f"total_experience_years implausible: {years}")
        print(f"   [FAIL] total_experience_years = {years}, expected ~7")
    else:
        print(f"   [PASS] total_experience_years = {years} (plausible)")

    if not parsed.projects:
        # Not a hard failure: the fixture has a PROJECTS section, but a model
        # may reasonably fold them into experience.
        print("   [WARN] projects list is empty (fixture has a PROJECTS section)")
    elif all(p.title for p in parsed.projects):
        print(f"   [PASS] all {len(parsed.projects)} project(s) have a title: "
              f"{', '.join(p.title for p in parsed.projects)}")
    else:
        failures.append("a project has an empty title")
        print("   [FAIL] at least one project has an empty title")

    leaked = sorted(_find_keys(raw, REMOVED_FIELDS))
    if leaked:
        failures.append(f"removed fields present in output: {leaked}")
        print(f"   [FAIL] removed field(s) reappeared: {', '.join(leaked)}")
    else:
        print(f"   [PASS] none of the {len(REMOVED_FIELDS)} removed fields reappeared")

    print(f"   [INFO] roles {len(parsed.experience)}, education {len(parsed.education)}")
    print()

    # --- embedding text ----------------------------------------------------
    print("5. Embedding text (the known cost of the trim)")
    text = build_resume_embedding_text(parsed)
    print(f"   {len(text)} chars, {len(text.split())} words")
    print("   " + "\n   ".join(text.splitlines()[:6]))
    print()

    print("=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} correctness failure(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: all correctness checks passed")
    return 0


def _find_keys(obj: Any, wanted: tuple[str, ...]) -> set[str]:
    """Every key in a nested structure matching one of ``wanted``."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in wanted:
                found.add(key)
            found |= _find_keys(value, wanted)
    elif isinstance(obj, list):
        for item in obj:
            found |= _find_keys(item, wanted)
    return found


# --- report -----------------------------------------------------------------


def summary_table(results: list[Result]) -> str:
    # ASCII only: the Windows console is cp1252 and renders an em-dash as "?".
    header = f"{'#':<4} {'MEASUREMENT':<38} {'MIN':>9} {'MEDIAN':>9} {'MAX':>9}  NOTES"
    lines = [header, "-" * len(header)]
    for result in results:
        if result.skipped:
            lines.append(
                f"{result.number:<4} {result.label:<38} "
                f"{'n/a':>9} {'n/a':>9} {'n/a':>9}  skipped"
            )
            continue
        if result.error or not result.runs:
            lines.append(
                f"{result.number:<4} {result.label:<38} "
                f"{'n/a':>9} {'n/a':>9} {'n/a':>9}  FAILED: {result.error}"
            )
            continue
        low = result.stat(min)
        mid = result.stat(statistics.median)
        high = result.stat(max)
        note = result.detail
        if result.cold_runs:
            # Only claim exclusion when a warm run actually exists to fall back
            # on; with --repeats 1 the cold run *is* the number being reported.
            note += ("; " if note else "") + (
                f"run {result.cold_runs[0]} cold, excluded"
                if result.warm
                else "COLD ONLY (includes model load)"
            )
        lines.append(
            f"{result.number:<4} {result.label:<38} "
            f"{low:>8.2f}s {mid:>8.2f}s {high:>8.2f}s  {note}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_llm",
        description="Measure whether extraction cost is hardware or schema size.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per step (default: 3)")
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip step 4 (the ~170s one) for a fast pass",
    )
    parser.add_argument(
        "--verify-trim",
        action="store_true",
        help="Measure the trimmed ParsedResume against the pre-trim baseline "
        "and assert extraction quality; skips the rest of the ladder",
    )
    args = parser.parse_args(argv)

    if args.verify_trim:
        return verify_trim()

    settings = get_settings()
    repeats = max(1, args.repeats)

    print("=" * 78)
    print("Skillian LLM / embedding benchmark")
    print("=" * 78)
    print(f"  host            {settings.ollama_host}")
    print(f"  llm model       {settings.ollama_llm_model}")
    print(f"  embed model     {settings.ollama_embed_model}")
    print(f"  timeout         {settings.ollama_timeout_seconds:g}s")
    print(f"  repeats         {repeats}")
    print(f"  resume fixture  {len(RESUME_TEXT.split())} words, {len(RESUME_TEXT)} chars")
    print()

    step_1_daemon(settings)
    print()

    # Schema sizes, so the timings below can be read against them.
    print("   Schema sizes (JSON schema sent as format=):")
    for model in (TinyExtract, ContactAndSkills, History, ParsedResume):
        chars, props = _schema_size(model)
        print(f"     {model.__name__:<18} {chars:>6} chars, {props} top-level field(s)")
    print()

    llm = get_llm_provider()
    embedder = get_embedding_provider()
    results: list[Result] = []

    # --- 2 -----------------------------------------------------------------
    print("2. Trivial generation, no schema")
    r2 = Result("trivial", "2", "Trivial generation, no schema")
    # First LLM call of the process: the model is loaded from disk here.
    run_repeats(
        r2,
        lambda: llm.complete_text("Reply with exactly one word: hello"),
        repeats,
        cold_first=True,
    )
    r2.detail = "baseline token speed"
    results.append(r2)
    print()

    # --- 3 -----------------------------------------------------------------
    print("3. Small constrained generation (2-field schema)")
    r3 = Result("tiny", "3", "Small constrained gen (2 fields)")
    run_repeats(
        r3,
        lambda: llm.complete(
            f"Extract the name and city.\n\n{SHORT_SENTENCE}", TinyExtract
        ),
        repeats,
    )
    r3.detail = "cost of constrained decoding"
    results.append(r3)
    print()

    # --- 4 -----------------------------------------------------------------
    print("4. Full ParsedResume extraction")
    r4 = Result("full", "4", "Full ParsedResume extraction")
    if args.skip_full:
        r4.skipped = True
        print("    skipped (--skip-full)")
    else:
        run_repeats(
            r4,
            lambda: llm.complete(_prompt(RESUME_TEXT), ParsedResume, system=_SYSTEM),
            repeats,
        )
    r4.detail = "the real path"
    results.append(r4)
    print()

    # --- 5 -----------------------------------------------------------------
    print("5a. Split extraction, sequential")
    r5a = Result("split_seq", "5a", "Split extraction, sequential")

    def sequential() -> None:
        llm.complete(_prompt(RESUME_TEXT), ContactAndSkills, system=_SYSTEM)
        llm.complete(_prompt(RESUME_TEXT), History, system=_SYSTEM)

    run_repeats(r5a, sequential, repeats)
    r5a.detail = "ContactAndSkills then History"
    results.append(r5a)
    print()

    print("5b. Split extraction, concurrent (2 threads)")
    r5b = Result("split_conc", "5b", "Split extraction, concurrent")
    run_repeats(r5b, lambda: step_5_concurrent(RESUME_TEXT), repeats)
    r5b.detail = "both halves dispatched at once"
    results.append(r5b)
    print()

    # --- 6 -----------------------------------------------------------------
    print("6. Single embedding (~300 words)")
    r6 = Result("embed_one", "6", "Single embedding (~300 words)")
    # First embedding call: a different model, loaded from disk here.
    run_repeats(r6, lambda: embedder.embed(EMBED_CHUNK), repeats, cold_first=True)
    r6.detail = f"{len(EMBED_CHUNK.split())} words"
    results.append(r6)
    print()

    # --- 7 -----------------------------------------------------------------
    print("7. Batched embeddings (32 chunks, one call)")
    words = RESUME_TEXT.split()
    # Distinct chunks: identical strings would let any cache flatter the result.
    chunks = [
        " ".join(words[(i * 7) % max(1, len(words) - 200) :][:200]) + f" variant {i}"
        for i in range(32)
    ]
    r7 = Result("embed_batch", "7", "Batched embeddings (32 chunks)")
    run_repeats(r7, lambda: embedder.embed_batch(chunks), repeats)
    results.append(r7)

    batch_median = r7.stat(statistics.median)
    single_median = r6.stat(statistics.median)
    if batch_median is not None:
        per_chunk = batch_median / 32
        r7.detail = f"{per_chunk * 1000:.0f}ms/chunk"
        print(f"    per-chunk average: {per_chunk:.4f}s")
        if single_median:
            print(
                f"    vs {single_median:.4f}s for one alone "
                f"-> {single_median / per_chunk:.1f}x faster per chunk"
            )
    print()

    # --- summary -----------------------------------------------------------
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(summary_table(results))
    print()
    print("Cold runs are the first call against a model, which includes loading")
    print("its weights from disk. They are shown above but excluded from the")
    print("min/median/max columns.")
    print()

    # --- derived figures ---------------------------------------------------
    trivial = r2.stat(statistics.median)
    tiny = r3.stat(statistics.median)
    full = r4.stat(statistics.median)
    seq = r5a.stat(statistics.median)
    conc = r5b.stat(statistics.median)

    print("DERIVED")
    print("-" * 78)
    if trivial is not None and tiny is not None:
        print(f"  constrained decoding overhead (3 - 2)   {tiny - trivial:+8.2f}s")
    if full is not None and tiny is not None:
        print(f"  large-schema cost (4 - 3)               {full - tiny:+8.2f}s")
    if full is not None and trivial is not None:
        print(f"  full extraction vs bare generation      {full / trivial:8.1f}x")
    if full is not None and seq is not None:
        print(f"  split sequential vs full (5a - 4)       {seq - full:+8.2f}s"
              f"  ({(1 - seq / full) * 100:+.0f}%)")
    if seq is not None and conc is not None:
        print(f"  concurrency gain (5a - 5b)              {seq - conc:+8.2f}s"
              f"  ({(1 - conc / seq) * 100:+.0f}%)")
        print(f"  -> Ollama is {'PARALLELISING' if conc < seq * 0.75 else 'SERIALISING'}"
              " these two requests")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Ollama preflight check. Run this before anything else that needs a model.

    python scripts/check_ollama.py

Verifies, in order, that the daemon is reachable, that both configured models are
pulled, that the embedding width matches the database schema, and then times one
real extraction and one real embedding. Exits non-zero on the first hard failure
so it is usable as a setup gate in a script or CI step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import ollama  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.models import EMBEDDING_DIM  # noqa: E402
from app.providers import get_embedding_provider, get_llm_provider  # noqa: E402
from app.structure import ParsedResume, extract_resume  # noqa: E402

# Short but realistic: long enough to exercise every branch of the schema
# (contact block, skills, two roles, education) without making the timing figure
# dominated by prompt length.
SAMPLE_RESUME = """
Priya Sharma
Bengaluru, India | priya.sharma@gmail.com | +91 98765 43210

SUMMARY
Backend engineer with 6 years building Python services at scale.

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS

EXPERIENCE
Senior Backend Engineer, Acme Technologies Pvt Ltd — Bengaluru
March 2022 - Present
Led migration of a monolith to FastAPI services. Cut p99 latency by 40%.

Backend Engineer, Globex Systems — Pune
July 2019 - February 2022
Built payment reconciliation pipelines in Python and PostgreSQL.

EDUCATION
B.E. Computer Science, Pune Institute of Technology, 2019
"""

SAMPLE_TEXT = "Senior Python engineer with FastAPI and PostgreSQL experience."

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[ ok ]"


def _normalize(model: str) -> str:
    """Ollama reports "name:latest" for an untagged pull; compare tag-insensitively."""
    return model.split(":")[0]


def main() -> int:
    settings = get_settings()
    print("=" * 68)
    print("Ollama preflight")
    print("=" * 68)
    print(f"  host            {settings.ollama_host}")
    print(f"  LLM_PROVIDER    {settings.llm_provider}")
    print(f"  EMBED_PROVIDER  {settings.embedding_provider}")
    print(f"  llm model       {settings.ollama_llm_model}")
    print(f"  embed model     {settings.ollama_embed_model}")
    print()

    # --- 1. daemon reachable ---------------------------------------------
    print("1. Daemon reachable")
    client = ollama.Client(host=settings.ollama_host, timeout=10.0)
    try:
        listing = client.list()
    except (httpx.TransportError, ConnectionError, ollama.ResponseError) as exc:
        print(f"   {FAIL} cannot reach {settings.ollama_host}: {exc}")
        print("          start it with:  ollama serve")
        return 1
    installed = [model.model for model in listing.models if model.model]
    print(f"   {PASS} reachable, {len(installed)} model(s) installed")
    print()

    # --- 2. models pulled -------------------------------------------------
    print("2. Required models present")
    available = {_normalize(name) for name in installed}
    missing: list[str] = []
    for label, model in (
        ("llm", settings.ollama_llm_model),
        ("embedding", settings.ollama_embed_model),
    ):
        if _normalize(model) in available:
            print(f"   {PASS} {label:9} {model}")
        else:
            print(f"   {FAIL} {label:9} {model} is NOT pulled")
            missing.append(model)
    if missing:
        print()
        for model in missing:
            print(f"          ollama pull {model}")
        return 1
    print()

    # --- 3. embedding, with a dimension check against the schema ----------
    print("3. Embedding")
    embedder = get_embedding_provider(settings)
    started = time.perf_counter()
    try:
        vector = embedder.embed(SAMPLE_TEXT)
    except Exception as exc:  # noqa: BLE001 - preflight reports, never tracebacks
        print(f"   {FAIL} embed failed: {exc!r}")
        return 1
    embed_seconds = time.perf_counter() - started
    print(f'   {INFO} input     "{SAMPLE_TEXT[:52]}..."')
    print(f"   {PASS} returned  {len(vector)} dimensions in {embed_seconds:.3f}s")

    # The check that matters: a mismatch here fails later at INSERT with an
    # opaque pgvector error, long after the cause.
    if len(vector) != EMBEDDING_DIM:
        print(f"   {FAIL} schema mismatch: models.EMBEDDING_DIM is {EMBEDDING_DIM}")
        print("          the vector() columns and this model disagree; "
              "fix the model or migrate the columns")
        return 1
    print(f"   {PASS} matches models.EMBEDDING_DIM ({EMBEDDING_DIM})")
    print()

    # --- 4. structured extraction ----------------------------------------
    print("4. Structured extraction")
    llm = get_llm_provider(settings)
    schema_keys = list(ParsedResume.model_json_schema().get("properties", {}))
    print(f"   {INFO} schema    {len(schema_keys)} fields: {', '.join(schema_keys[:6])}...")
    started = time.perf_counter()
    try:
        parsed = extract_resume(SAMPLE_RESUME, llm=llm)
    except Exception as exc:  # noqa: BLE001 - preflight reports, never tracebacks
        print(f"   {FAIL} extraction failed: {exc!r}")
        return 1
    extract_seconds = time.perf_counter() - started
    print(f"   {PASS} completed in {extract_seconds:.2f}s")
    print(f"   {INFO} name      {parsed.name}")
    print(f"   {INFO} email     {parsed.email}")
    print(f"   {INFO} location  {parsed.location}")
    print(f"   {INFO} years     {parsed.total_years_experience}")
    print(f"   {INFO} skills    {len(parsed.skills)}: {', '.join(parsed.skills[:8])}")
    print(f"   {INFO} roles     {len(parsed.experience)}")
    for entry in parsed.experience[:3]:
        print(f"            - {entry.title} @ {entry.company} ({entry.start_date} - {entry.end_date})")
    print(f"   {INFO} education {len(parsed.education)}")
    print()

    # --- summary ----------------------------------------------------------
    print("=" * 68)
    print(f"  embedding    {embed_seconds:8.3f}s  ({settings.ollama_embed_model})")
    print(f"  extraction   {extract_seconds:8.2f}s  ({settings.ollama_llm_model})")
    print("=" * 68)
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

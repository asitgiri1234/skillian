# Skillian

Job scraper and resume matcher.

**Status:** end-to-end matching works. Upload a resume, POST a search, poll the
run, read ranked matches with explanations. Jobs are fetched, normalised,
deduped and stored with a full audit trail; descriptions are chunked and
embedded; skills are extracted and classified required-vs-preferred; scoring is
skill overlap plus semantic similarity with an experience adjustment. No UI, no
auth.

Both models run locally through [Ollama](https://ollama.com) behind a provider
abstraction, so nothing leaves the machine and there is no API bill.

See [DECISIONS.md](DECISIONS.md) for why everything is the way it is.

## Setup

```bash
# 1. Configuration
cp .env.example .env        # then fill in your Adzuna credentials
                            # free key: https://developer.adzuna.com/

# 2. Database (Postgres 16 + pgvector)
docker compose up -d
# If port 5432 is already taken, set POSTGRES_PORT in .env and update
# DATABASE_URL to match.

# 3. Python environment
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

# 4. Models (~5 GB, one time)
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 5. Preflight — run this before anything else that needs a model
python scripts/check_ollama.py

# 6. Schema
alembic upgrade head
```

`check_ollama.py` verifies the daemon is up, both models are pulled, and that the
embedding width matches `models.EMBEDDING_DIM`, then times one real extraction
and one real embedding. It exits non-zero on the first hard failure, so it works
as a setup gate.

```
1. Daemon reachable          [PASS] reachable, 4 model(s) installed
2. Required models present   [PASS] llm       qwen2.5:7b
                             [PASS] embedding nomic-embed-text
3. Embedding                 [PASS] returned  768 dimensions in 2.820s
                             [PASS] matches models.EMBEDDING_DIM (768)
4. Structured extraction     [PASS] completed in 72.27s
  embedding       2.820s  (nomic-embed-text)
  extraction     72.27s  (qwen2.5:7b)
```

## Run an ingestion

```bash
python scripts/run_ingest.py --query "python backend" --location "Bengaluru"
```

```
TITLE                      COMPANY                    LOCATION              SALARY                                  STATUS
-------------------------  -------------------------  --------------------  --------------------------------------  ---------
Senior Python Engineer     Acme Technologies Pvt Ltd  Bengaluru, Karnataka  INR2,400,000 - INR3,600,000 per year     NEW
Backend Engineer - Remote  Globex Inc.                Remote, India         INR1,800,000 per year (estimated)        NEW
Platform Engineer          Initech Private Limited    Pune, Maharashtra     -                                        duplicate

run=8d0d7001-...  status=success  stored=3  new=2  duplicate=1
```

Options: `--source` (repeatable), `--max-results`, `--remote-only`.
Exit code is non-zero on a `partial` or `failed` run, so it is safe to schedule.

## Run a search

```bash
uvicorn app.main:app --reload
```

```bash
# 1. Upload a resume. SLOW — 60-120s, a local 7b model reads the whole document.
curl -X POST localhost:8000/resumes -H 'content-type: application/json' -d '{
  "email": "you@example.com",
  "raw_text": "ASHA MENON\nSenior Backend Engineer\n..."
}'
# -> {"id": "<resume_id>", "skills": [...], "has_embedding": true, ...}

# 2. Queue a search. Returns in single-digit milliseconds.
curl -X POST localhost:8000/searches -H 'content-type: application/json' -d '{
  "resume_id": "<resume_id>", "location": "Bengaluru"
}'
# -> {"run_id": "<run_id>", "status": "queued", "stage": "queued"}

# 3. Poll. Minutes, depending on how many new jobs need skill extraction.
curl localhost:8000/runs/<run_id>
# -> {"status":"running","stage":"embedding","stage_number":6,"stage_total":10,
#     "jobs_found":42,"is_terminal":false, ...}

# 4. Read the results.
curl 'localhost:8000/matches?resume_id=<resume_id>&limit=20&min_score=0.4'
curl localhost:8000/jobs/<job_id>          # full detail, including chunks
```

Poll on `is_terminal`, not on `status == "..."` — see "Two status vocabularies"
below.

`PATCH /resumes/{id}/skills` replaces the skill list when the parse got something
wrong. It re-embeds the resume and **deletes that resume's matches**, because
scores computed against the old skill set are wrong, not merely stale.

## How a search scores

```
overall = (0.6 * skill + 0.4 * semantic) * experience_multiplier
```

- **skill** — weighted *recall* of the job's requirements: `earned / possible`,
  where a required skill weighs 1.0 and a preferred one 0.4. Recall, not
  Jaccard, so knowing things the job never asked for never costs anything.
  A job whose requirements could not be parsed returns `None` here, not `0.0`;
  the score falls back to semantic alone and the row is flagged
  `skills_unparsed` so a UI can say so.
- **semantic** — cosine between the resume vector and each job *chunk*, mean of
  the best 3, rescaled from `[COS_LO, COS_HI]` onto `[0, 1]`. See calibration
  below — that rescaling is not cosmetic.
- **experience_multiplier** — 1.0 / 0.95 / 0.85 / 0.70 by how many years short
  the candidate is. Missing data on *either* side returns 1.0: most postings
  state no parseable requirement, and an absent number is not a shortfall.

Scoring makes **no model calls**. That is what makes a 200-job search take
seconds instead of ~25 minutes, and there is a test asserting the call counters.

## Calibrating the semantic score

`COS_LO, COS_HI = 0.45, 0.85` in `app/matching/scorer.py` are **placeholders**.

Cosine between related English text does not use the [0, 1] range — every
comparison here is a resume against a job description, so the scores cluster in a
narrow band. Unrescaled, the semantic term is nearly constant across a result set
and its 40% weight moves the ranking far less than it appears to.

```bash
python scripts/calibrate_similarity.py --resume-id <uuid>
```

Set `COS_LO` to about the printed p5 and `COS_HI` to about p95. Not min/max —
those are one weird posting each, and anchoring to them puts every real job back
in a narrow middle.

## Layout

```
app/
  config.py          pydantic-settings, reads .env
  db.py              engine, session factory, declarative Base
  models.py          all 9 tables; EMBEDDING_DIM lives here
  runs.py            run status + stage vocabularies, is_terminal()
  normalize.py       text normalisation + dedup hash (shared by all sources)
  ingest.py          fetch -> upsert -> record the run (the CLI path)
  structure.py       resume text -> ParsedResume; build_resume_embedding_text
  workers.py         BackgroundTasks entry point for a search run
  main.py            FastAPI app, routers, /health
  matching/
    chunking.py      description -> ordered chunks of ~200-400 tokens
    scorer.py        PURE. no ORM, no network, no LLM. the formula above
    skills.py        job skill extraction + canonicalisation to skills rows
    queries.py       parsed resume -> SearchQuery list (LLM-free)
    explain.py       2-3 sentence prose for a scored match
    pipeline.py      the 8 stages, and the ingestion_runs bookkeeping
  api/
    deps.py          session dependency, source registry
    resumes.py       POST /resumes, PATCH /resumes/{id}/skills
    searches.py      POST /searches, GET /runs/{id}, /matches, /jobs/{id}
  sources/
    base.py          JobSource ABC, SearchQuery, NormalizedJob
    adzuna.py        first implementation
  providers/
    llm.py           LLMProvider ABC: complete(...) -> dict, complete_text(...)
    ollama_llm.py    qwen2.5:7b via ollama.chat(format=..., temperature=0)
    embeddings.py    EmbeddingProvider ABC: embed(text) -> list[float]
    ollama_embed.py  nomic-embed-text, 768 dimensions
scripts/run_ingest.py
scripts/check_ollama.py
scripts/calibrate_similarity.py
tests/               138 tests; the pure ones need no services
alembic/             migrations
```

## The search pipeline

`run_search` in `app/matching/pipeline.py`, eight stages, each one writing its
progress to `ingestion_runs.stage` so polling shows real movement:

| Stage | Cost |
|---|---|
| a. load resume + skills | one query |
| b. build search queries | free, no model |
| c. fetch sources, dedupe on `dedup_hash`, upsert | network |
| d. extract + classify job skills | **LLM, per job with no skills yet** |
| e. chunk + embed jobs with no chunks | embeddings, batched |
| f. score every job | **no model calls** |
| g. bulk write matches | one statement |
| h. explain the top 20 | **LLM, hard cap of 20** |

Stages d and e skip work on "has no rows yet", not on "is new" — so a job whose
extraction was interrupted gets another chance, while one already done is never
re-read by the model.

Any exception marks the run `failed` with the error text written to the row, and
is then re-raised. Nothing is swallowed.

## Two status vocabularies

`ingestion_runs.status` carries both. Day 1's CLI writes `running -> success`;
the day-3 search pipeline writes `queued -> running -> succeeded`. Both are
declared in `app/runs.py`, and readers should use `is_terminal()` / `is_success()`
rather than comparing to a literal — `GET /runs/{id}` exposes `is_terminal` for
exactly this reason. It is a wart with a known fix; see DECISIONS 18.4.

## Tests

```bash
pytest                      # 138 tests
pytest tests/test_scorer.py tests/test_chunking.py   # pure, no services needed
```

The scorer and chunking tests need nothing running. The pipeline and API tests
need Postgres migrated to `0003` and skip cleanly without it. **No test calls
Ollama** — the LLM and embedding providers are injected everywhere they are used,
and `tests/conftest.py` supplies deterministic fakes.

## Swapping a model provider

`LLM_PROVIDER` and `EMBEDDING_PROVIDER` in `.env` select the implementation from
the registries in `app/providers/__init__.py`. Callers depend on the ABC and
never import a concrete provider, so adding a hosted backend is a new file plus
a registry entry.

Changing the *embedding model* is not just a config change: `EMBEDDING_DIM` in
`app/models.py` must match the model's width, and the `vector()` columns need an
Alembic migration to match. `check_ollama.py` asserts the two agree.

## Extraction: shape vs accuracy

`structure.py` passes `ParsedResume.model_json_schema()` to the model as
`format=`, so decoding is constrained by a grammar and the prompt never mentions
JSON. That guarantees **shape** — malformed JSON is unrepresentable.

It guarantees nothing about **accuracy**, which is why the validate-and-retry
loop stays. The validators check meaning rather than types: placeholder values
(the literal `"string"` is a real failure mode, copied out of the schema),
fake emails, duplicate skills, implausible years, and — most importantly — a
well-formed result with no skills and no experience, which means extraction
failed. On a validation failure the errors are fed back into the next prompt.

## Adding a source

1. Write `app/sources/<name>.py` with a class subclassing `JobSource`, setting
   `name`, and implementing `fetch(query) -> list[NormalizedJob]`.
2. Add it to `SOURCE_REGISTRY` in `scripts/run_ingest.py` (the CLI) and/or
   `app/api/deps.py` (the API). They are separate on purpose: enabling a source
   for scheduled ingestion is not the same decision as exposing it to an HTTP
   caller.

Nothing else changes. The source never touches the database, and it must not
compute its own `dedup_hash` — that is derived centrally so two providers agree
on what "the same posting" means.

## How duplicates work

Two distinct mechanisms, plus one policy difference between the two entry points:

- **`UNIQUE(source, source_job_id)`** — row identity. Re-fetching the same
  posting updates it in place; its `job_chunks` are keyed on `job_id` and are
  left alone, so a refresh never costs a re-embed.
- **`dedup_hash`** — `sha256` of normalised company + title + city, indexed but
  *not* unique.
- The **CLI** stores one row per board, because each copy has its own apply URL.
  A **search run** keeps one row per `dedup_hash` — showing a user the same job
  three times is bad, and each extra copy would cost a full local-model skill
  extraction.

## Known gaps

- **Adzuna has never been called for real.** `.env` still holds placeholder
  credentials; every response-shape assumption is unconfirmed.
- **`COS_LO`/`COS_HI` are uncalibrated** (see above). Do this first once real
  jobs are ingested.
- **`POST /resumes` blocks for 60-120s.** Once-per-resume, so it is tolerable,
  but it is the weakest endpoint here.
- **No auth, no rate limiting, no CORS.** `GET /matches?resume_id=` returns
  anyone's matches to anyone holding the UUID. Localhost only.
- **A background task dies with the process**, leaving the run at `running` with
  a stale stage. Visible, not silent — but not retried.

# Skillian

Job scraper and resume matcher.

**Status:** ingestion path + local model providers. Jobs are fetched,
normalised, deduped and stored with a full audit trail; resumes can be extracted
into structured data with a local LLM, and text can be embedded. Matching,
API endpoints and UI are not built yet.

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

## Layout

```
app/
  config.py          pydantic-settings, reads .env
  db.py              engine, session factory, declarative Base
  models.py          all 8 tables; EMBEDDING_DIM lives here
  normalize.py       text normalisation + dedup hash (shared by all sources)
  ingest.py          fetch -> upsert -> record the run
  structure.py       resume text -> ParsedResume, with validate-and-retry
  main.py            FastAPI app object (no routes yet)
  sources/
    base.py          JobSource ABC, SearchQuery, NormalizedJob
    adzuna.py        first implementation
  providers/
    llm.py           LLMProvider ABC: complete(prompt, schema) -> dict
    ollama_llm.py    qwen2.5:7b via ollama.chat(format=..., temperature=0)
    embeddings.py    EmbeddingProvider ABC: embed(text) -> list[float]
    ollama_embed.py  nomic-embed-text, 768 dimensions
scripts/run_ingest.py
scripts/check_ollama.py
alembic/             migrations
```

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
2. Add it to `SOURCE_REGISTRY` in `scripts/run_ingest.py`.

Nothing else changes. The source never touches the database, and it must not
compute its own `dedup_hash` — that is derived centrally so two providers agree
on what "the same posting" means.

## How duplicates work

Two distinct mechanisms:

- **`UNIQUE(source, source_job_id)`** — row identity. Re-fetching the same
  posting updates it in place (an `embedding`, once present, is preserved).
- **`dedup_hash`** — `sha256` of normalised company + title + city, indexed but
  *not* unique. The same job posted to several boards is stored once per board
  (each has its own apply URL) and grouped by this hash.

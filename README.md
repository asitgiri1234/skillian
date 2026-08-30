# Skillian

Job scraper and resume matcher.

**Day 1 status:** the ingestion path only — fetch jobs from a source, normalise
them, dedupe, and store them with a full audit trail. Matching, embeddings, LLM
calls, API endpoints and UI are not built yet.

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

# 4. Schema
alembic upgrade head
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
  models.py          all 8 tables
  normalize.py       text normalisation + dedup hash (shared by all sources)
  ingest.py          fetch -> upsert -> record the run
  main.py            FastAPI app object (no routes yet)
  sources/
    base.py          JobSource ABC, SearchQuery, NormalizedJob
    adzuna.py        first implementation
scripts/run_ingest.py
alembic/             migrations
```

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

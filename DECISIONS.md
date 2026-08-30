# Decisions — Day 1 (ingestion path)

Every significant choice made building the ingestion pipeline, what was rejected,
and why. Scope was deliberately limited to ingestion: no matching, embeddings,
LLM calls, API endpoints or UI.

---

## 1. Architecture

### 1.1 Sources are pure functions of a query, not database writers

`JobSource.fetch(query) -> list[NormalizedJob]`. A source performs HTTP, parses,
and returns pydantic objects. It never touches SQLAlchemy, the session, or the
`jobs` table.

- **Rejected:** letting each source write its own rows. That is how per-source
  drift starts — one source computes `dedup_hash` differently, another forgets
  `fetched_at`, and the table becomes untrustworthy.
- **Why it matters:** a source is testable against a recorded HTTP response with
  no Postgres running. Every Adzuna test in this build ran with a
  `httpx.MockTransport` and zero network access.

### 1.2 `NormalizedJob` mirrors the `jobs` table one-to-one

The contract is the table shape, so `ingest.py` maps fields mechanically and
there is no translation layer to keep in sync.

- **Rejected:** a looser "whatever the source has" dict. It pushes per-source
  conditionals into `ingest.py`, which is exactly what the ABC exists to prevent.

### 1.3 `dedup_hash` is computed in `base.py`, not in each source

It is a `@property` on `NormalizedJob` calling `app/normalize.py`. A source
*cannot* supply its own value.

- **Why:** the hash is only useful if two different providers produce an
  identical digest for the same posting. If a source could override it, that
  guarantee is gone. Verified end-to-end: two fake boards with
  `"Acme Pvt. Ltd."` / `"ACME Private Limited"` and
  `"Bengaluru, Karnataka"` / `"Bengaluru, KA, India"` produced the same hash.

### 1.4 `app/normalize.py` is a separate module

The one structural addition beyond the requested skeleton.

- **Why:** normalisation is shared by every current and future source and by any
  later re-hashing/backfill job. Putting it in `sources/base.py` would make an
  import of the ABC drag in text processing; putting it in `ingest.py` would make
  it unavailable to anything that is not an ingestion run.

### 1.5 Adding a source touches exactly two places

A new file in `app/sources/`, plus one line in `SOURCE_REGISTRY` in
`scripts/run_ingest.py`.

- **Honest caveat:** the brief said "zero changes outside its own file". A
  registry line is unavoidable unless sources are auto-discovered by scanning the
  package. I rejected auto-discovery: implicit import-time registration makes
  failures (a typo'd module, a source that raises on import) hard to trace, and
  the explicit dict is greppable. No *logic* outside the source file changes —
  the registry is one dictionary entry.

---

## 2. Database and schema

### 2.1 Identity is `UNIQUE(source, source_job_id)`, not `dedup_hash`

`dedup_hash` is indexed but **not** unique.

- **Why:** the same job legitimately appears on several boards, each with its own
  `apply_url`. Making `dedup_hash` unique would throw away rows a user needs (you
  cannot apply through a link you deleted). Instead every copy is stored and
  `dedup_hash` groups them — the UI can collapse the group and offer all links.
- **Rejected:** `UNIQUE(dedup_hash)`. Loses apply URLs, and any hash collision
  between two genuinely different jobs would silently drop a real posting.

### 2.2 UUID primary keys generated in Python

`default=uuid.uuid4`, not `gen_random_uuid()`.

- **Why:** no dependency on `pgcrypto`/`uuid-ossp` being installed, and the
  application knows a row's id before INSERT.
- **Rejected:** bigserial. Integer ids leak row counts and make merging data from
  multiple environments painful.

### 2.3 `Numeric`, never `Float`, for salaries and scores

`Numeric(14,2)` for money, `Numeric(5,4)` for scores, `Numeric(4,1)` for years.

- **Why:** binary floats accumulate representation error, and salary is a number
  users compare. `_to_decimal` also converts via `str()` first, because
  `Decimal(1200000.0)` would carry the float's error into the column.

### 2.4 `salary_raw` is always kept alongside the parsed numbers

- **Why:** parsing is best-effort. When the parse is wrong or absent, the raw
  string is the only thing that can be shown without inventing a figure. Adzuna
  supplies no display string, so `_build_salary_raw` synthesises one — and marks
  it `(estimated)` when `salary_is_predicted` is set, because that number is
  Adzuna's model output, not something an employer published. Showing a predicted
  salary as fact would be the single most damaging bug in this pipeline.

### 2.5 `salary_period` stored, not normalised to annual

- **Why:** converting hourly to annual needs an assumed hours/week that varies by
  contract and country. A wrong salary is worse than an unnormalised one. Adzuna
  happens to annualise everything, so the field is `"year"` there — but the
  column exists so the next source is not forced to lie.

### 2.6 All eight tables created on day 1

As instructed. Worth recording *why* it is right: the initial migration then
describes the complete shape, and days 2–5 write code against existing tables
instead of stacking migrations onto a half-built schema.

### 2.7 `skills.aliases` as `text[]`, not a join table

- **Why:** aliases are a small, read-mostly bag of strings always fetched with
  their skill and never queried independently. A join table would add a migration
  and a JOIN for no gain. Reversible if aliases ever need their own metadata.

### 2.8 Free-text status/requirement/source columns, not enums

`ingestion_runs.status`, `job_skills.requirement`, `resume_skills.source`.

- **Why:** a Postgres enum needs a migration to add a value. Adding a skill
  extractor or a run status should not be a schema change on a 5-day build.
- **Trade-off accepted:** no database-level validation. The constants live in
  `ingest.py` instead.

### 2.9 Four indexes beyond those specified

`ix_resumes_user_id`, `ix_resume_skills_skill_id`, `ix_job_skills_skill_id`,
`ix_matches_resume_id_overall_score`.

- **Why:** Postgres does *not* index foreign keys automatically, and every one of
  these covers a query the app will certainly make (a user's resumes; reverse
  skill lookups; a resume's matches ranked by score). Composite PKs already cover
  the forward direction, so only the reverse column is indexed.
- **Caught by verification:** `alembic check` initially reported drift because
  these existed in the migration but not on the models. Both now agree —
  `alembic check` exits clean.

### 2.10 `ingestion_runs.resume_id` is `ON DELETE SET NULL`

- **Why:** every other FK cascades, but deleting a resume must not erase the
  audit trail of fetches that already happened. The run is a record of work done,
  not a child of the resume.

---

## 3. Upsert and duplicate detection

### 3.1 `INSERT ... ON CONFLICT DO UPDATE`, not SELECT-then-INSERT

- **Why:** one atomic statement. Two ingestions racing on the same posting cannot
  both conclude it is new and trip the unique constraint. SELECT-then-INSERT has
  a race window between the two statements.

### 3.2 New-vs-duplicate detected with `RETURNING (xmax = 0)`

- **Why:** Postgres sets `xmax` to 0 on a fresh insert and to the updating
  transaction's id on a conflict update. This is the only way to learn which
  branch fired **without a second round-trip per job**. The alternative — SELECT
  first to check existence — is one extra query per job and reintroduces the race
  from 3.1.
- **Trade-off:** it relies on a Postgres implementation detail rather than a
  documented API. Accepted because the project is Postgres-only (pgvector is a
  hard dependency), and it is verified: run 1 reported 3 NEW, run 2 on identical
  input reported 3 duplicate with the row ids unchanged.

### 3.3 The upsert refreshes content but never overwrites `embedding`

`_UPSERT_FIELDS` deliberately excludes `id`, `source`, `source_job_id` and
`embedding`.

- **Why:** an embedding costs an API call. A re-fetch nulling it out would mean
  paying to regenerate it on every ingestion run. Verified: a job with a stored
  embedding kept it through an upsert that changed its title.
- **Known limitation, day 3's problem:** if a description changes, the preserved
  embedding is stale. The fix is to compare a content hash and clear the embedding
  only when the text actually changed — deliberately not built today, since
  nothing generates embeddings yet.

---

## 4. Normalisation rules

### 4.1 Only legal-form suffixes are stripped

`inc, ltd, limited, pvt, private, llc, gmbh, pte, bhd, …` — but **not**
`technologies`, `labs`, `group`, `systems`.

- **Why:** legal forms are noise ("Acme Pvt Ltd" and "Acme Private Limited" are
  one company). Descriptive words are identity — stripping them would collide
  "Acme Technologies" with "Acme Financial".

### 4.2 Suffixes stripped repeatedly, never down to nothing

The loop stops at one remaining token.

- **Why:** "Acme Pvt. Ltd." needs two strips. But a company genuinely named "Co"
  must not normalise to the empty string, which would collide with every other
  company that normalised to empty.

### 4.3 Fields joined with a NUL byte before hashing

- **Why:** plain concatenation makes `("ab","c")` and `("a","bc")` hash
  identically. A separator that cannot occur in the normalised text removes the
  ambiguity. Explicitly tested.

### 4.4 Unicode NFKD before casefolding; `casefold()` not `lower()`

- **Why:** "Café" typed two different ways (precomposed vs combining accent) must
  produce one hash. `casefold()` is the correct operation for caseless matching —
  `lower()` mishandles e.g. German ß.
- **Detail:** hyphens and slashes become spaces rather than being deleted, so
  "full-stack" becomes "full stack", not "fullstack".

### 4.5 City = first comma-separated segment

- **Why:** job boards order location components most-specific-first
  ("Bengaluru, Karnataka, India"). This is a heuristic and it is the weakest link
  in the hash — a board that orders them the other way would break it. It is
  isolated in one function (`extract_city`) so it can be replaced with a
  per-source override without touching the hash rule itself.

---

## 5. Adzuna source

### 5.1 Currency inferred from the endpoint country

- **Why:** Adzuna partitions its API by country and returns salary figures in
  that country's currency **without a currency field**. The country code is the
  only available signal. An unknown country yields `None` rather than a guess —
  a mislabelled currency is worse than an absent one.

### 5.2 Retries only on transient statuses

`408, 425, 429, 500, 502, 503, 504` plus transport errors. A 401 or 400 raises
immediately.

- **Why:** retrying a bad API key just sleeps four times before failing with the
  same error, and hides a config problem behind a delay. Verified: a 401 raises
  after exactly one request; a 503 sequence retried and then succeeded.

### 5.3 Exponential backoff with full jitter, honouring `Retry-After`

- **Why:** `Retry-After` is the server telling you exactly when to come back.
  Full jitter (`uniform(0, delay)`) rather than fixed backoff because several
  sources retrying in lockstep would otherwise re-collide on the same schedule.
  Both the retry delay and `Retry-After` are capped so a hostile header cannot
  stall the run indefinitely.

### 5.4 A malformed record is skipped, not fatal

Only `id` and `title` are required. Missing company, salary, location or a
null nested object yields `None` fields.

- **Why:** one bad record must not discard the 49 good ones on the page. Adzuna
  really does send explicit `null` for nested `company`/`location` objects, which
  is why the code uses `(raw.get("company") or {})` — chained `.get()` on `None`
  would raise. Verified with a fixture containing a null-company, null-location
  and a title-less record.

### 5.5 Pagination stops on a short page, an empty page, or `max_results`

- **Why:** three independent stop conditions, so a provider quirk in any one of
  them cannot cause an infinite loop. `max_pages` is a final backstop. A short
  page ends pagination without spending a request confirming the next one is
  empty — verified as 1 HTTP call for a 3-result response.

### 5.6 HTML stripped from titles and descriptions

- **Why:** Adzuna returns markup and HTML entities. The description text is what
  will be embedded on day 3; `<strong>` tags would become meaningless tokens.

### 5.7 Experience parsed with ordered regexes and a plausibility gate

Ranges are matched before the bare-number fallback; results outside `0 < y <= 15`
are rejected.

- **Why:** "3-5 years" must yield 3, not be read by a looser pattern first. The
  ceiling rejects "40 years in business", which is a company fact, not a
  requirement. `experience_raw` keeps the matched phrase so a bad parse is
  auditable against what the posting actually said.

### 5.8 Remote detection is a keyword heuristic with negation handling

- **Why:** Adzuna has no remote flag. `"not remote"` / `"no remote"` are checked
  first, because a plain keyword search flags "this role is not remote" as remote.
  Acknowledged as imperfect — it is a heuristic over free text, stored in a
  boolean the user can override later.

### 5.9 The `httpx.Client` is injectable

`AdzunaSource(settings=..., client=...)`.

- **Why:** it is what made every Adzuna test in this build runnable with no
  network and no credentials. Ownership is tracked so `close()` only closes a
  client the source created.

---

## 6. Run recording and failure handling

### 6.1 The run row is committed *before* any network call

- **Why:** if the process is killed mid-fetch, the row is already durable and the
  abandoned run is visible as `status="running"`. A run stuck in `running` is
  itself a diagnostic signal.

### 6.2 Three terminal statuses, not two

`success`, `partial` (some sources failed, some jobs stored), `failed`.

- **Why:** with one source they collapse, but from day 2 onward "Adzuna worked,
  the other board 500'd" is a materially different outcome from total failure,
  and the difference must survive in the database.

### 6.3 A failing source does not abort the others

Caught per-source inside the loop, recorded in `source_errors`.

- **Why:** one provider's outage should not cost you the other provider's
  results. Verified: with one raising source and one good source, the good
  source's job was stored and the run finished `partial`.

### 6.4 Failures are recorded on a **fresh** session

`_record_failure` opens a new session rather than reusing the one that raised.

- **Why:** this is the subtle one. After a database error the session sits in an
  aborted transaction where every further statement fails with "current
  transaction is aborted" — so the obvious implementation would lose exactly the
  error message it was trying to save. Verified with a genuine constraint
  violation: the run was still recorded as `failed` with its error text.

### 6.5 Errors are recorded, then re-raised

- **Why:** the brief says never let an exception escape *without recording it* —
  not to swallow it. Silently succeeding on a failed run is worse than crashing.
  The CLI catches at its boundary and returns exit code 1 without a traceback.
  Confirmed across all scenarios: zero runs left in `running`.

### 6.6 `repr(exc)`, not `str(exc)`, and truncated to 4000 chars

- **Why:** several httpx/psycopg exceptions stringify to `""`, which would store
  an empty error on a failed run. Truncation keeps a driver traceback from
  bloating the column — it is for triage, not log storage.

### 6.7 Jobs are committed before the run row is finalised

- **Why:** a failure while writing the summary must not roll back the jobs that
  were actually fetched and stored.

---

## 7. Tooling

### 7.1 psycopg 3 (`postgresql+psycopg://`), not psycopg2

- **Why:** actively developed, SQLAlchemy 2.x's preferred Postgres driver, and it
  supports async without swapping drivers when FastAPI endpoints arrive.

### 7.2 `pgvector/pgvector:pg16` image

- **Why:** the extension is pre-built for the matching major version. Building it
  into the official `postgres` image needs a custom Dockerfile and a compile step
  on every fresh machine. `CREATE EXTENSION` appears in both the Compose init
  script and the migration, so a database provisioned outside Compose also works.

### 7.3 Compose host port is `${POSTGRES_PORT:-5432}`

- **Why:** changed during this build after 5432 turned out to be occupied by
  another project on this machine. Hardcoding it makes the project fail to start
  for anyone in that situation.

### 7.4 Alembic reads the URL from `app.config`, not `alembic.ini`

`sqlalchemy.url` is left blank in the ini.

- **Why:** one source of truth, and no credential ever lands in a committed file.

### 7.5 The migration hardcodes `EMBEDDING_DIM = 1536` instead of importing it

- **Why:** a migration must describe the schema *as it was at that revision*.
  Importing the constant would let a later model change silently rewrite history,
  so the migration would no longer reproduce the schema it created.

### 7.6 `expire_on_commit=False` on the session factory

- **Why:** the CLI prints job details after the transaction closes; the default
  would fire a lazy reload per attribute, or raise once the session is gone.

### 7.7 `pool_pre_ping=True`

- **Why:** ingestion runs are long and mostly idle waiting on HTTP. A pooled
  socket dropped by the server in the meantime would surface as a random
  `OperationalError` mid-run.

### 7.8 The CLI table is hand-rolled, not `tabulate`/`rich`

- **Why:** ~25 lines of formatting against a dependency the project has no other
  use for. Columns size to their widest actual value and truncate with an
  ellipsis.

### 7.9 The CLI exits non-zero on `partial` or `failed`

- **Why:** it makes the command usable in a cron job or CI step without parsing
  its stdout.

### 7.10 `get_settings()` is `lru_cache`d

- **Why:** pydantic-settings re-reads and re-parses `.env` on every
  instantiation, and the CLI, orchestrator and every source ask for settings.

### 7.11 Adzuna credentials are optional in `Settings`

- **Why:** `import app.config` must work in a test run that never makes a network
  call. `AdzunaSource` raises `AdzunaConfigError` with a link to the signup page
  when they are actually needed — a clear message beats a validation error at
  import time in an unrelated test.

---

## 8. Scope

### 8.1 `app/main.py` declares the FastAPI app and no routes

- **Why:** the brief asked for FastAPI structure only and explicitly excluded API
  endpoints. The module exists so deployment and the import graph are settled. I
  did not add a `/health` route — it would have been useful, but it is still an
  endpoint, and the instruction was explicit.

### 8.2 No tests committed

- **Honest disclosure and the one thing I would change.** The code was verified
  by three throwaway scripts run against real Postgres and a mocked Adzuna
  (unit-level normalisation and parsing, end-to-end ingestion including all
  failure paths, and the CLI). Those scripts were written to a scratch directory
  and are not in the repo, because a `tests/` tree was not in the requested
  scope and pytest is not in `requirements.txt`.
  **Recommendation:** promote them to `tests/` on day 2 before the surface area
  grows — the fixtures already exist and the failure-path coverage is the part
  that will be expensive to recreate later.

### 8.3 Not built, by instruction

Matching, embeddings, LLM calls, API endpoints, UI. `jobs.embedding` and
`resumes.embedding` columns exist and stay `NULL`.

---

## 9. Verification performed

Everything below was executed, not assumed.

| Check | Result |
|---|---|
| All modules import; 8 tables registered | pass |
| `alembic upgrade head` against Postgres 16 + pgvector | pass |
| `alembic check` (models vs migration drift) | pass, after fixing 4 undeclared indexes |
| `alembic downgrade base` → re-`upgrade head` | pass (9 tables → 1 → 9) |
| Normalisation: suffixes, accents, punctuation, NUL separator | pass |
| Adzuna parsing: HTML, salary, experience, dates, missing fields | pass |
| Adzuna retry on 503; immediate raise on 401; config error on no creds | pass |
| Pagination stop conditions | pass |
| Upsert: 3 NEW then 3 duplicate, row ids stable | pass |
| Upsert preserves `embedding`, refreshes content | pass |
| Run statuses: success / partial / failed, all with `finished_at` | pass |
| Exception escapes but is always recorded; zero runs stuck in `running` | pass |
| Cross-source dedup: 2 boards → same hash, both rows kept | pass |
| CLI: `--query`/`--location`, table output, NEW vs duplicate, exit codes | pass |

**Not verified:** a live Adzuna API call. No real credentials were available, so
every Adzuna test ran against `httpx.MockTransport` with fixtures shaped to the
documented response. The response-shape assumptions (field names, `"0"`/`"1"`
string for `salary_is_predicted`, ISO-8601 `created`, country-implied currency)
are the thing to confirm first with a real key.

# Decisions

# Day 1 (ingestion path)

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

---

# Decisions — Day 2 (local model providers)

Ollama for both LLM and embeddings, behind provider abstractions. Still no
matching, no API endpoints, no UI.

---

## 10. Provider abstraction

### 10.1 Two separate ABCs, not one "AI provider"

`LLMProvider` and `EmbeddingProvider` live in different modules with different
error hierarchies.

- **Why:** they fail differently and scale differently. An embedding call is
  ~100ms and batches; a 7b extraction is ~70s and does not. Fusing them would
  force one timeout, one retry policy and one set of errors onto two operations
  that share nothing but a transport. They are also independently swappable —
  local embeddings with a hosted LLM is a reasonable configuration, and the
  separate `EMBEDDING_PROVIDER` / `LLM_PROVIDER` flags allow it.

### 10.2 `complete()` returns a plain `dict`, and deliberately does not validate

The signature is `complete(prompt, schema) -> dict`, exactly as specified. The
provider passes `schema` to the decoder but never calls `model_validate`.

- **Why:** a provider that validated would have to own a retry policy, and that
  policy belongs with the caller that knows what a good answer looks like.
  `structure.py` retries with error feedback; a future summariser might accept a
  partial result. Keeping the provider dumb keeps that choice open.
- **Verified:** `complete()` returns `{"skills": [], "experience": []}` — an
  object that fails `ParsedResume` validation — without raising.

### 10.3 Explicit registry dicts, not package scanning

`LLM_PROVIDERS` / `EMBEDDING_PROVIDERS` in `app/providers/__init__.py`.

- **Why:** same reasoning as `SOURCE_REGISTRY` on day 1. A typo in
  `LLM_PROVIDER` should produce "not registered, available: ollama", not an
  `ImportError` from a mistyped module name. Flags are matched
  case-insensitively.

### 10.4 Three error types, split by who can fix them

`LLMUnavailableError` (operator: daemon down, model not pulled) vs
`LLMResponseError` (backend replied with junk); same split for embeddings, plus
`EmbeddingDimensionError`.

- **Why:** it determines whether retrying is sensible. `structure.py` retries
  validation failures but lets `LLMUnavailableError` propagate immediately — a
  stopped daemon will not fix itself, and retrying only delays a clear message.
- The 404 case is mapped specially because "model not pulled" is by far the most
  common first-run failure. The error carries the literal fix:
  `ollama pull qwen2.5:7b`. A connection error suggests `ollama serve`.

### 10.5 `EmbeddingDimensionError` exists as its own type

- **Why:** a width mismatch otherwise surfaces as an opaque pgvector error at
  INSERT, far from the cause. Catching it at the provider boundary names the
  real problem (wrong model configured) at the point it can be understood.

### 10.6 `embed_batch` is concrete on the ABC, overridden by Ollama

The default loops over `embed()`; `OllamaEmbeddingProvider` overrides it because
`/api/embed` accepts a list.

- **Why:** a new provider only has to implement one method, but the one that has
  a real batch endpoint uses it. Measured: 3 texts in one round-trip took 0.104s
  against ~2s for a single cold call — the round-trip dominates completely.

### 10.7 Clients are injectable

`OllamaLLMProvider(settings=..., client=...)`, matching `AdzunaSource` from
day 1.

- **Why:** it is what let every error-mapping path (404, connection refused,
  non-JSON reply) be tested with no daemon and no model.

---

## 11. Embedding model and the 768 migration

### 11.1 `EMBEDDING_DIM` 1536 -> 768, in `models.py` before generating the migration

As instructed, and the ordering matters: autogenerate diffs the models against
the live database, so changing the constant first is what produces the migration
at all.

### 11.2 Migration `0002` nulls existing vectors before altering the column

- **Why (mechanical):** Postgres cannot cast `vector(1536)` to `vector(768)`;
  the ALTER fails outright on any non-null row.
- **Why (the real reason):** an embedding from a *different model* is not
  comparable to one from nomic-embed-text. Distances between them are
  meaningless, so the vectors must be regenerated regardless. Truncating them to
  768 would produce plausible-looking garbage — numerically valid, semantically
  worthless. The downgrade is symmetric for the same reason.
- Safe today because nothing generates embeddings yet, so every row is already
  NULL. Written explicitly so the migration stays correct when that changes.

### 11.3 Alembic autogenerate produced a broken migration; fixed at the source

The generated file referenced `pgvector.sqlalchemy.vector.VECTOR` with **no
matching import** — it would have died with `NameError` on first run.

- **Fix:** a `render_item` hook in `alembic/env.py` that renders `Vector(dim)`
  and registers the import. Patching the one generated file would have left the
  same trap for every future autogenerate; this fixes the generator.
- **Caught by:** reading the generated migration instead of trusting it.

### 11.4 The dimension check is in `check_ollama.py`, not at import time

- **Why:** `EMBEDDING_DIM` and the provider's `dimension` must agree, but
  asserting it at import would mean every `import app.models` needs a running
  Ollama. The preflight script is where that check belongs — it already requires
  the daemon.

---

## 12. Structured extraction

### 12.1 The schema is passed as `format=`; the prompt never mentions JSON

`ollama.chat(format=ParsedResume.model_json_schema(), options={"temperature": 0})`.

- **Why:** constrained decoding makes malformed output *unrepresentable* at the
  token level, rather than merely discouraged by an instruction the model may
  ignore. Instructions about formatting would be redundant tokens that also
  invite the model to editorialise ("Here is the JSON you asked for:").
- **Verified:** the system and user prompts are asserted to contain no mention
  of JSON.

### 12.2 `temperature=0`

- **Why:** extraction is not a creative task. The same resume must produce the
  same parse, or caching and the retry loop both become meaningless.
- **Consequence that had to be handled:** at temperature 0 a plain retry is
  deterministic and returns the identical bad answer. That is precisely why the
  retry feeds the validation errors back into the next prompt — changing the
  input is the only thing that changes the output. Without that, the retry loop
  would be three identical failures.

### 12.3 The validate-and-retry loop is kept, and its validators check *meaning*

Constrained decoding guarantees shape, not accuracy — so type checks would be
redundant with the grammar. Every validator here targets something the grammar
cannot express:

- **Placeholders**, including the literal string `"string"`. This is not
  hypothetical: it is copied straight out of the schema's own vocabulary, and
  constrained decoding makes it *more* likely, not less.
- **Fake emails** — `@example.com`, or anything without a plausible domain. A
  missing email is fine; a wrong one is worse than nothing.
- **Duplicate/junk skills** — JSON schema can say "array of strings", not "no
  duplicates, no sentences".
- **Implausible years** — a graduation year of 12 is a page number.
- **The empty parse** — a well-formed object with no skills and no experience.
  This is the single most valuable check: without it the pipeline would store an
  empty resume and every downstream match would be garbage.

**Verified:** the loop recovers on attempt 2 when attempt 1 returns an empty
parse, feeds the failure text into the retry prompt, and raises `StructureError`
carrying the last errors after exhausting attempts.

### 12.4 Every field is required-but-nullable — the day's most important finding

This was found by testing, not by reasoning, and it is worth defending carefully.

Pydantic only lists a field as `required` in its JSON schema when the field has
**no default**. With `= None` defaults everywhere, `required` was empty, the
generated grammar made every key optional, and the model took the cheapest legal
path: `qwen2.5:7b` returned a valid object containing `skills`, `experience`,
`education` and `total_years_experience` — and silently omitted `name`, `email`,
`phone`, `location` and `summary`, plus `company` inside every role. The
preflight reported `name: None`, `email: None` on a resume that plainly contains
both.

The fix is to give the fields no default while keeping the type nullable
(`str | None` with a bare `Field(description=...)`). The key must then be
emitted, but `null` remains a legal value. "I looked and there is nothing here"
is a real answer; a missing key is not.

- **Rejected:** adding `minItems: 1` to `skills`. It would force the grammar to
  emit at least one skill even for a document that is not a resume, converting a
  detectable failure into a confident hallucination. The empty-parse validator
  catches that case honestly instead.
- **Generalisable lesson:** with constrained decoding, the schema *is* the
  prompt. Anything optional in the schema is an invitation to omit.

### 12.5 Dates stay free text

`start_date: str | None`, not a date type.

- **Why:** resumes write dates a dozen ways ("Jan 2020", "2020-01", "Spring
  2020", "Present"). Forcing an ISO date makes the model guess, and a wrong
  parsed date is worse than the original string. Normalising is a later problem
  that can be done offline from the preserved text.

### 12.6 Input truncated at 24,000 chars

- **Why:** qwen2.5 has a 32k context, but a long resume plus the schema plus
  retry feedback can still crowd it. Explicit truncation with a warning beats
  silent server-side truncation that would drop the education section without
  saying so.

### 12.7 `ParsedResume` is not yet persisted

`resumes.parsed` (JSONB) and `resumes.embedding` exist and stay empty.

- **Why:** the brief scoped this session to providers plus extraction. Wiring
  extraction into a resume-upload path is day 3, and doing it now would mean
  guessing at that path's shape.

---

## 13. Models chosen

### 13.1 `qwen2.5:7b` for extraction, not 3b

- **Why:** measured, not assumed. `qwen2.5:3b` failed extraction three times in
  a row on the sample resume — it returned objects with no skills and no
  experience at all. 7b succeeds on the first attempt with all 8 skills, both
  roles and correct contact details. Extraction quality is the bottleneck for
  everything downstream, and locally a bigger model costs time rather than money.
- **Cost recorded:** ~72s per extraction on this machine (CPU). That is slow
  enough that batch resume processing will need to be a background job, not a
  request-time operation — worth knowing before endpoints get designed.

### 13.2 `nomic-embed-text` for embeddings

- 768 dimensions, 274 MB, ~0.03s per text warm.
- **Sanity-checked, not assumed:** cosine similarity of "python backend" to
  "python backend engineer" is 0.85, against 0.50 for "react frontend". The
  ordering is what the matching stage will depend on.

### 13.3 Ollama for both

- **Why:** no API cost and no data leaving the machine — resumes are personal
  data, which makes local inference a privacy property, not just a budget one.
- **Trade-off accepted and stated:** ~72s per extraction is far slower than a
  hosted model. The abstraction is what makes that reversible: swapping to a
  hosted LLM is a new file in `app/providers/` plus a config flag, with no
  caller changes.

---

## 14. Verification performed (day 2)

Ran against the real Ollama daemon and the real database.

| Check | Result |
|---|---|
| `check_ollama.py` full run, exit 0 | pass |
| Migration `0002` up, `vector(768)` on both columns | pass |
| `0002` downgrade -> `vector(1536)` -> re-upgrade -> `vector(768)` | pass |
| `alembic check` (no model/migration drift) | pass |
| Autogenerated migration missing pgvector import | found and fixed via `render_item` |
| Real embedding: 768 dims, matches `EMBEDDING_DIM` | pass |
| Embedding semantics: 0.85 related vs 0.50 unrelated | pass |
| `embed_batch` = one round-trip (0.104s for 3) | pass |
| Empty-text and dimension-mismatch guards | pass |
| Real extraction: name, email, location, 8 skills, 2 roles, education | pass |
| Schema passed as `format=`; prompts never mention JSON | pass |
| Retry loop recovers on attempt 2; feeds errors back | pass |
| `StructureError` after exhausted attempts, carries errors | pass |
| All accuracy validators (placeholder, email, dedup, year, empty) | pass |
| Registry: valid flags, case-insensitivity, unknown flag lists options | pass |
| Error mapping: 404 -> pull hint, conn refused -> `ollama serve`, non-JSON | pass |
| Day 1 suites re-run after the dimension change | pass, no regressions |

**Still not verified:** a live Adzuna API call (unchanged from day 1 — no
credentials). Extraction is verified against exactly one synthetic resume; real
resumes are messier (multi-column PDFs, tables, headers) and the 24k truncation
and date handling have not been exercised against them.

**Still no committed tests.** The day 2 work was verified by three more scratch
scripts. This is now two days of accumulated verification living outside the
repo, and the case for promoting it to `tests/` is stronger than it was
yesterday — the provider fakes and the retry-loop fixtures are exactly what
would be expensive to reconstruct later.

---

# Decisions — Day 3 (matching pipeline and async search)

Schema for chunk-level embeddings, the scorer, the eight-stage search pipeline,
background execution and the HTTP surface. The two questions this day is
organised around: *where is it legitimate to spend a local-model call*, and
*what does a score mean when the inputs are missing*.

---

## 15. Schema changes

### 15.1 `jobs.embedding` dropped; `job_chunks` replaces it

- **Why:** a job description is several documents stapled together — a company
  blurb, responsibilities, hard requirements, nice-to-haves, benefits, and an
  equal-opportunity paragraph that is word-for-word identical across thousands
  of postings. One vector over all of it is dominated by whichever section is
  longest, which is almost never the one a candidate should be matched on.
  Worse, the shared boilerplate pulls every posting toward the same centroid, so
  a payments backend role and a QA role end up closer to each other than either
  is to a resume.
- Chunking lets the strongest *passage* speak for the job, which is what
  `semantic_component`'s top-3 mean consumes.
- **No data migration.** An existing `jobs.embedding` cannot be split into chunk
  vectors after the fact — the chunk text it would have belonged to was never
  stored. Affected jobs are simply re-chunked on the next search, which the
  pipeline already does for any job with no `job_chunks` rows. The downgrade is
  symmetric and equally lossy, and says so.

### 15.2 `job_chunks.embedding` is NOT NULL, unlike `resumes.embedding`

- **Why:** a resume genuinely exists before it is embedded — it is created by an
  upload and embedded afterwards. A chunk has no such state: chunking and
  embedding happen in the same pipeline stage, and a chunk row is only ever
  written because it was embedded. A nullable column would invent a state the
  code cannot produce, and every reader would then have to handle it.

### 15.3 `UNIQUE(job_id, chunk_index)` rather than just an index

- **Why:** it makes re-chunking idempotent. The pipeline can `ON CONFLICT DO
  NOTHING` a whole batch without first deleting, so a run that dies partway
  through stage e and is retried rewrites nothing and duplicates nothing.

### 15.4 `ingestion_runs.stage` is free text, not an enum

- Consistent with 2.8. `status="running"` on its own tells a polling client
  nothing about whether to keep waiting; `stage` is what turns a four-minute
  blank wait into visible progress.
- The ordered `STAGE_ORDER` tuple in `app/runs.py` is what makes it renderable
  as "step 4 of 10" without every client hardcoding the list.

### 15.5 No ANN index on `job_chunks.embedding`

- **Why:** nothing does a nearest-neighbour *query*. Scoring loads the chunk
  vectors for the jobs a run already fetched and compares them in Python, which
  is the right shape when the candidate set is a few hundred rows the pipeline
  just wrote. An `ivfflat`/`hnsw` index would cost build time and write
  amplification for a query pattern that does not exist yet.
- It becomes necessary the moment there is a "search all stored jobs" path
  rather than "score the jobs this run fetched". That is the trigger to watch
  for.

---

## 16. Scoring

### 16.1 The scorer is pure; the pipeline is where I/O lives

- `app/matching/scorer.py` imports no ORM, no HTTP client and no provider. It
  takes frozen dataclasses and returns a `ScoreResult`.
- **Why:** two payoffs. It can be tested exhaustively with no database (46 of
  the 138 committed tests, running in 0.4s), and it is fast enough to run over
  every job in a search. The moment an LLM call enters that file, a 200-job
  search goes from seconds to roughly 25 minutes.

### 16.2 Semantic score is the mean of the top 3 chunks — not the max, not all

- **Against max:** every job description contains at least one paragraph of
  generic engineering prose that any technical resume scores well against. Max
  rewards a single lucky chunk, so the ranking ends up sorting on "which posting
  happened to contain the most generic paragraph".
- **Against mean-over-all:** it dilutes a genuinely strong requirements section
  with the benefits boilerplate sitting next to it — precisely the problem
  chunking was introduced to solve. A job with one perfect requirements chunk
  and six paragraphs of legal text would score below a mediocre job with a short
  description.
- **Top-3 mean** requires three good passages, which means the *posting*
  matches rather than one sentence of it. Three because the median chunked
  description in testing produced 2-5 chunks; a larger k collapses into
  mean-over-all for most jobs, and k=1 is max.
- Fewer than three chunks uses all of them. Dividing by a fixed 3 would penalise
  short postings for being short, which is not a property of the candidate.

### 16.3 Cosine rescaling exists, and `COS_LO`/`COS_HI` are labelled as guesses

- **Why rescale at all:** cosine similarity between related English text does
  not use the [0, 1] range. Everything a resume is compared against is a job
  description — same language, same register, same subject matter — so scores
  cluster in a narrow band. Fed straight into the weighted sum, the semantic
  term is nearly constant across the result set: it shifts the absolute score
  but barely moves the *ranking*, so its nominal 40% weight buys 40% of nothing.
  Rescaling the observed band onto [0, 1] is what restores the spread.
- **Why p5/p95 and not min/max:** the extremes are one weird posting each.
  Anchoring to them puts every real job back in a narrow middle, which is the
  problem being solved. Clamping deliberately sacrifices the top and bottom 5% —
  those jobs are already unambiguously good or bad, and their exact ordering
  matters far less than resolution across the middle 90%.
- `scripts/calibrate_similarity.py` prints the real distribution, and the
  constants carry a comment saying they are placeholders. See section 19 for the
  measured values and why they are not yet committed.

### 16.4 Weighted recall, not Jaccard

- `earned / possible` over the job's requirements, each weighted by
  `SKILL_WEIGHTS`. A candidate is never penalised for knowing things the job did
  not ask for.
- **Why not Jaccard:** the denominator would include the resume's skills. A
  senior engineer with forty skills would score *worse* against a five-skill
  posting than a junior with exactly those five — same intersection, union eight
  times larger. That is the opposite of useful, and it is the single most common
  way this metric is got wrong. There is a test named for it
  (`test_is_recall_not_jaccard`).
- **Why not precision:** "what fraction of your skills does this job want" ranks
  narrow jobs above broad ones for no reason a candidate cares about.
- `required: 1.0` / `preferred: 0.4` — a preferred skill counts, but at well
  under half the weight, so missing a "nice to have" barely moves the score.

### 16.5 `None` for unparsed requirements; `1.0` for missing experience data

Two applications of one principle: **absence of data is not evidence of a
shortfall**, and the two cases must stay distinguishable in the output.

- `skill_component` returns `None`, not `0.0`, when a job has no parsed
  requirements. "We could not read this posting" and "this candidate matches
  nothing in it" mean entirely different things, and collapsing them buries
  every thin posting at the bottom of the results as though the candidate had
  been rejected on merit. `ScoreResult.skills_unparsed` carries it to the UI,
  which must say so — presenting a semantic-only score as a full match is a lie.
- The fallback is `semantic * multiplier`, **not** `semantic * W_SEMANTIC`.
  Multiplying by 0.4 would push every unparsed job below every scored one for a
  reason unrelated to fit; dividing by 0.4 to "renormalise" would push them all
  above. Letting semantic carry full weight is the least-wrong option, and it is
  what the flag exists to disclose.
- `experience_multiplier` returns 1.0 when *either* side is None. Most postings
  state no parseable experience bar, and resumes frequently state no total.
  Treating unknown as zero would penalise the majority of pairs for a gap in our
  extraction rather than a gap in the candidate.
- A multiplier rather than a subtracted term, floored at 0.70: being three years
  short should discount an otherwise-excellent match, not flatten it.

### 16.6 Explanations are capped at the top 20

- **Why:** this is the pipeline's one genuinely expensive per-item operation — a
  local 7b model producing prose, measured at 8-30 seconds per job on this
  machine. Scoring 200 jobs takes seconds; explaining 200 would take the better
  part of an hour, and 180 of those explanations would never be read.
- 20 is one page of results. Everything below it is reachable by paginating,
  which re-running a search would extend.
- The cap is enforced in the pipeline rather than in `explain.py`, so the
  explanation function stays a pure "explain this one match" and the policy
  lives where the ranking is known.
- **Consequence, stated plainly:** `GET /matches?offset=20` returns rows with
  `explanation: null`. That is a visible gap in the API, not a bug, and the fix
  when it matters is an on-demand explain endpoint rather than a larger cap.

### 16.7 A weighted sum, not a learned model

- 0.6/0.4 is a judgement, not a fitted parameter, and there is no labelled data
  to fit against — nobody has told this system which matches were good.
- Skills lead because that component is *actionable* ("you are missing
  Kubernetes" is advice; "your cosine is 0.61" is not) and *checkable* — a skill
  match is either there or it is not.
- Semantic keeps a substantial 40% because skill extraction is lossy: it catches
  named technologies and misses everything phrased as a sentence ("you have
  operated something you built").
- The weights sum to 1.0 and a test asserts it, because if they ever do not,
  `overall_score` silently leaves [0, 1] and no longer fits
  `matches.overall_score`'s `Numeric(5,4)`.

---

## 17. Pipeline and execution

### 17.1 Stage ordering keeps model calls out of the wide part of the funnel

The ordering *is* the design. Scoring (stage f) runs over every job and is pure
arithmetic. The two stages that call a model are both bounded: skill extraction
by the number of jobs not already extracted, explanation by a hard 20. A test
(`test_scoring_makes_no_model_calls`) asserts the call counts rather than
trusting the comment.

### 17.2 Each stage commits on its own

- **Why:** the entire point of the `stage` column is that a *different*
  connection — the one serving `GET /runs/{id}` — can read progress while the
  run is going. Buffered inside the pipeline's transaction it would become
  visible exactly when it stops being useful.
- Stages d and h also commit per item, because each item cost real model seconds
  and a crash on job 90 of 100 must not discard the 89 already paid for.

### 17.3 Work is skipped on "has no rows", not on "is new"

Stage d skips jobs that already have `job_skills`; stage e skips jobs that
already have `job_chunks`. Not `is_new`, because a job whose extraction failed or
was interrupted last run would then never get another chance, while a job already
processed would be re-read by the model on every subsequent search.

### 17.4 The pipeline dedupes on `dedup_hash`; day 1's CLI does not

- Day 1 deliberately stores one row per board (decision 3 — each copy has its own
  apply URL). The search pipeline deliberately keeps one per `dedup_hash`.
- Both are right for their context: showing a user the same job three times is
  bad, and each extra copy costs a full local-model skill extraction. Row
  identity in the table is still `(source, source_job_id)` — this only decides
  what a single *run* bothers to write.

### 17.5 `BackgroundTasks`, not Celery or RQ

- **Why:** a search run is one long function, on one machine, with no fan-out, no
  cross-process scheduling and no retry semantics worth speaking of. The state a
  broker would manage is *already* in `ingestion_runs`, updated stage by stage,
  and it survives a process restart in a way an in-memory Celery result does not.
  Adding Redis plus a worker process would mean two more services, a
  serialization boundary and a deployment story in exchange for nothing this
  workload needs.
- **What is genuinely given up, stated rather than hidden:** a task dies with the
  process, and there is no retry and no concurrency limit. The failure is visible
  rather than silent — the run row is left at `running` with a stale stage, which
  is exactly the signal day 1 established for an abandoned run.
- **The trigger to change:** more than one API process, or runs long enough that
  deploys routinely kill one. At that point the pipeline itself needs no changes,
  because `run_search` already takes every dependency by injection; only
  `workers.py` is replaced.

### 17.6 Providers are constructed in the worker, not the request handler

So that a dead Ollama daemon fails the *run* — recorded, pollable, with a message
naming the fix — rather than the POST that queued it.

---

## 18. Departures from the brief, and gaps it did not cover

Each of these is a place where the day-3 specification was silent, or where
following it literally would have caused a problem. Listed rather than absorbed.

### 18.1 `queries.py` did not exist; it was written, and it is LLM-free

The brief referenced it as an existing module. Query generation runs once per
search, so a model call would have been affordable — but it would have meant a
30-90 second wait before the first HTTP request went out, non-deterministic
queries across runs of the same resume, and no benefit, because a job board's
keyword search is a bag-of-words matcher that gains nothing from a fluent
phrase. It builds from the candidate's job titles (seniority words stripped,
since boards match them literally) and their most specific skills.

### 18.2 `app/matching/skills.py` is a module the brief did not list

Stage (d) — "extract + canonicalize skills into job_skills" — is two concerns: a
model call, and a database identity mapping. Neither belongs inline in the
orchestration. Canonicalisation is not optional detail: two jobs that both want
"React" must resolve to the same `skills.id`, or the skill component becomes set
intersection over strings differing by case or alias, silently returns nothing,
and makes every candidate look unqualified.

### 18.3 Job skill extraction has no retry loop, unlike resume extraction

An empty result is a *legitimate* answer here — plenty of postings list no
concrete requirements — and `skill_component` already handles that case
explicitly. Retrying would spend another local-model minute per job arguing with
a model that was right the first time. Resume extraction retries because an empty
parse there is unambiguously a failure.

### 18.4 `queued`/`succeeded` versus day 1's `success`

The brief specified `queued -> running -> succeeded`; day 1 verified
`running -> success`. Renaming either would break verified behaviour or diverge
from the specification, so **both vocabularies now share the column** and
`app/runs.py` owns them, with `is_terminal()` / `is_success()` predicates that
every reader goes through instead of comparing to a literal. `GET /runs/{id}`
exposes `is_terminal` for exactly this reason.

**This is a wart and it is worth removing.** The clean fix is a migration that
rewrites `success` to `succeeded` and deletes the day-1 constant. It was not done
today because it touches day 1's verified paths and belongs in its own change.

### 18.5 `POST /resumes` and `PATCH /resumes/{id}/skills` had to be written

The brief required `build_resume_embedding_text` to be used "on both", but day 2
stopped short of persisting a parsed resume at all, so neither endpoint existed.
They live in `app/api/resumes.py`, sharing one invariant: **whenever a resume's
skills or parse change, its embedding is rebuilt and its matches are dropped.** A
match scored against the old skill set is not stale, it is wrong, and nothing in
the `matches` table records which skill set produced it.

### 18.6 The resume embedding excludes education, not just its boilerplate

The brief named "address, hobbies and education boilerplate". Education is
excluded entirely: a degree title matches every posting's degree requirement
about equally, so it adds distance to no one while consuming a large share of the
embedded tokens. Dates are dropped for the same reason — "Jan 2020 - Present" is
noise under cosine.

### 18.7 `POST /resumes` is synchronous and slow (60-120s)

Unlike a search, there is nothing useful to return early: the resume's id is
worthless to a caller who cannot search on it yet, and `POST /searches` rejects
an unparsed resume with a 409. Uploading is also once-per-resume, where searching
is the repeated action. **This is the weakest endpoint in the surface**, and the
fix is the pattern already built next door — a run row plus `BackgroundTasks`.

### 18.8 No auth, no rate limiting, no CORS

Not in scope and not built. Stated because the surface is now large enough that
its absence matters: `GET /matches?resume_id=` returns anyone's matches to anyone
holding the UUID, and `POST /searches` queues unbounded background work for an
unauthenticated caller. This is a localhost-only service until that changes.

### 18.9 `LLMProvider` gained an abstract `complete_text`

Explanations are prose. Forcing them through `complete()` with a one-field
wrapper schema would constrain the decoder for no benefit and spend tokens on
JSON punctuation. Abstract rather than defaulted in terms of `complete()`, since
every real backend has a plain completion call and a default would let a future
provider silently inherit a worse one.

---

## 19. Verification performed (day 3)

**The suite is finally in the repo.** Two days of decisions recommended it; day 3
committed it. 193 tests, `pytest` added to `requirements.txt`.

| Check | Result |
|---|---|
| Migration `0003` up: `job_chunks` created, `jobs.embedding` dropped | pass |
| `0003` downgrade -> re-upgrade | pass |
| `alembic check` (no model/migration drift) | pass |
| `tests/test_scorer.py` — 46 tests, no I/O | pass |
| `tests/test_chunking.py` — 25 tests, no I/O | pass |
| `tests/test_matching_units.py` — 30 tests (queries, explain, embedding text) | pass |
| `tests/test_pipeline.py` — 14 tests, real Postgres, faked models | pass |
| `tests/test_api.py` — 23 tests, real Postgres, TestClient | pass |
| `cosine_similarity` vs identical / orthogonal / opposite / zero vectors | pass |
| `skill_component` returns `None` (not `0.0`) for empty `job_skills` | pass |
| `experience_multiplier` returns 1.0 for a `None` requirement | pass |
| `semantic_component` with 1 and 2 chunks (fewer than top-k) | pass |
| Run transitions `queued -> running -> succeeded`, observed mid-run | pass |
| Scoring makes zero model calls (asserted on call counters) | pass |
| Explanations capped at 20 over a 25-job run, highest-scoring first | pass |
| Re-running: no re-extraction, no re-chunking, no duplicate rows | pass |
| Failure paths: partial, failed, exception recorded *and* re-raised | pass |
| No run left in a non-terminal state on any path | pass |
| `POST /searches` returns 202 in single-digit ms | pass |
| `PATCH .../skills` re-embeds and clears matches | pass |
| End-to-end with the **real** Ollama daemon (see below) | pass |

### 19.1 The end-to-end run against real models, and the defect it found

Because Adzuna credentials are still placeholders, the source was local, but
every model call was real: `qwen2.5:7b` for resume extraction, skill
classification and explanations, `nomic-embed-text` for all embeddings. Twelve
job descriptions spanning a deliberate relevance gradient (senior backend
payments through to enterprise SaaS sales) against one synthetic backend resume.

**The first run's ranking was wrong, and the reason is worth recording.**

Asked for a posting's skills, `qwen2.5` answers in the posting's own words,
because that is what the document says. It returned entries like:

    "Strong Python"
    "Comfortable with Docker"
    "FastAPI or Django in production"
    "5+ years of professional backend development"
    "Experience running services in production, including on-call"

Stored verbatim, none of those canonicalise onto the resume's `Python`,
`Docker`, `FastAPI`. So a candidate holding Python, FastAPI, Docker, Kubernetes
and Kafka scored **`skill = 0.065`, one requirement matched out of eight**,
against the senior backend posting she was an obvious fit for. And because a few
postings happened to yield short names ("Python", "SQL", "PyTorch"), the ranking
*inverted*: Machine Learning Engineer came first, Senior Backend Engineer third,
Backend Engineer fourth at `skill = 0.000`.

This is the failure mode that makes canonicalisation load-bearing rather than
tidy-up, and it is invisible without an end-to-end run: every unit test passed,
because the scorer was doing exactly what it was told with the ids it was given.
The `skills` table had filled up with 94 rows of requirement prose.

**The fix has two halves,** because either alone is unreliable:

1. The prompt and the field description now demand bare canonical names, with
   explicit rewrites ("Write 'Python', not 'Strong Python'").
2. `clean_skill_name()` enforces it regardless of what the model returns —
   stripping stacked qualifiers, dropping durations and prose, splitting lists
   ("FastAPI or Django" is two skills), and rejecting phrases that name a
   discipline rather than a skill. It is applied to resume skills too: both
   sides of the intersection have to be cleaned the same way, or a resume saying
   "Strong Python" still misses a job saying "Python".

`tests/test_skill_names.py` pins all of it, using the model's real output as the
test cases.

**After the fix, the same twelve jobs against the same resume:**

| Job | skill before | skill after | rank before | rank after |
|---|---|---|---|---|
| Senior Backend Engineer | 0.065 (1/8) | **1.000 (7/7)** | 3 | **1** |
| Python Developer | 0.077 (1/7) | **0.871 (6/8)** | 2 | 2 |
| Backend Engineer, Platform | 0.000 (0/6) | **0.783 (6/7)** | 4 | 3 |
| DevOps Engineer | 0.000 (0/6) | 0.515 (4/9) | 7 | 4 |
| Machine Learning Engineer | 0.333 (3/9) | 0.333 (3/9) | **1** | **7** |
| Senior Frontend Engineer | 0.000 (0/10) | 0.000 (0/10) | 9 | 11 |
| Enterprise Sales Executive | 0.000 (0/6) | 0.000 (0/4) | 12 | 12 |

The extractor now returns `Python, PostgreSQL, Docker` as required and
`FastAPI, Django, Kafka, Kubernetes` as preferred for the senior backend
posting — the same requirements it read before, named canonically. Note that
Machine Learning Engineer's skill score did not change at all: it was never
wrong, it was simply the only job whose extraction happened to produce matchable
names, which is precisely what put it top of a broken ranking.

Also confirmed by the run:

- `build_resume_embedding_text` excluded phone, email, university and hobbies.
- Skill classification distinguished required from preferred using the posting's
  own "nice to have" / "bonus" language, and got it right on every sample.
- Explanations named specific held and missing skills without inventing any,
  and correctly said "you have a strong background in sales but lack..." for the
  sales role rather than inventing an overlap.
- No run was left in a non-terminal state; stage advanced through the declared
  order and finished at `done`. `ingestion_runs.stage` was readable from a
  second connection while the run was in progress, which is the whole point of
  17.2.
- Timings on this machine: resume extraction 121s; the full 12-job pipeline
  672s, essentially all of it in stages d and h (one skill extraction and one
  explanation per job). Scoring and match-writing were not measurable against
  that.

**Not exercised by this corpus:** multi-chunk descriptions. All twelve postings
chunked to exactly one chunk, because they are 150-250 words and the target
chunk size is 200-400 tokens. Real postings are longer, and the multi-chunk path
is covered only by `tests/test_chunking.py`, not by a live run.

### 19.2 A second defect: a timeout reported as an unreachable daemon

The re-run after the skill-name fix failed with:

    LLMUnavailableError: Cannot reach the Ollama daemon at
    http://localhost:11434. Is it running? Try: ollama serve

The daemon was running and answering. The real cause was a **read timeout**:
resume extraction had measured 170.6s against `ollama_timeout_seconds = 180`, and
the second run landed on the wrong side of that margin.

The bug is in the day-2 error mapping, which this day's refactor moved but did
not re-examine: `httpx.TimeoutException` **subclasses** `httpx.TransportError`,
so the connection-failure branch swallowed every timeout and gave advice that
would waste an operator's time on a daemon that is already up.

Fixed in both providers by catching `httpx.TimeoutException` *before*
`TransportError` — clause ordering is the whole fix — with a message that names
the elapsed limit and the setting to change. `ollama_timeout_seconds` was also
raised from 180 to 300, since a 180s budget for a task that measures 170s is a
coin flip rather than a limit. `tests/test_providers.py` pins the ordering,
including an explicit assertion that `ReadTimeout` really is a `TransportError`
subclass, so the two clauses cannot be reordered back.

Worth stating plainly: this was only ever going to be found by running the thing
end to end on a slow machine. Nine of the tests now covering it were written
after the failure, not before it.

**Note for `.env.example`:** it should gain an `OLLAMA_TIMEOUT_SECONDS=300` line
to match. That edit was not made — the file was not writable in this
environment — so the setting is currently documented only in `app/config.py` and
in the timeout error message itself.

### 19.3 Calibration was measured but is NOT committed

`scripts/calibrate_similarity.py` was run against the corpus above, and the
placeholder `COS_LO, COS_HI = 0.45, 0.85` remains in the code deliberately.

The measurement, over 12 jobs:

```
Per-job top-3 mean
  n=12  min=0.5017  mean=0.6765  max=0.7982
  p5=0.5645  p25=0.6321  p50=0.6630  p75=0.7224  p95=0.7942
```

**This is the narrow-band problem, measured.** The single worst match in the
set — a backend engineer's resume against an enterprise SaaS sales role, which
share essentially nothing — still scores **0.50**. The best scores 0.80. Nothing
is near 0, nothing is near 1, and 90% of the mass sits inside a 0.23-wide band.
Fed unrescaled into the weighted sum, the semantic term would vary by about
0.09 across the entire result set while nominally carrying 40% of the score.
That is the failure 16.3 describes, and it is not hypothetical.

The placeholder `[0.45, 0.85]` turns out to be a reasonable guess but slightly
too wide on both ends, which compresses the rescaled range. The measured p5/p95
would suggest roughly `[0.56, 0.79]`.

**It is still not committed**, and the script itself declined to recommend it:

```
Not enough data to calibrate confidently (fewer than 20 samples).
Ingest more jobs and re-run.
```

That guard firing is the right outcome. The band is real but the *corpus* is
not — twelve hand-written descriptions do not have the distribution of a few
hundred real Adzuna postings, and every one of them is a software role, which
narrows the band further than a real result set would. Committing a
fitted-looking constant derived from synthetic data is worse than leaving a
labelled guess, because the next person to read it cannot tell the difference.

**This is the first thing to do once Adzuna credentials are real:** ingest a few
hundred postings, re-run the script, and set the two constants from p5/p95.

### 19.4 Still not verified

- **A live Adzuna API call.** Unchanged since day 1: `.env` holds `test_id` /
  `test_key`, and a real fetch returns 401. Every Adzuna response-shape
  assumption remains unconfirmed.
- **Real resumes.** Extraction has now been exercised against two synthetic
  resumes. Real ones are messier — multi-column PDFs, tables, headers — and the
  24k truncation has still never fired against one.
- **Concurrency.** Two searches for the same resume running at once is handled in
  principle (every write is an upsert, `SkillCanonicalizer` loses races safely)
  but has not been run.
- **Scale.** The largest run tested is 25 jobs. The N+1-avoiding bulk loads in
  `_load_postings` were written for 200+ but have not been measured there.

---

# Decisions — Day 3b (extraction latency)

One change, driven entirely by measurement: the extraction schema no longer asks
the model to write prose. Everything here is a number first and an opinion
second, because the obvious guesses were all wrong.

---

## 20. Extraction latency

### 20.1 The measurement, before any change

Resume extraction took ~217s warm for a 600-word resume (170s was the earlier
cold figure that started this). `scripts/benchmark_llm.py` isolated the cause by
adding one variable at a time. All figures warm, same machine, same model
(`qwen2.5:7b`), same 600-word fixture:

| Test | Time | In | Out |
|---|---|---|---|
| Trivial generation, no schema ("reply with one word") | 0.39s | — | ~1 |
| 600w resume, **no schema at all**, free-form JSON asked | 253.9s | 1074 | 958 |
| Full schema, **100-word input** | 68.1s | 222 | 275 |
| **Flat half-schema** (ContactAndSkills), 600w input | 67.9s | 1029 | 295 |
| Full schema, 600w input | 217.1s | 1029 | 941 |

**Time scales with output token count and nothing else.** Divide every row
through and the rate is 0.23-0.26s per output token — a flat **~4 tokens/sec**
across all five. Nothing else correlates:

- **Input size is irrelevant.** 222 input tokens and 1029 input tokens both took
  ~68s, a 4.6x difference in input for a 0.3% difference in time, because both
  emitted ~285 output tokens.
- **Schema nesting is irrelevant.** The flat half-schema took the same ~68s as
  the deeply-nested small-input run, again because output volume matched.
- **Constrained decoding is not merely free, it is negative.** Removing the
  schema entirely made the same extraction *slower* — 254s against 217s —
  because without a grammar the model adds preamble and prose it would otherwise
  be forbidden from emitting. This is the result that surprised me most, and it
  is why `format=` and `temperature=0` are kept.

### 20.2 The 0.39s baseline was misleading, and I read it wrong the first time

An earlier pass measured warm trivial generation at 0.39s and I concluded "the
hardware is fine, keep it local". That was wrong: a one-token reply measures
prompt latency, not throughput. At 4 tok/s a 7B model is slow — that is
CPU-only inference — and the correct reading is that the machine is slow *and*
the output was enormous, with the second factor being the one we control.
Recorded because the mistake is instructive: never infer throughput from a
single-token response.

### 20.3 The split was measured and rejected

Splitting `ParsedResume` into two calls was the obvious fix and it does not
work. Sequential: 205s. Concurrent on two threads: 174s. Against 217s for the
single call. The reason is arithmetic — the two halves emit the same total
tokens as the whole, so at a fixed 4 tok/s the total time cannot move. The
concurrent run's modest gain is Ollama overlapping one request's prompt
evaluation with the other's generation, not real parallelism; it does not
serialise perfectly, but it does not parallelise either, and it doubles peak
memory to buy ~20%.

**Rejected. The only lever is fewer tokens.**

### 20.4 Prose fields removed; `raw_text` is where prose lives

The model was transcribing text that already exists verbatim in
`resumes.raw_text`, which is stored, unchanged, and free to read. So the schema
now returns identifiers and short labels only:

| Removed | Was |
|---|---|
| `ExperienceEntry.summary` | a sentence or two per role — the single largest contributor |
| top-level `summary` | the professional-summary paragraph |
| `ProjectRef.description` | never shipped; excluded from the new schema by the same rule |
| `start_date` / `end_date` / `is_current` | three fields collapsed to one free-text `duration` |
| `location`, `field_of_study` | short, but nothing reads them |

`skills` is kept **in full** and deliberately untouched. It is the sole input to
the skill component, which carries 60% of every match score and is the half a
candidate can act on. A faster extraction that lost skills would be a worse
system, not a better one, which is why `--verify-trim` asserts skill recall
rather than only timing.

Renames: `total_years_experience` -> `total_experience_years`,
`ExperienceEntry` -> `ExperienceRef` (`title` -> `role`), `EducationEntry` ->
`EducationRef` (`graduation_year: int` -> `year: str | None`). New:
`ProjectRef(title, tech)`.

The `Ref` models use bare `str` rather than `str | None` where the old ones were
nullable. A nullable field lets the grammar emit `null`, and a `null` costs
tokens to say nothing; `""` is the empty case instead, and list-level validators
drop entries that are empty in every field.

### 20.5 What this costs: the resume embedding is thinner

`build_resume_embedding_text()` previously included each role's `summary` — the
most job-description-like prose available anywhere in the parse. That is gone.
It now builds from skills, role titles, employers, project titles and project
tech. **This is a real loss of semantic signal and it should not be read as a
free win.**

Two mitigations, neither of which fully replaces prose:

- `projects[].tech` is new, and often names tools the skills list omits.
- The skill component — 60% of the score — is completely unaffected.

The consequence to watch: the embedded text is now mostly nouns, and nouns embed
further from a job description's prose than prose does. The resume-to-job cosine
band will have moved, so **`COS_LO`/`COS_HI` must be re-derived** with
`scripts/calibrate_similarity.py` before the semantic component is trusted —
they were already uncalibrated placeholders (19.3), and this makes the existing
values staler rather than newly wrong.

### 20.6 Stored resumes parsed before this change are stale

`resumes.parsed` is JSONB, so there is no migration and nothing errors. But a
row written under the old schema has `total_years_experience` and
`experience[].title`, and the new readers look for `total_experience_years` and
`experience[].role`. Those reads return `None`, which degrades gracefully —
`experience_multiplier` returns 1.0 for missing data by design, and
`build_search_queries` falls back to skills — but the resume is quietly matching
on less than it should.

**Old rows need re-parsing.** Not automated here: there is one such row in this
database, and a backfill for a one-row table would be ceremony. It is a real
migration task the moment there is real data.

### 20.7 Two levers not taken

- **A smaller model (`qwen2.5:3b`).** Would raise tokens/sec, and the 3b model
  is already pulled. Not taken because day 2 chose 7b specifically for
  extraction accuracy on messy resume text (13.1), and the trim already
  addresses the latency without touching quality. This is the right next lever
  *if* latency is still a problem, and it should be measured against skill
  recall, not just the clock.
- **GPU offload.** 4 tok/s is CPU-only inference; a GPU would move throughput by
  an order of magnitude and is by far the largest available win. Not taken
  because it is a hardware/deployment change, not a code change, and nothing in
  the codebase would differ.

Both remain open. The trim was chosen first because it is the only one of the
three that costs nothing at run time and is portable to whatever hardware this
eventually runs on.

---

## 21. Verification of the trim (day 3b)

`python scripts/benchmark_llm.py --verify-trim`, same 600-word fixture, same
model, warm. Run twice, because the first run's throughput did not match the
baseline's and that made it uninterpretable.

| | Baseline | Run 1 | Run 2 |
|---|---|---|---|
| Output tokens | 941 | **483** | **483** |
| Wall clock | 217.1s | 146.6s | **100.9s** |
| Throughput | 4.33 tok/s | 3.30 tok/s | 4.79 tok/s |
| Speedup | — | 1.48x | **2.15x** |

**The fix worked, and it worked in proportion.** Run 2 is the valid comparison:
its throughput (4.79 tok/s) is within noise of the baseline's 4.33, so the
change in wall clock is attributable to the change in output volume. **51% of
the tokens, 46% of the time** — proportional, slightly better.

### 21.1 Run 1 was discarded on evidence, not on preference

Run 1 came in at 146.6s / 3.30 tok/s — a 24% slower machine state than the
baseline was measured at. That is a confound, not a result, and the script says
so itself rather than quietly reporting the ratio:

    -> CHANGED: machine state differs, speedup not attributable

Both runs are recorded here because discarding the less convenient of two
numbers is only legitimate if the discarded one is visible. Note also that the
output token count was **identical across both runs (483)** — `temperature=0`
makes extraction deterministic, so token count is a stable measurement and
wall clock is the noisy one. That is what makes the throughput check able to
separate them at all.

### 21.2 Output tokens landed at 483, not the estimated 200-250

The estimate was wrong; the change was not. Removing prose removed prose. What
remains is still substantial because this fixture has a genuinely long skills
section and the model returned **27 skills**, four roles, two projects and one
education entry. `skills` is now the largest remaining contributor and is kept
in full deliberately (20.4) — it feeds 60% of every match score. Reaching 200
tokens would have meant trimming skills, i.e. trading match quality for latency,
which is the wrong trade.

So: the target of 50-65s was not met. **100.9s is the real number**, from 217.1s.

### 21.3 Extraction quality did not degrade

This mattered more than the clock, and is why `--verify-trim` asserts
correctness rather than only timing.

| Check | Result |
|---|---|
| Skills list non-empty | pass — 27 extracted |
| All 12 skills known to be in the fixture found | pass |
| `total_experience_years` populated and plausible | pass — 7.0 |
| Every `projects[]` entry has a non-empty title | pass — 2 of 2 |
| No removed field reappeared (9 checked) | pass |

Nothing was lost: roles 4, education 1, both complete.

### 21.4 The embedding text, measured

618 chars / 72 words, against **1142 chars** for a comparable resume before the
trim — roughly **half**. It is now three noun-heavy lines: a skills list,
`Roles: <title> at <employer>`, and `Projects: <title> (<tech>)`. The role
summaries that carried the only job-description-like prose are gone.

This is the cost named in 20.5, now quantified. `COS_LO`/`COS_HI` must be
re-derived with `scripts/calibrate_similarity.py` before the semantic component
is trusted.

---

# Decisions — Day 3c (real Adzuna data, Groq as parse provider)

Two changes: the Adzuna credentials became real, so the first genuine corpus
exists; and resume parsing moved to a hosted provider with a local fallback.

---

## 22. Groq as the parse provider

### 22.1 Structured output mode: full JSON schema, not loose JSON

**Found by probing the live API, not by reading docs.** Groq's first response to
`ParsedResume.model_json_schema()` was a 400:

    invalid JSON schema for response_format: 'ParsedResume':
    /$defs/ProjectRef: `additionalProperties:false` must be set on every object

That is a **schema-shape requirement, not a missing feature**, and the
distinction matters more than it looks. Read as "json_schema is unsupported",
the obvious next move is to fall back to `{"type": "json_object"}` — loose mode,
which guarantees only *some* JSON and silently transfers responsibility for
field names and types onto the validate-and-retry loop in `app.structure`. The
grammar would be gone and nothing would say so.

`groq_llm.harden_schema()` adds `additionalProperties: false` and a full
`required` list to every object node, including the `$defs`. With that,
`json_schema` + `strict: true` works on every model tried, and behaviour mirrors
Ollama's `format=` exactly: malformed shape stays unrepresentable, and the retry
loop in `structure.py` stays scoped to *accuracy*, which is what it was written
for.

So: **full schema support. The validate-and-retry loop did not become load
bearing.**

### 22.2 httpx, not the `groq` SDK

httpx is already a dependency for the job sources, the surface used here is one
endpoint, and an SDK would be another package to pin for no capability needed.

### 22.3 Model choice: `qwen/qwen3.8-27b`, chosen 2026-08-31

`llama-3.3-70b-versatile` — the plausible-looking default — **does not exist on
this account**. The model list was fetched from `/openai/v1/models` and all three
viable candidates were measured on the same 600-word resume rather than picked
by reputation:

| Model | Time | Completion tokens | of which hidden *reasoning* | Quality |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | 3.99s | 1187 | **894** | 27 skills, 0 missed |
| `openai/gpt-oss-20b` | 3.07s | 1295 | **996** | 27 skills, 0 missed |
| **`qwen/qwen3.8-27b`** | **2.50s** | **367** | 0 | 27 skills, 0 missed |

All three were **quality-identical**. The decider was token cost against a free
tier metered at 8000 tokens/minute: the gpt-oss models spend 75-77% of their
completion budget on hidden reasoning tokens that are billed, counted against
the limit, and thrown away. qwen3.8 emits none, is the fastest, and is the same
family as the local `qwen2.5:7b`, so prompt behaviour is consistent across the
primary and the fallback.

**The model list moves.** Re-run the listing before changing `GROQ_MODEL` rather
than assuming any of these still exist.

### 22.4 Groq parses resumes; Ollama still does everything else

`PARSE_PROVIDER=groq` is deliberately narrower than `LLM_PROVIDER`, and the two
are separate settings:

- **Resume parsing** is one call per upload, and it is the only thing a user sits
  and waits for. It goes to Groq.
- **Job-skill extraction and match explanations** run per-job inside a search —
  80 to 200 calls — and stay on `LLM_PROVIDER=ollama`, because Groq's free tier
  is metered per minute and a search would spend the whole budget in one run.

`get_parse_provider()` is a separate factory from `get_llm_provider()` for this
reason, and `app/structure.py` is the only caller.

**Follow-up worth taking, with numbers rather than a hunch:** the 80-job
ingestion in 23.1 spent ~100 minutes in Ollama skill extraction. Those
descriptions are ~500 chars, so roughly 600 tokens per call, ~48,000 tokens
total — about 6 minutes at 8000 TPM. Routing stage (d) to Groq is very likely a
10x win on the slowest stage of the pipeline. Not done here: it is a change to
search behaviour, not to parsing, and it belongs in its own pass with its own
measurement.

### 22.5 Fallback is per call, and logged

`FallbackLLMProvider` tries Groq and, on any `LLMError`, retries that call on
Ollama at WARNING. Rationale:

- **A resume upload must not fail because the network is down.** The project's
  premise (13.3) is that everything runs locally; a hosted dependency that could
  take the feature down with it would be a straight downgrade.
- **Per call, not per process.** A transient outage must not pin the application
  to the 100-second path until it restarts, and a persistent one should surface
  on every request rather than in one log line that scrolled away hours ago.
- **WARNING, not ERROR**, because the call is about to succeed — but it must be
  visible, since silent degradation to a 70x slower path is exactly what goes
  unnoticed until a demo.

`EMBEDDING_PROVIDER` stays `ollama`: Groq serves no embedding endpoint, and
local batched embeddings already measured 0.51s/chunk.

### 22.6 Free-tier rate limits, observed 2026-08-31

From response headers on `qwen/qwen3.8-27b`:

| Header | Value |
|---|---|
| `x-ratelimit-limit-requests` | 1000 |
| `x-ratelimit-limit-tokens` | 8000 |
| `x-ratelimit-reset-requests` | ~4m19s (observed, decreasing) |
| `x-ratelimit-reset-tokens` | ~8s |

The token bucket refills on the order of seconds, so 8000 reads as **tokens per
minute**; requests reset on a much longer horizon, so 1000 reads as **requests
per day**. One resume upload costs ~1,430 tokens (1060 prompt + 367 completion),
so a single upload uses roughly **18% of one minute's token budget** and 0.1% of
the daily request budget. Nowhere near a limit for the demo path — but the
per-minute ceiling is exactly why 22.4 keeps per-job extraction off Groq, and
it is the number to check first if a bulk path is ever pointed at it.

### 22.7 Measured: 1.44s against 203.83s

Same fixture, same schema, same temperature, run back to back
(`benchmark_llm.py --verify-trim`):

| | Model | Out tok | Seconds | tok/s | Skills | Quality |
|---|---|---|---|---|---|---|
| baseline (recorded) | qwen2.5:7b | 941 | 217.1 | 4.33 | — | — |
| ollama | qwen2.5:7b | 483 | 203.83 | 2.37 | 27 | PASS |
| **groq** | qwen/qwen3.8-27b | **367** | **1.44** | **254.50** | 27 | PASS |

**Quality is identical, and that is the result that matters.** The skill-set diff
between the two providers is empty in both directions — same 27 skills, same
`total_experience_years` of 7.0, same 4 roles, 2 projects, 1 education entry.
Every correctness assertion passed on both.

Two honest caveats on the timing:

- **The Ollama row is contended.** It ran at 2.37 tok/s against the 4.79 tok/s
  measured on a quiet machine, because the 80-job ingestion in 23.1 was doing
  skill extraction on the same CPU at the time. Against the clean 100.9s figure
  from 21, Groq is ~**70x** faster, not 141x. The honest range is 70-140x
  depending on what else the machine is doing — which is itself an argument for
  the hosted provider, since it is unaffected by local load.
- **Token drift is 24%** (367 vs 483), just inside the 25% threshold the script
  flags at. This is not a schema-handling difference: the *content* is
  identical, so it is the two models formatting the same information slightly
  differently (whitespace, key ordering). Same schema, same temperature,
  different model — 24% is expected. It would be worth investigating if the
  extracted content differed, and it does not.

---

## 23. Real Adzuna data

### 23.1 Credentials verified, and the shape assumptions finally checked

`scripts/verify_adzuna.py` (new) is the gate. It goes through `AdzunaSource`
rather than raw httpx, so a pass means the real ingestion path works, and it
prints the upstream status *and body* on failure because 401 (bad key) and 403
(key not yet activated) need different responses.

Day 1's response-shape assumptions held: `created` parsed to a date on 3/3 rows,
currency matched the country, `apply_url` and `company` present.

**The one that did not: `description` is truncated to 500 characters.** Adzuna's
search endpoint returns a teaser, not the posting. Two consequences, neither of
which was visible with the hand-written fixtures:

- **Every job produces exactly one chunk.** The chunker targets 200-400 tokens
  and 500 chars is ~90 tokens, so `chunk_description` returns a single chunk for
  essentially every real posting. The top-3-mean in `semantic_component` (16.2)
  therefore degenerates to "the only chunk" on this corpus. The design is not
  wrong, but its central mechanism is inert against this source, and that is
  worth knowing before concluding anything about chunking from these numbers.
- **Skill extraction has far less to read**, so recall per job is bounded by the
  teaser rather than by the model.

Fixing it means fetching each posting's own page, which is a scraping change,
out of scope here and recorded rather than done.

### 23.2 The stored resume was re-parsed before ingestion

The one resume in the database still had the pre-trim keys
(`total_years_experience`, `experience[].title`) and an embedding built from
prose — exactly the staleness predicted in 20.6. Calibrating the new noun-based
band against an old prose-based embedding would have measured nothing, so the
resume was re-parsed under the trimmed schema, re-embedded, its skills relinked
and its matches dropped, before the search ran.

The 12 synthetic `source='local'` jobs from day-3 verification were deleted for
the same reason: they were hand-written to be easy, and leaving them in would
have flattered the corpus.

---

# Decisions — Day 3d (dictionary skill extraction)

**Provisional.** Committed as a trial so it can be exercised on real data. Two
known problems are recorded below unfixed (24.4, 24.5); read them before
trusting a match score from this corpus.

---

## 24. Pipeline stage (d): dictionary lookup, not an LLM

### 24.1 Why the generative call had to go

Stage (d) made one constrained-decoding call per job at roughly **45 seconds**.
For the 80-job Adzuna run that is an hour of wall clock, and the run was killed
at 19/80 rather than waiting it out.

The model was not reasoning about anything. It was reading technology names out
of a document and writing them back — a lookup, dressed as generation.

| | LLM | Dictionary |
|---|---|---|
| 80 jobs | ~45s/job, ~60 min | **1.27s** |
| Per job | ~45,000ms | **15.9ms** |
| Index build | — | 0.20s, once per process |

**What is genuinely lost:** a dictionary cannot find what it does not know. The
LLM discovered names nobody had entered. Two mitigations, neither complete: the
vocabulary grows from every resume parse, and `skill_seed.py` provides a base.
A posting naming something novel now yields nothing for that term, and
`skill_component` reports it as *unparsed* rather than as a mismatch — which is
the honest failure mode, and the one the day-3 `None`-not-`0.0` decision (16.5)
was built for.

### 24.2 The canonical-form filter, and the bug it started as

`build_index` skips any row where ``clean_skill_name(name) != [name]``.

The first version tested ``if not clean_skill_name(name)`` — "does this clean to
something non-empty" — and that is wrong in a way worth recording, because it
cost a debugging session and was caught by a test rather than by reading:

* `"Strong Python"` is a real row in `skills`, written by the pre-fix LLM path.
* `clean_skill_name("Strong Python")` returns `["Python"]`, which is truthy, so
  the non-empty check **kept it**.
* Being longer than `"Python"`, it then won longest-first matching.
* It resolved to *its own* `skill_id` — a different row from the one the resume
  canonicalised onto.
* The job said Python, the resume said Python, and the intersection was empty.

Testing for *already canonical* raises the skip count from 19 rows to 61 and
fixes it. Rows are filtered at index time rather than deleted, because
`job_skills` and `resume_skills` still reference them.

### 24.3 Short names need positive evidence

`R`, `Go`, `C`, `C#`, `D`, `js` are real skills whose names are also ordinary
English. A false positive here is not cosmetic: a job wrongly tagged `Go`
matches every Go developer.

Rules, in order:

1. **An adjacent `&` disqualifies outright.** "R&D" is one idiom.
2. List punctuation on either side accepts — `Python, Go, Rust`.
3. An *immediately adjacent* qualifier token accepts — "Go developer".

Two details that were wrong in the first draft:

* **Start-of-string is not evidence.** Treating position 0 as a list boundary
  matches "Go to our website".
* **The qualifier check must be adjacency, not a window.** A 24-character window
  containing `engineer` passed "R&D engineer", because `engineer` appears in
  essentially every job description. Only the neighbouring token counts now.

### 24.4 OPEN: the vocabulary still contains non-skills

`Backend Engineer` is the **single most-matched "skill" in the corpus (25 of 80
jobs)**, alongside `architecture`, `Databases`, `backend systems`, `Architect`
and `APIs`. These are job titles and vague nouns that earlier LLM runs wrote
into `skills`. They are short and well-formed, so the canonical-form filter in
24.2 does not catch them, and they inflate the skill component of every score.

Not fixed here: pruning rows is a data decision with foreign-key implications
and belongs in its own pass. The likely fix is a blocklist of job-title and
abstract-noun forms applied at index time, mirroring 24.2.

### 24.5 OPEN: the required/preferred split is untested on real data

**All 246 extracted rows came back `required`. Zero `preferred`.**

Not a bug in the splitter — it is verified against synthetic input and by 45
unit tests. The cause is 23.1: Adzuna truncates descriptions at 500 characters,
and a "nice to have" section essentially never appears that early. One sample
posting opens with "Must have :" and is cut off before reaching anything else.

Consequence: `SKILL_WEIGHTS["preferred"] = 0.4` is **currently dead code against
this corpus**, and every job's skill score is computed as though every named
skill were mandatory. It will start mattering the moment descriptions are
fetched in full.

### 24.6 The seed vocabulary

`app/matching/skill_seed.py` — ~150 technologies with alias forms.

Needed because the `skills` table had **225 rows and zero aliases**: the alias
column existed but had never been populated, so `k8s` could not reach Kubernetes
and `js` could not reach JavaScript. Dictionary matching without aliases throws
away most of its own mechanism.

Seeding is idempotent and non-destructive — an existing row keeps its id, so
every `job_skills` and `resume_skills` reference survives, and aliases are merged
rather than replaced. First run: 83 added, 31 alias-backfilled.

### 24.7 The index loads lazily, not at module import

The spec said "at module import". It is a lazily-built process-wide cache
instead, because importing `app.matching` must not require a live database —
`pytest` collection, Alembic and `--help` all import this package, and an
import-time query would fail every one of them without Postgres. "Once per
process" is the property that matters and is preserved. `reset_index()` exists
for tests and for after seeding.

### 24.8 Extraction reads the title as well as the description

Aggregator postings truncate the body, and the title is often the only place a
technology is named ("Python Developer"). This is also what makes 24.4 worse
than it would otherwise be: the junk row `Backend Engineer` matches the *title*
of 25 postings.

### 24.9 Measured on 80 real Adzuna postings

| | |
|---|---|
| Wall clock, 80 jobs | **1.274s** (15.9ms/job) |
| Jobs with >= 1 skill | 56 |
| Jobs with zero skills | 24 |
| — of which pure boilerplate teaser | **22** |
| — genuine extraction misses | **2** |
| `job_skills` rows | 246 (avg 4.4 per job) |
| Requirement split | 246 required, **0 preferred** |

The zero-skill number reads badly until it is broken down. 22 of the 24 are
postings whose entire 500-character description is an equal-opportunity notice
with no technology named anywhere — "This job is with X, an inclusive employer
and a member of myGwork...". The marker phrases did not miss; there was nothing
to find. That is a source-truncation problem (23.1), not an extraction problem.

---

# Decisions — Day 3e (vocabulary pruning, Greenhouse and Lever)

---

## 25. Pruning the skill vocabulary

### 25.1 A flag, not a delete

`skills.active` (migration 0004), default true, false on 38 blocklisted rows.

Deleting was rejected: `job_skills` and `resume_skills` reference these rows, a
delete destroys the record of what was pruned, and it is irreversible. The flag
keeps every foreign key valid and the decision auditable. Extraction
(`jd_skills.build_index`), canonicalisation (`SkillCanonicalizer`), scoring
(`pipeline._load_postings`) and the API (`GET /jobs/{id}`) all filter
`active = true` — the last two matter because a pruned skill still has
`job_skills` rows written before it was pruned.

The 210 rows matching nothing were left alone, as instructed: inert, and
deleting them is unnecessary risk.

### 25.2 The blocklist is data, with a reason per term

`app/data/skill_blocklist.csv` — 38 rows of `term,category,reason`. Not a
Python list: it is a set of *judgements about data*, each needing a stated
reason, and it should be reviewable in a diff by someone who does not read
Python. Adding a term is a one-line data change.

Categories used: `job_title`, `job_function`, `abstract_noun`, `category`,
`buzzword`, `industry`, `domain`, `soft_skill`, `quality_attribute`,
`fragment`, `ambiguous_word`.

**My judgement on the 9 terms not covered by explicit instruction**, applying
the stated test — named technology or checkable competency, versus fragment,
job function or abstract quality:

| Term | Decision | Reason |
|---|---|---|
| API development | blocked | Job function; "REST APIs" is the skill |
| Cloud-based applications | blocked | Abstract noun phrase; AWS/Azure/GCP are the skills |
| ML solutions | blocked | "solutions" fragment; Machine Learning already exists |
| Python frameworks | blocked | Category; Django/Flask/FastAPI are the skills |
| NoSQL databases | blocked | Category suffix duplicating the kept `NoSQL` |
| Software Architecture | blocked | Same class as the blocked `architecture` |
| Technical Architecture | blocked | Same class as the blocked `architecture` |
| Agentic | blocked | Adjective buzzword |
| Promises | blocked | Ordinary English word; matches "promises to deliver" far more often than the JS concept |
| asynchronous programming | **kept** | Genuine competency, comparable to the kept `Data Structures` |
| Data Modelling, Golang, Nodejs, React.js, Mysql, RabbitMq, Agile Scrum | **kept** | Real technologies; some are unnormalised spellings, which is a separate merge task |
| Sales | **kept** | A real skill for sales roles — see 25.5, where this turned out to matter |

### 25.3 The recurrence guard

`blocklist.is_disallowed()` runs at skill *creation*, refusing:

* anything on the checked-in blocklist;
* anything whose **final** word is Engineer, Developer, Architect, Manager,
  Analyst, Specialist, Consultant, Lead or Intern.

Checked at creation because `active` lives on the *row*: without this, a
blocklisted term reappearing in a future posting is inserted under a fresh id
and the pruning is silently undone. Only a trailing title word counts, so
"Engineering Productivity Tooling" survives.

---

## 26. Greenhouse and Lever

### 26.1 Why: Adzuna's 500-character cap was breaking matching

Measured, not assumed. Median description length:

| Source | n | median | min | max |
|---|---|---|---|---|
| adzuna | 90 | **500** | 283 | 500 |
| greenhouse | 118 | **5543** | 1130 | 12072 |
| lever | 118 | **5047** | 1239 | 13642 |

Adzuna's median is exactly the cap — every posting of substance is truncated.
Eleven times more text from the board APIs.

### 26.2 Per company, not per query

Neither API has a search endpoint; both take a company slug. So `fetch` ignores
`query.keywords` and returns the whole board, letting dedup and scoring filter —
the scorer already ranks, and a keyword pre-filter here would discard postings
before they could be scored. `remote_only` and `max_results` *are* honoured,
because both are caps rather than search terms.

`base.py` and `adzuna.py` are untouched.

### 26.3 The slug list is checked-in data, and verified

`app/sources/company_boards.py`: 34 Greenhouse + 6 Lever slugs, **every one
verified to return at least one posting on 2026-08-31**.

**Indian coverage is thinner than intended and the reason is worth recording:**
most large Indian tech employers do *not* expose a public Greenhouse or Lever
board. Razorpay, Swiggy, Zomato, Flipkart, PhonePe, Meesho, Zerodha, Freshworks,
Zoho, Byju's, Unacademy, Lenskart, Nykaa, Delhivery, Udaan, BrowserStack,
Hasura, Chargebee and ~30 others were probed and all 404'd; they run their own
careers sites or a private ATS. The `NOT_FOUND` dict records every failure so
nobody rediscovers this. What survives is India-headquartered companies that do
use these boards (Groww, Postman, Druva, Netradyne, HighRadius, Glance, Zenoti,
HackerRank, Rubrik, Zeta, FamPay, CRED, Fi) plus remote-friendly global
companies hiring into India.

### 26.4 Round-robin interleaving, because the boards are wildly unequal

The 40 boards hold ~6,500 postings, and Databricks alone has 855. Concatenating
and truncating to a cap would fill it entirely from the first two boards.
`interleave()` takes one posting from each board in turn, so a 120-job cap is
spread across all 40 employers.

### 26.5 `asyncio.gather` over `asyncio.to_thread`

Forty sequential HTTPS round trips is ~40s of pure waiting. `JobSource.fetch` is
a synchronous interface that must not change, so the blocking per-board call is
pushed to a worker thread and gathered, with a semaphore capping in-flight
requests at 8 so a 40-board list does not open 40 sockets against two hosts.
`return_exceptions=True`: one dead board must not lose the other 39.

### 26.6 HTML handling

Greenhouse `content` is entity-encoded markup. Block-level tags are converted to
newlines *before* tags are stripped — without that the posting collapses to one
paragraph and `jd_skills.split_sections`, which anchors its markers to line
starts, never fires. Lever additionally returns `lists`, an already-structured
array of `{heading, content}` blocks; those headings are re-emitted on their own
lines rather than flattened, which is why Lever produces the most `preferred`
skills of any source.

---

## 27. Measured after both changes

326 jobs (90 Adzuna, 118 Greenhouse, 118 Lever) from 342 fetched, 16 dropped as
cross-source duplicates.

### 27.1 The preferred branch finally fires

**This was the key number, and it moved.**

| Source | required | preferred |
|---|---|---|
| adzuna | 150 | **0** |
| greenhouse | 451 | **42** |
| lever | 601 | **50** |
| **total** | 1202 | **92** |

Adzuna still returns exactly zero preferred, on 90 postings. That is now
positively diagnostic rather than a mystery: the section markers are fine, the
descriptions are simply cut off before a "nice to have" section appears.
`SKILL_WEIGHTS["preferred"] = 0.4` is live code again.

### 27.2 Skills per job, by source

| Source | jobs | mean | median | zero |
|---|---|---|---|---|
| adzuna | 90 | 1.7 | **0** | **45 (50%)** |
| greenhouse | 118 | 4.2 | 3 | 8 (7%) |
| lever | 118 | 5.5 | 4 | 5 (4%) |

Half of all Adzuna postings still yield nothing. The board sources are an order
of magnitude better on the same extractor and the same vocabulary — the
difference is entirely the source text.

### 27.3 Four of the six missing resume skills now match

| | before | after |
|---|---|---|
| Kubernetes | 0 | **14** |
| Linux | 0 | **9** |
| Kafka | 0 | **5** |
| Terraform | 0 | **5** |
| Celery | 0 | 0 |
| pytest | 0 | 0 |

Celery and pytest remain at zero. Both are real and both are in the vocabulary;
they are simply niche enough not to appear in a 326-job sample dominated by
large-company postings.

### 27.4 OPEN: pruning worked, and surfaced the next layer

The 38 blocked terms are gone from the results. But the new most-matched head
contains a *different* class of false positive, which the short Adzuna teasers
had been masking:

| Matches | Skill | Assessment |
|---|---|---|
| 95 | Sales | Mostly legitimate — the Binance and GoHighLevel boards post many Account Executive roles — but "Full Stack Developer Intern" also matched, from prose |
| 75 | Security | **False positives.** Matched "Account Executive", "ABM Manager", "Accountant II" — the word appears in ordinary prose |
| 54 | Observability | **False positives.** Matched "Accountant II", "AI Advisory Consultant". My seed gave it the alias `monitoring`, which is far too broad |
| 35 | UPI | Plausible on fintech-heavy boards, unverified |

The mechanism is new: generic single-word skills matching *prose* in a
5,000-character document. A 500-character teaser rarely contained enough prose
to trigger it. Two fixes suggest themselves — tightening over-broad aliases
(`monitoring` → Observability is the clearest offender), and requiring
list-context for generic single-word terms, much as `_short_name_ok` already
does for `R` and `Go`.

**Not fixed in this pass.** It is a new finding from new data, and folding it in
silently would hide that the previous prune both worked and was incomplete.

### 27.5 Cost

Extraction 25.1s for 326 jobs (77ms/job — up from 16ms because the documents are
11x longer). Embedding 953 chunks in 474.6s, which is now the slowest stage by a
wide margin at ~0.5s/chunk, exactly as measured on day 3b.

---

# Decisions — Day 3f (vocabulary scoped to named things; first real calibration)

## 28. The scoping rule

Vocabulary is now **named technologies and named specific practices only**.
Abstract competencies and category nouns are out.

The test: *can a matcher verify this against a posting?* "Does this job require
Kubernetes" is answerable. "Does this job require Security" is not — it meant
different things in 75 different postings, including "Account Executive" and
"Accountant II".

**This reverses 25.2, which kept Security and Agile as borderline.** That call
was made against 500-character Adzuna teasers. Full-length descriptions changed
the answer, and the reversal is recorded rather than quietly applied: an
abstract term in a 5,000-character document has ten times the surface area to
collide with prose.

21 further terms blocked (59 total): Security, Observability, Agile, Scrum,
Sales, Presentation, written communication, Technical Writing, Accessibility,
Authentication, Authorization, Code Review, Incident Response, Performance
Tuning, API Design, Data Modelling, Data Pipelines, Unit Testing, Web Scraping,
design systems, Excel.

**Kept**, and worth naming because they look similar but survive the test:
Microservices (named architecture), System Design, distributed systems and Data
Structures (named specific practices), RAG and LLM (named techniques), UPI, P2P,
QR (narrow domain terms), Looker and Salesforce (named products).

### 28.1 The alias audit

`monitoring -> Observability` was the direct cause of that term matching
accountancy roles. Auditing all seeded aliases against the rule — *an alias must
be an abbreviation or spelling variant, never an ordinary English word that
merely co-occurs* — found 13 more of the same failure mode:

| Alias | Was pointing at | Why it is wrong |
|---|---|---|
| monitoring | Observability | ordinary word; matched "Accountant II" |
| shell, sh | Bash | "shell company"; `sh` is two letters |
| spring | Spring Boot | the season |
| rest | REST APIs | "the rest of the team" |
| node | Node.js | "node in the graph" |
| torch | PyTorch | ordinary word |
| **cv** | Computer Vision | **curriculum vitae — in job postings** |
| lambda | Serverless | Python keyword, mathematical term |
| on-call, on call | Incident Response | prose in most engineering postings |
| scraping | Web Scraping | ordinary word |
| unix | Linux | a different OS, not an alias at all |

`k8s -> Kubernetes`, `postgres -> PostgreSQL`, `js -> JavaScript` are correct and
kept. Recorded in `skill_seed.REJECTED_ALIASES` so they are not re-added.

### 28.2 Ambiguous technology names get an evidence test, not a blocklist

Spark, Vault, Envoy, Playwright, Jest, Express, Rails, Oracle, pandas, Cypress
and Selenium are genuine verifiable technologies whose names are ordinary
English words. Blocklisting them would be wrong — "does this posting require
Apache Spark" is exactly what a matcher *can* answer. They now go through the
same evidence test as `R` and `Go` (`_short_name_ok`): list punctuation or an
adjacent qualifier, or no match. Measured cause: "Spark" matched "Account
Maintenance Associate" via prose like "spark innovation".

Excel is the exception — blocklisted rather than guarded, because the English
verb ("excel at this role") is overwhelmingly the more common sense in postings.

### 28.3 Results

Active rows 270 -> 249 (308 total, unchanged — nothing deleted). Index 295 ->
252 surface forms. Re-extraction 23.3s over 326 jobs.

**The head is now entirely named technologies:**

    68 Python · 45 Machine Learning · 40 Microservices · 39 SQL · 37 Java
    35 UPI · 33 LLM · 27 AWS · 23 Salesforce · 17 Azure · 17 GCP
    15 TypeScript · 15 distributed systems · 14 Spark · 14 Kubernetes

No obvious false-positive class remains in the head.

| | before | after |
|---|---|---|
| required / preferred | 1202 / 92 | **824 / 67** |
| adzuna | 150 / 0 | 138 / 0 |
| greenhouse | 451 / 42 | 268 / 24 |
| lever | 601 / 50 | 418 / 43 |

Zero-skill jobs rose from 58 to 115 of 326, and **57 jobs lost all their
skills**. That number looks alarming and mostly is not:

* **63 of the 115 are non-engineering postings** — Account Executive, Affiliate
  BD, Associate General Counsel, Accounting Manager. Their only "skills" were
  Sales, Security, Presentation or Excel. Zero is the *correct* answer for a
  legal-counsel role, and `skills_unparsed` routes it to semantic-only scoring.
* **49 of the 52 engineering-titled zeroes are Adzuna** — the 500-character
  teaser again, not over-pruning. Greenhouse contributed 2, Lever 0.

---

## 29. First calibration against real data (326 jobs, 953 chunks)

**Nothing was set.** COS_LO/COS_HI remain at the placeholder 0.45/0.85.

### 29.1 Resume-to-chunk cosine

```
Per-chunk (raw)      n=953  min=0.4099  mean=0.5964  max=0.7940
                     p5=0.5118  p25=0.5642  p50=0.5959  p75=0.6288  p95=0.6842

Per-job top-3 mean   n=326  min=0.4983  mean=0.6100  max=0.7940
                     p5=0.5322  p25=0.5818  p50=0.6043  p75=0.6389  p95=0.6911
```

The script suggests `COS_LO, COS_HI = 0.53, 0.69`. The band is **0.16 wide**,
against the placeholder's 0.40 — the narrow-band problem from 16.3, now measured
on a real corpus. This supersedes the 12-job synthetic measurement in 19.3
(0.56/0.79): the embedding trim (21.4) replaced prose with nouns, so that
earlier band is void, exactly as predicted.

Every job produced exactly one chunk on Adzuna and few elsewhere, so the
"top-3 mean" is close to a max in practice — worth knowing when reading the
per-job row.

### 29.2 Final overall_score

```
overall_score    n=326  min=0.0822  mean=0.3152  max=0.8750
                 p5=0.1227  p25=0.1718  p50=0.3035  p75=0.4078  p95=0.6655

semantic_score   n=326  min=0.1208  mean=0.4000  max=0.8600
                 p5=0.2055  p25=0.3295  p50=0.3856  p75=0.4721  p95=0.6028

skill_score      n=326  min=0.0000  mean=0.1373  max=1.0000
                 p5=0.0000  p25=0.0000  p50=0.0000  p75=0.2000  p95=0.6667
```

`skill_score` has a **median of zero** — over half the corpus scores nothing on
skills, because 115 jobs extract none and many others share nothing with this
resume. 115 of 326 matches are `skills_unparsed` and therefore semantic-only.

### 29.3 OPEN, and it should be settled before thresholds are frozen

The ranking is wrong at both ends, and not because of the vocabulary.

Top of the ranking, by `overall_score`:

| score | sem | skill | title | matched |
|---|---|---|---|---|
| 0.7764 | 0.441 | **1.000** | Account Executive - Italy | **Git** |
| 0.7326 | 0.331 | **1.000** | Logistics Manager | **SQL** |
| 0.7318 | 0.330 | **1.000** | Accounting Manager | **SQL** |

Bottom of the ranking: `SWE-3 Backend Engineer, ML Systems` at **0.0822**.

Two mechanisms, both structural:

1. **Weighted recall saturates on thin extraction.** `skill_component` is
   `earned / possible`. A posting with exactly one parsed requirement that the
   candidate happens to hold scores a perfect 1.0. An Accounting Manager needing
   SQL therefore out-ranks most backend roles. This is the cost of the
   recall-not-Jaccard decision (16.4), which is right in general and degenerate
   when `possible` is 1.
2. **`skills_unparsed` is a penalty in practice.** Semantic-only scoring lands
   near the middle of a narrow band, so a genuine backend role whose Adzuna
   teaser yielded no skills sinks below a sales role that matched one.

Both were invisible until a real corpus existed. Reported, not acted on, as
instructed. Likely fixes: a confidence floor on `possible` (a job with one
parsed requirement should not be able to reach 1.0), and revisiting what
`skills_unparsed` should contribute now that it affects a third of the corpus.

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

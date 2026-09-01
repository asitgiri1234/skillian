# Skillian frontend

Vite + React, plain CSS, `fetch`. No router — one view with state.

```bash
npm install
npm run dev      # http://localhost:5173
```

The backend must be running on `http://localhost:8000`
(`uvicorn app.main:app --reload` from the repo root). Override with
`VITE_API_BASE`.

## What is here (parts 1-2 of 3)

Upload → parse → search → poll → results. The skills sidebar and the design
pass are part 3.

```
src/api/client.js          every call to the API; nothing else uses fetch
src/lib/extractText.js     PDF/DOCX -> plain text, in the browser
src/hooks/useRunPoller.js  polls GET /runs/{id} until is_terminal
src/components/
  UploadView    file -> parse -> "Search jobs"
  PollingView   real stage progress, not a spinner
  ResultsView   split view: ranked list left, detail right
  MatchCard     one result; tier badge, thin-evidence badge, explanation
  JobDetail     full posting, requirements, comparison, score breakdown
  ErrorNotice   every failure surface
```

## Two rules the results view is built on

**Never re-sort or re-rank client-side.** Ranking is computed server-side over
the complete set. Sorting a page of 25 against itself would silently disagree
with page 2.

**Tier is colour *and* text, and null is a real value.** Roughly 1 in 12 men
cannot reliably distinguish red from green, so the word carries the meaning.
Unparsed matches have `tier: null` and get no badge at all — the backend
withholds that judgement deliberately and the UI must not invent one.

## Two API facts this code is built around

**Score fields arrive as JSON strings.** `"0.8500"`, from `Numeric(5,4)`.
`client.js` converts them to numbers at the boundary, so no component ever sees
a string. Sorting them as strings puts `"0.9"` above `"0.85"`.

**Poll `is_terminal`, not `status`.** Two status vocabularies share that column
(`success` from the CLI path, `succeeded` from the search pipeline), so
`status === 'succeeded'` would poll forever against a run written by the other
one.

## Text extraction happens here, not on the server

`POST /resumes` is `application/json` and takes `raw_text`. It has no PDF or
DOCX handling, and the backend is closed, so `src/lib/extractText.js` does the
extraction with `pdfjs-dist` and `mammoth` and posts plain text.

One upside: a scanned PDF is detectable here — pages but no text layer — so the
user gets "this looks like a scanned or image-only PDF, run it through OCR"
instead of a generic parser failure.

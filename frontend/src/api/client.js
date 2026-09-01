/**
 * The only module that talks to the API.
 *
 * Two things happen here and nowhere else:
 *
 * 1. **Score strings become numbers.** Every score field arrives as a JSON
 *    string ("0.8500") because the backend stores them as Numeric(5,4) and the
 *    serialiser preserves precision. `"0.8500" >= 0.36` is a string/number
 *    comparison that JavaScript resolves by coercion — it happens to work — but
 *    `"0.8500" > "0.36"` is a *string* comparison that returns false, and
 *    sorting an array of them lexicographically puts "0.9" above "0.85" above
 *    "0.1". Converting at the boundary means no component can ever hit that.
 *
 * 2. **Failures become typed.** A dead backend and a rejected upload need
 *    different words on screen, and `catch (e)` around a bare fetch cannot tell
 *    them apart: fetch only rejects on network failure, and resolves happily
 *    with a 500.
 */

const BASE_URL = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

/** Every score field the API returns as a string. */
const SCORE_FIELDS = [
  'overall_score',
  'semantic_score',
  'skill_score',
  'skill_recall',
  'skill_confidence',
  'salary_min',
  'salary_max',
  'experience_min_years',
]

export class ApiError extends Error {
  /**
   * @param {string} message  what to show the user
   * @param {'network'|'client'|'server'} kind
   * @param {number|null} status
   * @param {unknown} body    parsed response body, when there was one
   */
  constructor(message, kind, status = null, body = null) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.body = body
  }

  /** The backend puts its message in `detail`; fall back to ours. */
  get detail() {
    if (this.body && typeof this.body.detail === 'string') return this.body.detail
    return this.message
  }

  /** Network failure means "is the server up", which is a different fix. */
  get isUnreachable() {
    return this.kind === 'network'
  }
}

/** Recursively convert known score fields from string to number. */
function parseScores(value) {
  if (Array.isArray(value)) return value.map(parseScores)
  if (value === null || typeof value !== 'object') return value

  const out = {}
  for (const [key, inner] of Object.entries(value)) {
    if (SCORE_FIELDS.includes(key) && typeof inner === 'string') {
      const asNumber = Number(inner)
      // A non-numeric string here would silently become NaN and poison every
      // comparison downstream, so keep the original and let it be visible.
      out[key] = Number.isNaN(asNumber) ? inner : asNumber
    } else {
      out[key] = parseScores(inner)
    }
  }
  return out
}

async function request(path, { method = 'GET', body, signal } = {}) {
  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    })
  } catch (cause) {
    // fetch rejects only for network-level failure: server down, DNS, CORS
    // preflight refused. Never for a 4xx or 5xx.
    if (cause?.name === 'AbortError') throw cause
    throw new ApiError(
      `Cannot reach the API at ${BASE_URL}.`,
      'network',
      null,
      null,
    )
  }

  let payload = null
  const text = await response.text()
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = { detail: text.slice(0, 500) }
    }
  }

  if (!response.ok) {
    const kind = response.status >= 500 ? 'server' : 'client'
    const detail =
      (payload && (payload.detail ?? payload.message)) ||
      `${method} ${path} failed with ${response.status}`
    throw new ApiError(
      typeof detail === 'string' ? detail : JSON.stringify(detail),
      kind,
      response.status,
      payload,
    )
  }

  return parseScores(payload)
}

/**
 * Create a resume from already-extracted plain text.
 *
 * NOTE: the API is `application/json` and takes `raw_text`, not a multipart
 * file. It has no PDF or DOCX handling, so extraction happens in the browser
 * (see src/lib/extractText.js) and this posts the result.
 */
export function createResume({ email, rawText, label, fileName, signal }) {
  return request('/resumes', {
    method: 'POST',
    body: {
      email,
      raw_text: rawText,
      label: label ?? null,
      file_path: fileName ?? null,
    },
    signal,
  })
}

/** Queue a search. Returns {run_id, status, stage} immediately; work continues. */
export function createSearch({ resumeId, location, remoteOnly, sources, maxResults, signal }) {
  const body = { resume_id: resumeId }
  if (location) body.location = location
  if (remoteOnly) body.remote_only = true
  if (sources?.length) body.sources = sources
  if (maxResults) body.max_results = maxResults
  return request('/searches', { method: 'POST', body, signal })
}

/** Poll target. Stop on `is_terminal` — NOT on `status`, which has two vocabularies. */
export function getRun(runId, { signal } = {}) {
  return request(`/runs/${runId}`, { signal })
}

export function getMatches({ resumeId, limit = 25, offset = 0, minScore, signal } = {}) {
  const params = new URLSearchParams({ resume_id: resumeId, limit, offset })
  if (minScore != null) params.set('min_score', minScore)
  return request(`/matches?${params}`, { signal })
}

export function getJob(jobId, { signal } = {}) {
  return request(`/jobs/${jobId}`, { signal })
}

export function getHealth({ signal } = {}) {
  return request('/health', { signal })
}

export { BASE_URL }

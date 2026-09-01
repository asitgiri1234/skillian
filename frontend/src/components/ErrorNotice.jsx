/**
 * One component for every failure surface.
 *
 * Each error says what happened and what to do next. A spinner that never ends
 * is the failure mode this exists to prevent — during a demo, "nothing is
 * happening" and "the backend is down" look identical without this.
 */
export function ErrorNotice({ title, detail, hint, onRetry, retryLabel = 'Try again' }) {
  return (
    <div className="notice notice--error" role="alert">
      <p className="notice__title">{title}</p>
      {detail && <p className="notice__detail">{detail}</p>}
      {hint && <p className="notice__hint">{hint}</p>}
      {onRetry && (
        <button type="button" className="button button--secondary" onClick={onRetry}>
          {retryLabel}
        </button>
      )}
    </div>
  )
}

/**
 * Turn any thrown value into the three things the notice needs.
 *
 * The distinction that matters: an unreachable API is an operator problem
 * ("start the server") while a 4xx is a user problem ("this file will not
 * work"). Collapsing them into "Something went wrong" sends people to fix the
 * wrong thing.
 */
export function describeError(error) {
  if (!error) return null

  if (error.name === 'ExtractionError') {
    return { title: error.message, hint: error.hint }
  }

  if (error.name === 'ApiError') {
    if (error.isUnreachable) {
      return {
        title: 'Cannot reach the Skillian API.',
        detail: error.message,
        hint: 'Start the backend with `uvicorn app.main:app --reload`, then try again. Nothing was lost.',
      }
    }
    if (error.status === 422) {
      return {
        title: 'The server could not read that resume.',
        detail: error.detail,
        hint: 'The text was extracted but the parser could not make sense of it. A plainer layout — no columns or tables — usually works.',
      }
    }
    if (error.status === 409) {
      return { title: 'That resume has not been parsed yet.', detail: error.detail }
    }
    if (error.kind === 'server') {
      return {
        title: 'The server hit an error.',
        detail: error.detail,
        hint: `HTTP ${error.status}. This is a bug on the backend, not something you did.`,
      }
    }
    return { title: 'That request was rejected.', detail: error.detail }
  }

  return { title: 'Something went wrong.', detail: String(error?.message ?? error) }
}

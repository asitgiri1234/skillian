import { useEffect, useRef, useState } from 'react'
import { getJob } from '../api/client'
import { ErrorNotice, describeError } from './ErrorNotice'

/**
 * The right-hand panel. Fetches the full job on selection.
 *
 * The match item is passed in alongside because the two carry different things:
 * `GET /jobs/{id}` knows the posting (description, every requirement it states)
 * while the match knows the *comparison* (which of those the resume has, and
 * the score breakdown). Neither is derivable from the other.
 */
export function JobDetail({ match, onClose }) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!match) return undefined
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setJob(null)

    getJob(match.job_id, { signal: controller.signal })
      .then((data) => setJob(data))
      .catch((cause) => {
        if (cause?.name !== 'AbortError') setError(cause)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    // Stacked layout only: the panel sits below the fold, so a click would
    // otherwise appear to do nothing. matchMedia rather than a resize
    // listener — this is a one-shot question, not a subscription.
    if (window.matchMedia('(max-width: 900px)').matches) {
      ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }

    // Aborting on change matters when clicking down a list quickly: without it
    // a slow earlier request can resolve last and overwrite the current job.
    return () => controller.abort()
  }, [match])

  if (!match) {
    return (
      <aside className="detail detail--empty">
        <p className="muted">Select a job to see the posting and why it matched.</p>
      </aside>
    )
  }

  return (
    <aside className="detail" ref={ref}>
      {/* Only shown once the split view has stacked; on a wide screen the list
          is still visible beside this, so there is nothing to go back to. */}
      <button type="button" className="detail__back" onClick={onClose}>
        ← Back to results
      </button>
      <h2 className="detail__title">{match.title}</h2>
      <p className="detail__company">
        {match.company}
        {match.location && ` · ${match.location}`}
        {match.is_remote && <span className="tag tag--remote">Remote</span>}
      </p>

      <p className="detail__facts">
        {match.salary_raw ? (
          <span>{match.salary_raw}</span>
        ) : (
          <span className="muted">Salary not stated</span>
        )}
        {match.posted_date && <span> · posted {match.posted_date}</span>}
      </p>

      {/* The comparison. Rendered from the match, so it is available
          immediately — no waiting on the job fetch to see why this matched. */}
      <SkillComparison match={match} />

      {loading && <p className="status" role="status">Loading the full posting…</p>}

      {error && (
        <ErrorNotice
          {...describeError(error)}
          onRetry={() => setJob(null) || setError(null)}
        />
      )}

      {job && (
        <>
          <Requirements skills={job.skills} />

          <section className="detail__section">
            <h3 className="detail__heading">Full description</h3>
            <p className="detail__description">
              {job.description || <span className="muted">No description was stored.</span>}
            </p>
          </section>
        </>
      )}

      <ScoreBreakdown match={match} />

      {match.apply_url && (
        <a
          className="button"
          href={match.apply_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          Apply on {match.company || 'the job board'}
        </a>
      )}
    </aside>
  )
}

function SkillComparison({ match }) {
  if (match.skills_unparsed) {
    return (
      <div className="notice notice--warning">
        This posting&rsquo;s requirements could not be read, so nothing was
        checked against your skills. The score reflects overall similarity only.
      </div>
    )
  }
  const matching = match.matching_skills ?? []
  const missing = match.missing_skills ?? []
  return (
    <section className="detail__section">
      <h3 className="detail__heading">Against your resume</h3>
      <div className="compare">
        <div>
          <p className="compare__label compare__label--have">
            You have ({matching.length})
          </p>
          {matching.length ? (
            <ul className="chips">
              {matching.map((s) => (
                <li key={s} className="chip chip--have">{s}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">None of its stated requirements.</p>
          )}
        </div>
        <div>
          <p className="compare__label compare__label--missing">
            Missing ({missing.length})
          </p>
          {missing.length ? (
            <ul className="chips">
              {missing.map((s) => (
                <li key={s} className="chip chip--missing">{s}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">Nothing it asks for is missing.</p>
          )}
        </div>
      </div>
    </section>
  )
}

/** What the posting asks for, required before preferred. */
function Requirements({ skills }) {
  const list = skills ?? []
  if (!list.length) {
    return (
      <section className="detail__section">
        <h3 className="detail__heading">Requirements</h3>
        <p className="muted">No requirements could be extracted from this posting.</p>
      </section>
    )
  }
  const required = list.filter((s) => s.requirement === 'required')
  const preferred = list.filter((s) => s.requirement !== 'required')

  return (
    <section className="detail__section">
      <h3 className="detail__heading">What the posting asks for</h3>
      {required.length > 0 && (
        <>
          <p className="compare__label">Required ({required.length})</p>
          <ul className="chips">
            {required.map((s) => (
              <li key={s.skill_id} className="chip chip--required">{s.name}</li>
            ))}
          </ul>
        </>
      )}
      {preferred.length > 0 && (
        <>
          <p className="compare__label">Preferred ({preferred.length})</p>
          <ul className="chips">
            {preferred.map((s) => (
              <li key={s.skill_id} className="chip chip--preferred">{s.name}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}

/**
 * The machinery, deliberately small and last.
 *
 * A reviewer needs to see that a 100% skill score on one requirement is not the
 * same as 100% on nine; a job seeker should be able to ignore all of it.
 */
function ScoreBreakdown({ match }) {
  const pct = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`)
  return (
    <details className="breakdown">
      <summary>How this score was calculated</summary>
      <dl className="breakdown__grid">
        <div><dt>Overall</dt><dd>{pct(match.overall_score)}</dd></div>
        <div><dt>Semantic</dt><dd>{pct(match.semantic_score)}</dd></div>
        <div><dt>Skill recall</dt><dd>{pct(match.skill_recall)}</dd></div>
        <div><dt>Confidence</dt><dd>{pct(match.skill_confidence)}</dd></div>
        <div><dt>Requirements read</dt><dd>{match.parsed_count ?? 0}</dd></div>
      </dl>
      <p className="breakdown__note">
        Skill recall is how much of what this job asks for you have. Confidence
        weights that by how many requirements were actually readable — matching
        one of one counts for less than matching nine of nine.
      </p>
    </details>
  )
}

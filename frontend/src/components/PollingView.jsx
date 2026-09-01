import { useEffect, useState } from 'react'
import { useRunPoller } from '../hooks/useRunPoller'
import { ErrorNotice, describeError } from './ErrorNotice'

/** Rotated every 5s, secondary to the real progress. */
const ENCOURAGEMENTS = [
  'Reading job descriptions in full, not just the headlines.',
  'Most searches turn up a few roles you would not have found by keyword.',
  'Skills are matched against what each posting actually asks for.',
  'Postings whose requirements cannot be read are kept separate, not ranked against the rest.',
  'Almost there — scoring is the fast part.',
]

const ROTATE_MS = 5000

export function PollingView({ runId, onDone, onRestart }) {
  const { run, error } = useRunPoller(runId)
  const [tick, setTick] = useState(0)

  const failed = run?.is_terminal && (run.status === 'failed' || Boolean(run.error))
  const done = run?.is_terminal && !failed

  // Handing off in an effect, not during render: setting parent state while a
  // child renders is what produces "Cannot update a component while rendering
  // a different component".
  useEffect(() => {
    if (done) onDone(run)
  }, [done, run, onDone])

  useEffect(() => {
    const timer = setInterval(() => setTick((t) => t + 1), ROTATE_MS)
    return () => clearInterval(timer)
  }, [])

  if (!run && error) {
    // Nothing has come back at all — most likely the API went away between
    // queuing the search and the first poll.
    return (
      <section className="panel">
        <ErrorNotice {...describeError(error)} onRetry={onRestart} retryLabel="Start over" />
      </section>
    )
  }

  if (!run) {
    return (
      <section className="panel">
        <p className="status" role="status">
          Starting your search…
        </p>
      </section>
    )
  }

  if (failed) {
    return (
      <section className="panel">
        <ErrorNotice
          title="The search run failed."
          detail={run.error || 'The backend did not say why.'}
          hint={
            run.jobs_found > 0
              ? `${run.jobs_found} job${run.jobs_found === 1 ? '' : 's'} were fetched before it stopped, so some sources worked.`
              : 'No jobs were fetched. Check that the job sources are configured.'
          }
          onRetry={onRestart}
          retryLabel="Run it again"
        />
      </section>
    )
  }

  // The effect above hands off; render nothing while the parent swaps views.
  if (done) return null

  const pct =
    run.stage_total > 0 ? Math.round((run.stage_number / run.stage_total) * 100) : 0

  return (
    <section className="panel">
      <h2 className="title">Searching</h2>

      {/* Real progress from the run row. The backend populates stage_label,
          stage_number and jobs_found specifically so this can be shown; a
          generic spinner would throw all of it away. */}
      <p className="stage" aria-live="polite">
        {run.stage_label ?? run.stage ?? 'Working'}
      </p>
      <p className="stage__meta">
        Step {run.stage_number} of {run.stage_total}
        {run.jobs_found > 0 && ` · ${run.jobs_found} jobs found`}
      </p>

      <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress__bar" style={{ width: `${pct}%` }} />
      </div>

      <p className="encouragement">{ENCOURAGEMENTS[tick % ENCOURAGEMENTS.length]}</p>

      {/* A transient poll failure does not stop the run on the server, so the
          last known stage stays on screen alongside the warning. */}
      {error && (
        <p className="notice notice--warning">
          Lost contact with the API — still trying. Last known stage:{' '}
          {run.stage_label ?? run.stage}.
        </p>
      )}
    </section>
  )
}

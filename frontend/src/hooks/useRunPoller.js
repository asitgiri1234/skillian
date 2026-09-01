import { useEffect, useRef, useState } from 'react'
import { getRun } from '../api/client'

export const POLL_INTERVAL_MS = 2000

/**
 * Poll `GET /runs/{id}` until the run finishes.
 *
 * Stops on **`is_terminal`**, not on `status`. Two status vocabularies share
 * that column — the day-1 CLI writes `success` and the search pipeline writes
 * `succeeded` — so `status === 'succeeded'` would poll forever against a run
 * created by the other path. `is_terminal` is the backend's own answer to
 * "should you stop asking", and it covers `partial` and `failed` too.
 *
 * Cleanup matters more than it looks. Three things can end a poll — unmount, a
 * terminal state, and a new runId replacing the old — and every one of them
 * must clear the timer. A missed clear leaves a timer hammering the API for the
 * lifetime of the tab, and because each tick still resolves and calls setState
 * on an unmounted component, it is invisible until the network tab is open.
 */
export function useRunPoller(runId) {
  const [run, setRun] = useState(null)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    if (!runId) {
      setRun(null)
      setError(null)
      return undefined
    }

    // `cancelled` guards against the in-flight request that resolves *after*
    // cleanup has run; clearTimeout cannot cancel a promise already awaiting.
    let cancelled = false
    setError(null)

    const stop = () => {
      cancelled = true
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
      abortRef.current?.abort()
      abortRef.current = null
    }

    const tick = async () => {
      if (cancelled) return
      const controller = new AbortController()
      abortRef.current = controller
      try {
        const next = await getRun(runId, { signal: controller.signal })
        if (cancelled) return
        setRun(next)
        if (next.is_terminal) {
          stop()
          return
        }
      } catch (cause) {
        if (cancelled || cause?.name === 'AbortError') return
        // A transient blip should not kill the poll — the run is still going
        // on the server. Surface it and keep trying; the UI shows the last
        // known stage alongside the warning.
        setError(cause)
      }
      if (cancelled) return
      // setTimeout chained after completion, not setInterval: with a 2s
      // interval and a slow response, setInterval would stack overlapping
      // requests. This guarantees one in flight at a time.
      timerRef.current = setTimeout(tick, POLL_INTERVAL_MS)
    }

    tick()
    return stop
  }, [runId])

  return { run, error, isPolling: Boolean(runId) && !run?.is_terminal }
}

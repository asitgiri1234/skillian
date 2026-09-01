import { useState } from 'react'

/**
 * "Upload a different resume" — a full reset back to the upload view.
 *
 * Confirms inline rather than with `confirm()`. A native dialog steals focus,
 * cannot be styled, and reads as a browser error; this is a two-button swap in
 * place. Losing a completed set of matches to a misclick is worse than one
 * extra click, but only when there is something to lose — with no results on
 * screen the confirmation is skipped entirely.
 */
export function ResetControl({ matchCount, onReset }) {
  const [confirming, setConfirming] = useState(false)

  if (!confirming) {
    return (
      <button
        type="button"
        className="reset__link"
        onClick={() => (matchCount > 0 ? setConfirming(true) : onReset())}
      >
        Upload a different resume
      </button>
    )
  }

  return (
    <span className="reset__confirm" role="alertdialog" aria-label="Confirm reset">
      <span className="reset__question">
        Discard {matchCount} match{matchCount === 1 ? '' : 'es'} and start over?
      </span>
      <button type="button" className="button button--small" onClick={onReset}>
        Discard
      </button>
      <button
        type="button"
        className="button button--small button--secondary"
        onClick={() => setConfirming(false)}
        autoFocus
      >
        Keep results
      </button>
    </span>
  )
}

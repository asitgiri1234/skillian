import { useState } from 'react'
import { createSearch } from './api/client'
import { PollingView } from './components/PollingView'
import { ResultsView } from './components/ResultsView'
import { UploadView } from './components/UploadView'
import './App.css'

/**
 * Single view with state; no router. Three phases:
 *
 *   upload  -> pick a file, see the parse, decide whether to search
 *   polling -> watch the run
 *   (part 2 replaces the terminal placeholder with results)
 */
export default function App() {
  const [email, setEmail] = useState('')
  const [resume, setResume] = useState(null)
  const [runId, setRunId] = useState(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState(null)
  const [finishedRun, setFinishedRun] = useState(null)
  // Bumped on reset so UploadView remounts with a brand-new <input type=file>.
  // Browsers do not fire `change` when the same file is re-selected unless the
  // input has been cleared, and a fresh DOM node is the surest way to clear it.
  const [uploadKey, setUploadKey] = useState(0)

  async function startSearch() {
    if (!resume) return
    setSearching(true)
    setSearchError(null)
    try {
      const run = await createSearch({ resumeId: resume.id })
      setFinishedRun(null)
      setRunId(run.run_id)
    } catch (cause) {
      setSearchError(cause)
    } finally {
      setSearching(false)
    }
  }

  /**
   * PATCH /resumes/{id}/skills deletes every match for the resume, so the only
   * correct next step is a fresh search. Showing the previous results would be
   * showing results for a skill set that no longer exists.
   *
   * The re-run is cheap: the jobs are already stored, chunked and embedded, so
   * only scoring repeats.
   */
  async function handleSkillsUpdated(updatedResume) {
    setResume(updatedResume)
    setFinishedRun(null)
    setRunId(null)
    setSearching(true)
    setSearchError(null)
    try {
      const run = await createSearch({ resumeId: updatedResume.id })
      setRunId(run.run_id)
    } catch (cause) {
      setSearchError(cause)
    } finally {
      setSearching(false)
    }
  }

  /**
   * Back to a blank upload view.
   *
   * Every piece of per-resume state goes, not just the view flag. A surviving
   * `resume` would be the real bug here: ResultsView keys its fetch on
   * `resume.id`, so a leftover id would quietly render the *previous* resume's
   * matches under a newly uploaded one.
   *
   * Clearing `runId` unmounts PollingView, and useRunPoller's effect cleanup
   * clears its timer and aborts the in-flight request — so no poller survives
   * a reset even if one was still running.
   *
   * `email` is kept: it is the same person. Nothing is deleted server-side —
   * the old resume and its matches stay keyed to their own resume_id, orphan
   * nothing, and remain reachable by user_id.
   */
  function resetAll() {
    setResume(null)
    setRunId(null)
    setFinishedRun(null)
    setSearchError(null)
    setSearching(false)
    setUploadKey((k) => k + 1)
  }

  function restart() {
    // Clearing runId is what unmounts PollingView and stops its poller.
    setRunId(null)
    setFinishedRun(null)
    setSearchError(null)
  }

  return (
    <main className={finishedRun ? 'app app--wide' : 'app'}>
      {finishedRun ? (
        <ResultsView
          /* Keyed on the run: a re-run must remount, clearing the selected
             job and the page number along with the stale match list. */
          key={finishedRun.run_id}
          resume={resume}
          jobsFound={finishedRun.jobs_found}
          onRestart={restart}
          onReset={resetAll}
          onSkillsUpdated={handleSkillsUpdated}
        />
      ) : runId ? (
        <PollingView runId={runId} onRestart={restart} onDone={setFinishedRun} />
      ) : (
        <UploadView
          key={uploadKey}
          email={email}
          onEmailChange={setEmail}
          resume={resume}
          onParsed={setResume}
          onSearch={startSearch}
          searching={searching}
          searchError={searchError}
        />
      )}
    </main>
  )
}

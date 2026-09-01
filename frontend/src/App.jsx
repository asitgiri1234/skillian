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
          resumeId={resume.id}
          jobsFound={finishedRun.jobs_found}
          onRestart={restart}
        />
      ) : runId ? (
        <PollingView runId={runId} onRestart={restart} onDone={setFinishedRun} />
      ) : (
        <UploadView
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

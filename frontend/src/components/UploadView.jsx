import { useRef, useState } from 'react'
import { createResume } from '../api/client'
import { ACCEPTED_EXTENSIONS } from '../lib/constants'
import { ErrorNotice, describeError } from './ErrorNotice'

/**
 * Upload and parse, then — as a separate action — start a search.
 *
 * Two actions on purpose. Parsing on file selection gives immediate feedback
 * that the resume was read correctly; the user checks the skills list before
 * committing to a search that takes minutes. Bundling them would mean
 * discovering a bad parse only after the wait.
 */
export function UploadView({ email, onEmailChange, resume, onParsed, onSearch, searching, searchError }) {
  const [status, setStatus] = useState('idle') // idle | extracting | parsing
  const [error, setError] = useState(null)
  const [fileName, setFileName] = useState(null)
  const inputRef = useRef(null)

  const emailValid = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim())

  async function handleFile(event) {
    const file = event.target.files?.[0]
    // Reset immediately so re-selecting the same file after an error re-fires
    // change; without this, picking the identical file twice does nothing.
    event.target.value = ''
    if (!file) return

    setError(null)
    setFileName(file.name)

    try {
      setStatus('extracting')
      // Dynamic import: pdf.js and mammoth are ~1.2MB and nothing needs them
      // until a file is actually chosen, so they stay out of the initial load.
      const { extractText } = await import('../lib/extractText')
      const rawText = await extractText(file)

      setStatus('parsing')
      const parsed = await createResume({
        email: email.trim(),
        rawText,
        label: file.name,
        fileName: file.name,
      })
      onParsed(parsed)
      setStatus('idle')
    } catch (cause) {
      setError(cause)
      setStatus('idle')
    }
  }

  const busy = status !== 'idle'
  const described = describeError(error)

  return (
    <section className="panel">
      <h1 className="title">Skillian</h1>
      <p className="subtitle">
        Upload your resume. We read it, then match it against live job postings.
      </p>

      <label className="field">
        <span className="field__label">Your email</span>
        <input
          type="email"
          className="input"
          value={email}
          placeholder="you@example.com"
          onChange={(e) => onEmailChange(e.target.value)}
          disabled={busy || Boolean(resume)}
        />
        <span className="field__hint">
          Used to keep your resumes together. No account, no password.
        </span>
      </label>

      {!resume && (
        <div className="field">
          <span className="field__label">Resume</span>
          <input
            ref={inputRef}
            type="file"
            className="input"
            accept={ACCEPTED_EXTENSIONS.join(',')}
            onChange={handleFile}
            disabled={busy || !emailValid}
          />
          <span className="field__hint">
            {emailValid
              ? 'PDF or Word document, up to 5MB. Parsing starts as soon as you choose a file.'
              : 'Enter a valid email first.'}
          </span>
        </div>
      )}

      {status === 'extracting' && (
        <p className="status" role="status">
          Reading {fileName}…
        </p>
      )}
      {status === 'parsing' && (
        <p className="status" role="status">
          Parsing your resume — this takes a couple of seconds.
        </p>
      )}

      {described && (
        <ErrorNotice
          title={described.title}
          detail={described.detail}
          hint={described.hint}
          onRetry={() => {
            setError(null)
            inputRef.current?.click()
          }}
          retryLabel="Choose another file"
        />
      )}

      {resume && <ParsedSummary resume={resume} />}

      {resume && (
        <>
          <button
            type="button"
            className="button"
            onClick={onSearch}
            disabled={searching}
          >
            {searching ? 'Starting…' : 'Search jobs'}
          </button>
          {searchError && (
            <ErrorNotice {...describeError(searchError)} onRetry={onSearch} />
          )}
        </>
      )}
    </section>
  )
}

/**
 * What the system understood. This is the user's first and best chance to catch
 * a bad parse, so it shows the skills list in full rather than a count — the
 * skills are what matching actually runs on.
 */
function ParsedSummary({ resume }) {
  const parsed = resume.parsed ?? {}
  const skills = resume.skills ?? []
  const years = parsed.total_experience_years

  return (
    <div className="parsed">
      <p className="parsed__heading">Here is what we read:</p>
      <dl className="parsed__facts">
        <div>
          <dt>Name</dt>
          <dd>{parsed.name || <span className="muted">not found</span>}</dd>
        </div>
        <div>
          <dt>Experience</dt>
          <dd>
            {years == null ? (
              <span className="muted">not found</span>
            ) : (
              `${years} year${years === 1 ? '' : 's'}`
            )}
          </dd>
        </div>
        <div>
          <dt>Skills</dt>
          <dd>{skills.length}</dd>
        </div>
      </dl>

      {skills.length > 0 ? (
        <ul className="chips">
          {skills.map((skill) => (
            <li key={skill} className="chip">
              {skill}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">
          No skills were extracted. Matching relies on these, so results will be
          based on overall similarity only.
        </p>
      )}
    </div>
  )
}

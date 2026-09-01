import { useId, useMemo, useState } from 'react'
import { updateResumeSkills } from '../api/client'
import { ErrorNotice, describeError } from './ErrorNotice'

/**
 * Edit the resume's skills, then re-run the search.
 *
 * The parser gets things wrong, and a user who cannot correct it gets bad
 * matches with no way to understand why. This is the fix, and it is the one
 * place in the interface where the user changes the input rather than reading
 * the output.
 *
 * A collapsible strip above the list rather than a third column: three columns
 * do not fit on a laptop, and this is used occasionally, not constantly.
 */
export function SkillsPanel({ resume, suggestions, disabled, onUpdated }) {
  const [open, setOpen] = useState(false)
  const [skills, setSkills] = useState(() => resume.skills ?? [])
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const listId = useId()

  const original = resume.skills ?? []
  // Order-insensitive: re-ordering the same set is not a change worth a
  // multi-minute re-run.
  const dirty = useMemo(() => {
    if (skills.length !== original.length) return true
    const a = new Set(skills.map((s) => s.toLowerCase()))
    return original.some((s) => !a.has(s.toLowerCase()))
  }, [skills, original])

  const locked = disabled || saving

  function add(raw) {
    const name = raw.trim()
    if (!name) return
    const exists = skills.some((s) => s.toLowerCase() === name.toLowerCase())
    if (!exists) setSkills((current) => [...current, name])
    setDraft('')
  }

  function remove(name) {
    setSkills((current) => current.filter((s) => s !== name))
  }

  async function submit() {
    setSaving(true)
    setError(null)
    const previous = skills
    try {
      const updated = await updateResumeSkills(resume.id, skills)
      // The PATCH already deleted this resume's matches. The parent re-runs
      // the search; there is deliberately no path here that leaves the old
      // results on screen.
      onUpdated(updated)
    } catch (cause) {
      // Restore what the user had, so a failed save does not silently discard
      // their edits along with the request.
      setSkills(previous)
      setError(cause)
    } finally {
      setSaving(false)
    }
  }

  const unused = suggestions.filter(
    (s) => !skills.some((existing) => existing.toLowerCase() === s.toLowerCase()),
  )

  return (
    <section className="skills">
      <button
        type="button"
        className="skills__toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="skills__caret">{open ? '▾' : '▸'}</span>
        Your skills
        <span className="skills__count">{skills.length}</span>
        {dirty && <span className="skills__dot" title="Unsaved changes" />}
      </button>

      {open && (
        <div className="skills__body">
          <p className="skills__note">
            Matching runs on these. Remove anything the parser got wrong, add
            what it missed, then update.
          </p>

          <ul className="chips">
            {skills.map((skill) => (
              <li key={skill} className="chip chip--editable">
                {skill}
                <button
                  type="button"
                  className="chip__remove"
                  onClick={() => remove(skill)}
                  disabled={locked}
                  aria-label={`Remove ${skill}`}
                >
                  ×
                </button>
              </li>
            ))}
            {skills.length === 0 && (
              <li className="muted">No skills — matching will use similarity only.</li>
            )}
          </ul>

          <div className="skills__add">
            <input
              className="input"
              list={listId}
              value={draft}
              placeholder="Add a skill…"
              disabled={locked}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  add(draft)
                }
              }}
            />
            {/* Suggestions come from what the fetched jobs actually ask for, so
                they are the terms that would change the ranking if added. */}
            <datalist id={listId}>
              {unused.slice(0, 200).map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
            <button
              type="button"
              className="button button--secondary"
              onClick={() => add(draft)}
              disabled={locked || !draft.trim()}
            >
              Add
            </button>
          </div>

          {error && (
            <ErrorNotice
              {...describeError(error)}
              onRetry={submit}
              retryLabel="Try updating again"
            />
          )}

          <div className="skills__actions">
            <button
              type="button"
              className="button"
              onClick={submit}
              /* One explicit submit, never auto-save on each toggle: five quick
                 edits would fire five overlapping re-scores that race and make
                 the ranking flicker. */
              disabled={locked || !dirty}
            >
              {saving ? 'Updating…' : 'Update matches'}
            </button>
            {dirty && !saving && (
              <button
                type="button"
                className="button button--ghost"
                onClick={() => setSkills(original)}
                disabled={locked}
              >
                Discard changes
              </button>
            )}
            {dirty && (
              <span className="skills__warn">
                Updating re-scores every job against the new list.
              </span>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

import { useEffect, useMemo, useState } from 'react'
import { getMatches } from '../api/client'
import { ErrorNotice, describeError } from './ErrorNotice'
import { JobDetail } from './JobDetail'
import { MatchCard } from './MatchCard'
import { SkillsPanel } from './SkillsPanel'

const PAGE_SIZE = 25

/**
 * Split view: results left, detail right.
 *
 * Selection is state, not navigation — clicking a card swaps the right panel
 * without a reload, and the list keeps its scroll position because it never
 * unmounts.
 */
export function ResultsView({ resume, jobsFound, onRestart, onSkillsUpdated }) {
  const resumeId = resume.id
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [unparsedPage, setUnparsedPage] = useState(0)
  const [selected, setSelected] = useState(null)
  const [showUnparsed, setShowUnparsed] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)

    getMatches({
      resumeId,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      unparsedLimit: PAGE_SIZE,
      unparsedOffset: unparsedPage * PAGE_SIZE,
      signal: controller.signal,
    })
      .then(setData)
      .catch((cause) => {
        if (cause?.name !== 'AbortError') setError(cause)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [resumeId, page, unparsedPage])

  const ranked = data?.ranked
  const unparsed = data?.unparsed
  const rankedPages = ranked ? Math.ceil(ranked.total / PAGE_SIZE) : 0
  // Suggestions are the skills these jobs actually ask for and the resume
  // lacks — i.e. exactly the terms that would move the ranking if added.
  const suggestions = useMemo(() => {
    const seen = new Set()
    for (const item of ranked?.items ?? []) {
      for (const skill of item.missing_skills ?? []) seen.add(skill)
      for (const skill of item.matching_skills ?? []) seen.add(skill)
    }
    return [...seen].sort((a, b) => a.localeCompare(b))
  }, [ranked])
  const unparsedPages = unparsed ? Math.ceil(unparsed.total / PAGE_SIZE) : 0

  if (loading && !data) {
    return (
      <div className="panel">
        <p className="status" role="status">
          Loading matches{jobsFound ? ` for ${jobsFound} jobs` : ''}…
        </p>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="panel">
        <ErrorNotice
          {...describeError(error)}
          onRetry={() => setPage((p) => p)}
          retryLabel="Reload matches"
        />
        <button type="button" className="button button--secondary" onClick={onRestart}>
          Start another search
        </button>
      </div>
    )
  }

  return (
    <div className="results">
      <div className="results__list">
        {/* Above the list, not a third column: three columns do not fit on a
            laptop, and this is used occasionally rather than constantly. */}
        <SkillsPanel
          resume={resume}
          suggestions={suggestions}
          onUpdated={onSkillsUpdated}
        />

        <header className="results__header">
          <h2 className="title">
            {ranked.total} match{ranked.total === 1 ? '' : 'es'}
          </h2>
          <button type="button" className="button button--secondary" onClick={onRestart}>
            New search
          </button>
        </header>

        {/* A transient failure on a later page should not blank results that
            are still on screen. */}
        {error && (
          <p className="notice notice--warning">
            Could not refresh the list — showing the last result that loaded.
          </p>
        )}

        {ranked.total === 0 ? (
          <div className="notice">
            <p className="notice__title">No matches scored above the filter.</p>
            <p className="notice__hint">
              Every posting either matched nothing on your resume, or its
              requirements could not be read — check the section below.
            </p>
          </div>
        ) : (
          <>
            {/* Order is exactly as returned. Ranking is computed server-side
                over the complete set; re-sorting here would rank one page of 25
                against itself and silently disagree with page 2. */}
            <ul className="cards">
              {ranked.items.map((match, index) => (
                <MatchCard
                  key={match.job_id}
                  match={match}
                  /* Absolute position in the full ranked set, not position
                     within this page: on page 2 the first card is #26. The API
                     returns items already ordered, so offset + index + 1 is
                     the rank. */
                  rank={page * PAGE_SIZE + index + 1}
                  rankTotal={ranked.total}
                  selected={selected?.job_id === match.job_id}
                  onSelect={setSelected}
                />
              ))}
            </ul>

            <Pager
              page={page}
              pages={rankedPages}
              total={ranked.total}
              onChange={setPage}
            />
          </>
        )}

        <section className="unparsed">
          <button
            type="button"
            className="unparsed__toggle"
            onClick={() => setShowUnparsed((v) => !v)}
            aria-expanded={showUnparsed}
          >
            {showUnparsed ? '▾' : '▸'} {unparsed.total} posting
            {unparsed.total === 1 ? '' : 's'} with unreadable requirements
          </button>

          {showUnparsed && (
            <>
              <p className="unparsed__note">
                Nothing could be checked against your skills for these, so they
                are scored on overall similarity alone and are not ranked
                against the list above. They are shown rather than hidden
                because they are a third of what was found.
              </p>
              <ul className="cards">
                {unparsed.items.map((match) => (
                  <MatchCard
                    key={match.job_id}
                    match={match}
                    selected={selected?.job_id === match.job_id}
                    onSelect={setSelected}
                  />
                ))}
              </ul>
              <Pager
                page={unparsedPage}
                pages={unparsedPages}
                total={unparsed.total}
                onChange={setUnparsedPage}
              />
            </>
          )}
        </section>
      </div>

      <JobDetail match={selected} onClose={() => setSelected(null)} />
    </div>
  )
}

/** Page buttons, not infinite scroll: position must be nameable and stable. */
function Pager({ page, pages, total, onChange }) {
  if (pages <= 1) return null
  return (
    <nav className="pager" aria-label="Pagination">
      <button
        type="button"
        className="button button--secondary"
        onClick={() => onChange(page - 1)}
        disabled={page === 0}
      >
        Previous
      </button>
      <span className="pager__status">
        Page {page + 1} of {pages} · {total} total
      </span>
      <button
        type="button"
        className="button button--secondary"
        onClick={() => onChange(page + 1)}
        disabled={page >= pages - 1}
      >
        Next
      </button>
    </nav>
  )
}

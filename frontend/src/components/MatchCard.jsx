/**
 * One result in the list. Shows only what is needed to decide whether to click.
 */

/** parsed_count at or below which a score rests on very little evidence. */
export const LOW_CONFIDENCE_MAX = 2

const TIER_LABELS = {
  strong: 'Strong match',
  moderate: 'Worth improving',
  weak: 'Weak fit',
}

/**
 * Colour *and* text, never colour alone.
 *
 * Roughly 1 in 12 men cannot reliably tell red from green, and tier is the
 * single most important signal on the card. The word carries the meaning; the
 * colour is redundant reinforcement, which is the only safe way round.
 */
export function TierBadge({ tier }) {
  // null is a real value here: unparsed matches have no tier, and rendering a
  // badge for them would assert a judgement the backend explicitly withheld.
  if (!tier) return null
  return (
    <span className={`badge badge--${tier}`}>{TIER_LABELS[tier] ?? tier}</span>
  )
}

export function MatchCard({ match, selected, onSelect }) {
  const lowConfidence =
    !match.skills_unparsed &&
    match.parsed_count != null &&
    match.parsed_count > 0 &&
    match.parsed_count <= LOW_CONFIDENCE_MAX

  // overall_score is already a number — client.js converts at the boundary, so
  // this is arithmetic and not a string coercion that happens to work.
  const percent = Math.round(match.overall_score * 100)

  return (
    <li>
      <button
        type="button"
        className={`card${selected ? ' card--selected' : ''}`}
        onClick={() => onSelect(match)}
        aria-current={selected ? 'true' : undefined}
      >
        <div className="card__head">
          <span className="card__title">{match.title}</span>
          <span className="card__score">{percent}%</span>
        </div>

        <div className="card__meta">
          {match.company || <span className="muted">unknown company</span>}
          {match.location && ` · ${match.location}`}
          {match.is_remote && <span className="tag tag--remote">Remote</span>}
        </div>

        <div className="card__badges">
          <TierBadge tier={match.tier} />
          {lowConfidence && (
            <span
              className="badge badge--thin"
              title={`Scored against only ${match.parsed_count} stated requirement${
                match.parsed_count === 1 ? '' : 's'
              }`}
            >
              Thin evidence
            </span>
          )}
        </div>

        {match.explanation && <p className="card__why">{match.explanation}</p>}
      </button>
    </li>
  )
}

/**
 * One result in the list. Shows only what is needed to decide whether to click.
 */

/** parsed_count at or below which a score rests on very little evidence. */
export const LOW_CONFIDENCE_MAX = 2

/**
 * Rank-relative language, deliberately not absolute claims.
 *
 * `overall_score` is a weighted product of recall, an evidence confidence
 * factor and a cosine rescaled from a 0.16-wide band. Nothing in this system
 * can score 90%: the best match across the whole 211-job corpus is 0.782 and
 * the measured p95 is 0.5431. "Strong match" against a mental 0-100 scale
 * therefore reads as a claim the number cannot support, and a 0.38 labelled
 * "Strong" reads as broken.
 *
 * "Top match" says the same thing honestly — this is near the top of what was
 * found, which is what the score actually measures. See DECISIONS 36.
 */
const TIER_LABELS = {
  strong: 'Top match',
  moderate: 'Possible fit',
  weak: 'Unlikely fit',
}

/**
 * The backend's explanation templates open with the *old* tier vocabulary —
 * "Strong match.", "Decent match.", "Weak fit." — because they are rendered
 * server-side and the backend is closed.
 *
 * Left alone, a card would read "TOP MATCH" in the badge and "Strong match."
 * one line below it, which is the exact claim this relabel exists to remove:
 * on a 0.38 card, "Strong match" is the sentence that reads as absurd.
 *
 * So the leading phrase is remapped at render time to the same vocabulary as
 * the badge. Only that opening clause is touched; every specific claim the
 * sentence makes — which skills matched, which are missing, how many
 * requirements it rests on — is the backend's and is passed through verbatim.
 */
const EXPLANATION_PREFIXES = [
  [/^Strong match\./, 'Top match.'],
  [/^Decent match on overall fit/, 'Possible fit on overall similarity'],
  [/^Decent match\./, 'Possible fit.'],
  [/^Weak fit\./, 'Unlikely fit.'],
]

export function relabelExplanation(text) {
  if (!text) return text
  for (const [pattern, replacement] of EXPLANATION_PREFIXES) {
    if (pattern.test(text)) return text.replace(pattern, replacement)
  }
  return text
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

export function MatchCard({ match, selected, onSelect, rank, rankTotal }) {
  const lowConfidence =
    !match.skills_unparsed &&
    match.parsed_count != null &&
    match.parsed_count > 0 &&
    match.parsed_count <= LOW_CONFIDENCE_MAX

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
          {/* Position, not a percentage. The score is rank-like rather than
              exam-like, and showing it as a percentage invites a comparison
              against a scale it does not live on. The raw number stays in the
              detail panel's breakdown for anyone who wants the machinery.
              Unparsed items are not part of the ranked ordering, so they have
              no rank and render nothing here. */}
          {rank != null && (
            <span className="card__rank">
              <span className="card__rank-n">#{rank}</span>
              <span className="card__rank-of">of {rankTotal}</span>
            </span>
          )}
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

        {match.explanation && (
          <p className="card__why">{relabelExplanation(match.explanation)}</p>
        )}
      </button>
    </li>
  )
}

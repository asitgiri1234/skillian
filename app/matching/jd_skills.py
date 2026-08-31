"""Dictionary-based skill extraction from a job description. No LLM.

Replaces the generative call in pipeline stage (d). That call cost ~45 seconds
*per job* — 80 jobs was an hour of wall clock — to do something a dictionary
lookup does for the whole batch in well under a second. The model was not
reasoning about anything; it was reading a list of technology names out of a
document and writing them back.

What is genuinely lost, stated up front: **a dictionary can only find skills it
already knows.** The LLM discovered names nobody had entered. The mitigation is
that the vocabulary grows from the resume side — every resume parse canonicalises
its skills into the `skills` table — plus the seed below. A posting asking for
something genuinely novel now yields nothing for that term, and
`skill_component` correctly reports it as unparsed rather than as a mismatch.

Two things the naive version of this gets wrong, both handled here:

* **Section context decides requirement level.** "Kubernetes" under "Nice to
  have" is not the same claim as "Kubernetes" under "Requirements".
* **Short names are ambiguous.** "Go to our website" is not the Go language and
  "R&D team" is not R. See :func:`_short_name_ok`.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.matching.skills import clean_skill_name
from app.models import Skill

logger = logging.getLogger(__name__)

REQUIRED = "required"
PREFERRED = "preferred"

# --- section markers --------------------------------------------------------
#
# Ordered longest-first within each group so that "preferred qualifications"
# wins over "qualifications" — otherwise a preferred section would be read as
# required, which is the more damaging direction of error.

_PREFERRED_MARKERS = (
    "preferred qualifications",
    "desired qualifications",
    "nice to have",
    "nice-to-have",
    "good to have",
    "would be a plus",
    "is a plus",
    "bonus points",
    "bonus skills",
    "desirable",
    "advantageous",
    "preferred skills",
    "preferred",
    "bonus",
    "optional",
)

_REQUIRED_MARKERS = (
    "minimum qualifications",
    "basic qualifications",
    "required qualifications",
    "what you'll need",
    "what you will need",
    "what we're looking for",
    "what we are looking for",
    "skills and experience",
    "technical skills",
    "key skills",
    "requirements",
    "requirement",
    "qualifications",
    "must have",
    "must-have",
    "must haves",
    "essential",
    "responsibilities",
    "you will",
    "you have",
    "the role",
    "about the role",
    "skills",
)

# A marker only counts near a line start or after a bullet — otherwise the word
# "requirements" inside a sentence would split the document mid-paragraph.
_MARKER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:[-*•\d.)\s]{0,4})?(?P<marker>"
    + "|".join(
        re.escape(m)
        for m in sorted(_PREFERRED_MARKERS + _REQUIRED_MARKERS, key=len, reverse=True)
    )
    + r")\b[ \t]*:?",
    re.IGNORECASE,
)

_PREFERRED_SET = {m.casefold() for m in _PREFERRED_MARKERS}

# --- short-name handling ----------------------------------------------------
#
# These are real skills whose names are also ordinary English. Matched only when
# the surrounding text looks like a list of technologies rather than prose.
_SHORT_NAMES: frozenset[str] = frozenset({"r", "go", "c", "c#", "c++", "d", "js"})

# Punctuation that means "this is an item in a list".
_LIST_BEFORE = set(",;/|([{•-–—\n\t")
_LIST_AFTER = set(",;/|)]}•\n\t")

# Words that disambiguate a short name, checked only as the *immediately*
# adjacent token. A window would be far too loose: "engineer" appears in
# essentially every job description, so "R&D engineer" would qualify R.
_QUALIFIERS: frozenset[str] = frozenset(
    {
        "programming", "program", "language", "languages", "lang",
        "developer", "developers", "dev", "engineer", "engineers",
        "golang", "coding", "scripting", "sharp", "dotnet", "stack",
    }
)

# Characters that positively indicate this is *not* a standalone technology
# name. "&" is the R&D case: an ampersand binds two letters into an idiom.
_DISQUALIFYING_ADJACENT = frozenset("&'’")

_TOKEN_RE = re.compile(r"[A-Za-z0-9.+#_-]+")


def _short_name_ok(text: str, start: int, end: int) -> bool:
    """Is this short-name match a technology, or an ordinary English word?

    Requires positive evidence. Two rules that matter:

    * **Start-of-string is not evidence.** "Go to our website" opens a
      sentence, and treating position 0 as a list boundary would match it.
    * **An adjacent ``&`` disqualifies outright.** "R&D" is one idiom, not the
      R language next to the D language — and it beats the qualifier check,
      because "R&D engineer" would otherwise pass on the word "engineer".
    """
    # Immediately adjacent, before any whitespace is skipped.
    if start > 0 and text[start - 1] in _DISQUALIFYING_ADJACENT:
        return False
    if end < len(text) and text[end] in _DISQUALIFYING_ADJACENT:
        return False

    before = text[:start].rstrip(" \t")
    after = text[end:].lstrip(" \t")

    if before and before[-1] in _LIST_BEFORE:
        return True
    if after and after[0] in _LIST_AFTER:
        return True

    # Only the single token on either side counts as a qualifier.
    next_token = _TOKEN_RE.match(after)
    if next_token and next_token.group(0).casefold() in _QUALIFIERS:
        return True
    previous = _TOKEN_RE.findall(before)
    if previous and previous[-1].casefold() in _QUALIFIERS:
        return True
    return False


# --- the index --------------------------------------------------------------


@dataclass(frozen=True)
class JobSkillHit:
    """One skill found in a description."""

    skill_id: UUID
    name: str
    requirement: str


@dataclass(frozen=True)
class SkillIndex:
    """A compiled surface-form matcher over the whole `skills` vocabulary."""

    pattern: re.Pattern[str] | None
    #: normalised surface form -> (skill_id, canonical display name)
    lookup: dict[str, tuple[UUID, str]]

    @property
    def size(self) -> int:
        return len(self.lookup)


_index: SkillIndex | None = None
_index_lock = threading.Lock()


def _boundary(form: str) -> str:
    r"""Word-boundary-ish pattern for one surface form.

    ``\b`` is wrong here: it is defined against ``\w``, so ``C#`` and ``C++``
    would match the ``C`` and leave the suffix, and ``.NET`` would fail at the
    leading dot. Explicit lookarounds over the character class that actually
    appears in technology names fix both.
    """
    return (
        r"(?<![A-Za-z0-9+#.])" + re.escape(form) + r"(?![A-Za-z0-9+#])"
    )


def build_index(session: Session) -> SkillIndex:
    """Compile every skill name and alias into one alternation.

    Non-canonical surface forms are skipped rather than deleted: the `skills`
    table still holds rows written before ``clean_skill_name`` existed —
    "Strong Python", "Comfortable with Docker", "5+ years of professional
    backend development". Filtering at index time keeps the fix local and leaves
    the rows (which `job_skills` may still reference) alone. See DECISIONS 24.2.

    The test is ``clean_skill_name(name) == [name]`` — *already canonical* — not
    merely "cleans to something non-empty". That distinction is load bearing and
    cost a debugging session: "Strong Python" cleans to ["Python"], which is
    truthy, so a non-empty check keeps it. Being longer than "Python", it then
    wins longest-first matching and resolves to its *own* skill_id — a different
    row from the one the resume canonicalised onto. The job and the résumé both
    say Python and the intersection is empty.
    """
    lookup: dict[str, tuple[UUID, str]] = {}
    skipped = 0

    for skill_id, name, aliases in session.execute(
        select(Skill.id, Skill.name, Skill.aliases)
    ):
        if clean_skill_name(name) != [name]:
            skipped += 1
            continue
        for form in (name, *(aliases or [])):
            form = (form or "").strip()
            if not form or len(form) < 1:
                continue
            key = form.casefold()
            # First writer wins, and canonical names are inserted before
            # aliases would overwrite them.
            lookup.setdefault(key, (skill_id, name))

    if not lookup:
        logger.warning("Skill index is empty; no skills will be extracted")
        return SkillIndex(pattern=None, lookup={})

    # Longest first, so "machine learning" beats "learning" and "C++" beats "C".
    forms = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(
        "|".join(_boundary(form) for form in forms), re.IGNORECASE
    )
    logger.info(
        "Skill index built: %s surface form(s), %s junk row(s) skipped",
        len(lookup), skipped,
    )
    return SkillIndex(pattern=pattern, lookup=lookup)


def get_index(session: Session, *, refresh: bool = False) -> SkillIndex:
    """Return the process-wide index, building it at most once.

    Lazy rather than literally at module import, deliberately: importing
    ``app.matching`` must not require a live database. Alembic, ``pytest``
    collection and ``--help`` all import this package, and an import-time query
    would make every one of them fail without Postgres. "Once per process" is
    the property that matters, and this has it.
    """
    global _index
    if _index is not None and not refresh:
        return _index
    with _index_lock:
        if _index is None or refresh:
            _index = build_index(session)
    return _index


def reset_index() -> None:
    """Drop the cached index. For tests, and after seeding new skills."""
    global _index
    with _index_lock:
        _index = None


# --- section splitting ------------------------------------------------------


def split_sections(description: str) -> list[tuple[str, str]]:
    """Split a description into ``(requirement_level, text)`` spans.

    Text before the first marker is ``required``: a posting that opens with a
    bare list of technologies is stating requirements, and defaulting to
    ``preferred`` would under-weight every such job.
    """
    if not description:
        return []

    matches = list(_MARKER_RE.finditer(description))
    if not matches:
        return [(REQUIRED, description)]

    spans: list[tuple[str, str]] = []
    head = description[: matches[0].start()].strip()
    if head:
        spans.append((REQUIRED, head))

    for index, match in enumerate(matches):
        level = (
            PREFERRED
            if match.group("marker").casefold() in _PREFERRED_SET
            else REQUIRED
        )
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(description)
        )
        body = description[start:end].strip()
        if body:
            spans.append((level, body))
    return spans


# --- extraction -------------------------------------------------------------


def extract_skills(description: str | None, index: SkillIndex) -> list[JobSkillHit]:
    """Find every known skill in ``description``, tagged by section.

    ``required`` wins when a skill appears in both sections: a posting that
    lists Python under Requirements and again under "nice to have" is asking
    for Python.
    """
    if not description or not description.strip() or index.pattern is None:
        return []

    # skill_id -> (name, level). Insertion order is document order.
    found: dict[UUID, tuple[str, str]] = {}

    for level, span in split_sections(description):
        for match in index.pattern.finditer(span):
            surface = match.group(0)
            entry = index.lookup.get(surface.casefold())
            if entry is None:
                continue
            skill_id, name = entry

            if surface.casefold() in _SHORT_NAMES and not _short_name_ok(
                span, match.start(), match.end()
            ):
                continue

            existing = found.get(skill_id)
            if existing is None:
                found[skill_id] = (name, level)
            elif existing[1] == PREFERRED and level == REQUIRED:
                found[skill_id] = (name, REQUIRED)

    return [
        JobSkillHit(skill_id=skill_id, name=name, requirement=level)
        for skill_id, (name, level) in found.items()
    ]

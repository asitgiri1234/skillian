"""Split a job description into embeddable passages.

Why chunk at all: a job description is several documents stapled together —
a company blurb, responsibilities, hard requirements, nice-to-haves, benefits,
and an equal-opportunity paragraph that is word-for-word identical across
thousands of postings. Embedding the whole thing produces one vector dominated
by whichever section is longest, which is usually not the one a candidate should
be matched on.

The unit of splitting is the *block* (a paragraph, a heading, or a run of
bullets), never the raw character offset. Cutting mid-sentence produces a
fragment that embeds to nothing useful, so a block is only ever split further
when it alone exceeds the maximum, and then only on sentence boundaries.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Targets, in estimated tokens. A chunk is closed once it reaches TARGET_MIN and
# the next block would push it past TARGET_MAX.
TARGET_MIN_TOKENS = 200
TARGET_MAX_TOKENS = 400

# A chunk below this is merged into its neighbour. A 20-token fragment — a bare
# heading, a one-line sign-off — embeds to a near-meaningless point that would
# nonetheless compete for a top-3 slot in semantic_component.
MIN_CHUNK_TOKENS = 50

# Words-to-tokens for English prose. Measured against nomic-embed-text's
# tokenizer on a sample of job descriptions: subword splits and punctuation put
# it a little above 1.3, and rounding up is the safe direction (it makes chunks
# slightly smaller than the target rather than overflowing the model's window).
_TOKENS_PER_WORD = 1.35

# Blank line = paragraph boundary. \r\n normalised away before this is applied.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

# Sentence boundary: terminator, then whitespace, then a capital or a digit.
# Deliberately crude — it only has to break up an oversized block, and the cost
# of an occasional bad split inside one chunk is far lower than the cost of a
# dependency on a sentence tokenizer.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Lines that start a new section of a JD. Matched case-insensitively against the
# start of a short line, so "Requirements", "REQUIREMENTS:" and "What you'll do"
# all fire, but a sentence that happens to contain the word "requirements" does
# not (see _is_section_heading's length gate).
_SECTION_WORDS = (
    "responsibilit",
    "requirement",
    "qualification",
    "what you.ll do",
    "what we.re looking for",
    "who you are",
    "about (you|us|the role|the team|the company)",
    "must have",
    "nice to have",
    "preferred",
    "skills",
    "experience",
    "benefit",
    "perks",
    "compensation",
    "salary",
    "how to apply",
    "equal opportunit",
    "diversity",
)
_SECTION_RE = re.compile(
    r"^\s*(?:[#*\-•]\s*)?(?:" + "|".join(_SECTION_WORDS) + r")",
    re.IGNORECASE,
)

# A bullet continues the block above it rather than starting a new one.
_BULLET_RE = re.compile(r"^\s*(?:[-*•‣◦⁃∙]|\d+[.)])\s+")

_WHITESPACE_RE = re.compile(r"[ \t]+")


def estimate_tokens(text: str) -> int:
    """Approximate token count for ``text``.

    A word-count heuristic rather than a real tokenizer. Bringing in tiktoken (or
    worse, the model's own tokenizer over HTTP) to decide where to cut a
    paragraph would be a dependency and a round-trip in service of a boundary
    that is approximate by nature — the chunk targets are a range, not a limit
    anything enforces.
    """
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _is_section_heading(line: str) -> bool:
    """True for a line that reads as a section header rather than prose.

    The length gate does most of the work: headings are short. Without it,
    "We have no formal requirements for this role beyond..." would split.
    """
    stripped = line.strip()
    if not stripped or len(stripped.split()) > 8:
        return False
    if _BULLET_RE.match(stripped):
        return False
    # An explicit colon or an ALL-CAPS line is a heading regardless of wording.
    if stripped.endswith(":"):
        return True
    if stripped.isupper() and len(stripped) > 2:
        return True
    return bool(_SECTION_RE.match(stripped))


def _clean(text: str) -> str:
    """Normalise line endings and horizontal whitespace, keep paragraph breaks."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    normalised = _WHITESPACE_RE.sub(" ", normalised)
    # Collapse 3+ newlines to exactly two so paragraph splitting is predictable.
    normalised = re.sub(r"\n{3,}", "\n\n", normalised)
    return normalised.strip()


def _split_blocks(text: str) -> list[tuple[str, bool]]:
    """Break the description into ``(block_text, starts_section)`` pairs.

    A block is a paragraph, except that a heading line always begins a new block
    (job boards very often emit "Requirements" and the list under it as one
    paragraph with single newlines) and bullets stay attached to what precedes
    them.
    """
    blocks: list[tuple[str, bool]] = []

    for paragraph in _PARAGRAPH_RE.split(text):
        if not paragraph.strip():
            continue

        current: list[str] = []
        starts_section = False

        for line in paragraph.split("\n"):
            if _is_section_heading(line) and current:
                blocks.append(("\n".join(current).strip(), starts_section))
                current = []
                starts_section = True
            elif _is_section_heading(line):
                starts_section = True
            current.append(line)

        if current:
            blocks.append(("\n".join(current).strip(), starts_section))

    return [(body, flag) for body, flag in blocks if body]


def _split_oversized(block: str) -> list[str]:
    """Break a single block that exceeds TARGET_MAX_TOKENS on sentence bounds.

    Only reached for a wall-of-text description with no paragraph breaks, which
    is common enough on scraped boards to be worth handling. If even one
    "sentence" is oversized (no punctuation at all), it is emitted as-is rather
    than cut mid-word.
    """
    sentences = _SENTENCE_RE.split(block)
    if len(sentences) == 1:
        return [block]

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = estimate_tokens(sentence)
        if current and current_tokens + tokens > TARGET_MAX_TOKENS:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += tokens

    if current:
        parts.append(" ".join(current))
    return parts


def _merge_short_chunks(chunks: list[str]) -> list[str]:
    """Fold any chunk under MIN_CHUNK_TOKENS into its neighbour.

    Backwards into the previous chunk, as specified. The first chunk has no
    previous one, so it merges *forwards* instead — otherwise a description that
    opens with a one-line title would keep a 6-token chunk forever.
    """
    if len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    for chunk in chunks:
        if merged and estimate_tokens(chunk) < MIN_CHUNK_TOKENS:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)

    # The forward case: the head is short and everything after it merged
    # elsewhere, so fix it in one pass at the end.
    if len(merged) > 1 and estimate_tokens(merged[0]) < MIN_CHUNK_TOKENS:
        merged[1] = f"{merged[0]}\n\n{merged[1]}"
        merged.pop(0)

    return merged


def chunk_description(text: str | None) -> list[str]:
    """Split a job description into ordered, embeddable chunks.

    Order is preserved throughout: chunk *i* always precedes chunk *i+1* in the
    original document, which is what makes ``job_chunks.chunk_index`` meaningful.

    A short description yields exactly one chunk — that is the common case for
    aggregator postings, which are often a two-line teaser, and it must not
    produce zero chunks or the job becomes unscoreable.

    Returns ``[]`` only for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return []

    cleaned = _clean(text)
    blocks = _split_blocks(cleaned)
    if not blocks:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
            current, current_tokens = [], 0

    for body, starts_section in blocks:
        for part in (
            _split_oversized(body)
            if estimate_tokens(body) > TARGET_MAX_TOKENS
            else [body]
        ):
            tokens = estimate_tokens(part)

            # A section heading is a natural seam, but only worth taking once the
            # open chunk is already big enough to stand alone. Splitting at every
            # heading would produce a chunk per bullet list.
            if starts_section and current_tokens >= TARGET_MIN_TOKENS:
                flush()
            # Otherwise close when adding this block would overflow, and the open
            # chunk has enough in it to be worth closing.
            elif (
                current
                and current_tokens + tokens > TARGET_MAX_TOKENS
                and current_tokens >= TARGET_MIN_TOKENS
            ):
                flush()

            current.append(part)
            current_tokens += tokens
            # Only the first part of a split block inherits the section flag.
            starts_section = False

    flush()

    result = _merge_short_chunks(chunks)
    logger.debug(
        "Chunked %s chars into %s chunk(s): %s tokens",
        len(text), len(result), [estimate_tokens(c) for c in result],
    )
    return result

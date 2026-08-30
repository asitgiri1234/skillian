"""Text normalisation and the cross-source dedup hash.

This lives outside ``app/sources/`` on purpose: every source must produce the
*same* hash for the same posting, so the rule cannot be a per-source detail.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Legal-form suffixes only. Words like "technologies" or "labs" are part of a
# company's actual identity and distinguish real companies, so they stay.
_COMPANY_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "pvt",
        "private",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
        "gmbh",
        "mbh",
        "ag",
        "bv",
        "nv",
        "sa",
        "sas",
        "sarl",
        "srl",
        "spa",
        "pte",
        "pty",
        "oy",
        "ab",
        "as",
        "kk",
        "sdn",
        "bhd",
    }
)

# Unicode-aware: keeps letters/digits/whitespace, drops everything else. Matching
# "not word chars" would also delete non-Latin scripts we want to preserve.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Returns "" for None/blank so callers never have to special-case missing
    fields when building a hash.
    """
    if not value:
        return ""
    # NFKD first so "Café" and "Café" normalise identically; combining
    # marks are then dropped as non-alphanumeric by the punctuation pass.
    text = unicodedata.normalize("NFKD", value)
    text = text.casefold()
    # Hyphens/slashes join words ("full-stack"), so they become spaces rather
    # than vanishing, which would produce "fullstack".
    text = re.sub(r"[-/_]+", " ", text)
    text = _PUNCT_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_company(value: str | None) -> str:
    """Normalise a company name and drop trailing legal-form suffixes.

    Suffixes are stripped repeatedly from the end so "Acme Pvt. Ltd." and
    "Acme Private Limited" both reduce to "acme".
    """
    tokens = normalize_text(value).split()
    # Stop at one token: a company genuinely named "Co" must not become "".
    while len(tokens) > 1 and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def extract_city(location: str | None) -> str:
    """Best-effort city from a free-text location string.

    Takes the first comma-separated segment because job boards order location
    components most-specific-first ("Bengaluru, Karnataka, India"). Falls back to
    the whole string when there is no comma.
    """
    if not location:
        return ""
    return location.split(",")[0].strip()


def compute_dedup_hash(
    company: str | None, title: str | None, city: str | None
) -> str:
    """sha256 of normalised company + title + city.

    Fields are joined with a NUL byte rather than concatenated directly so that
    ("ab", "c") and ("a", "bc") cannot collide into the same digest.
    """
    parts = [normalize_company(company), normalize_text(title), normalize_text(city)]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

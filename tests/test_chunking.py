"""Chunking. Pure — no database, no network, no model."""

from __future__ import annotations

import pytest

from app.matching.chunking import (
    MIN_CHUNK_TOKENS,
    TARGET_MAX_TOKENS,
    chunk_description,
    estimate_tokens,
)


def _words(count: int, word: str = "engineering") -> str:
    return " ".join([word] * count)


class TestEmptyInput:
    @pytest.mark.parametrize("value", [None, "", "   ", "\n\n\t "])
    def test_produces_no_chunks(self, value: str | None) -> None:
        assert chunk_description(value) == []


class TestShortDescriptions:
    def test_a_short_jd_produces_exactly_one_chunk(self) -> None:
        """The common case for aggregator postings, and it must not yield zero —
        a job with no chunks is unscoreable on the semantic component."""
        text = "Backend engineer needed. Python and Postgres. Apply within."
        assert len(chunk_description(text)) == 1

    def test_a_single_sentence_survives(self) -> None:
        chunks = chunk_description("We are hiring a Python developer.")
        assert chunks == ["We are hiring a Python developer."]

    def test_short_paragraphs_are_merged_not_emitted_separately(self) -> None:
        """Three 6-token paragraphs are one chunk, not three near-meaningless
        fragments competing for a top-3 slot."""
        text = "First short line here.\n\nSecond short line.\n\nThird short line."
        assert len(chunk_description(text)) == 1


class TestOrdering:
    def test_order_is_preserved(self) -> None:
        """chunk_index is only meaningful if this holds."""
        sections = [f"SECTION {i}\n{_words(150, f'word{i}')}" for i in range(5)]
        chunks = chunk_description("\n\n".join(sections))
        positions = [
            next(i for i, chunk in enumerate(chunks) if f"word{n}" in chunk)
            for n in range(5)
        ]
        assert positions == sorted(positions)

    def test_no_content_is_lost(self) -> None:
        text = "\n\n".join(f"Paragraph {i}. {_words(80)}" for i in range(6))
        chunks = chunk_description(text)
        for i in range(6):
            assert any(f"Paragraph {i}." in chunk for chunk in chunks)


class TestSizing:
    def test_long_description_is_split(self) -> None:
        text = "\n\n".join(_words(120, f"topic{i}") for i in range(10))
        chunks = chunk_description(text)
        assert len(chunks) > 1

    def test_no_chunk_is_below_the_minimum_unless_it_is_the_only_one(self) -> None:
        text = "\n\n".join(_words(150, f"topic{i}") for i in range(8))
        chunks = chunk_description(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert estimate_tokens(chunk) >= MIN_CHUNK_TOKENS

    def test_a_trailing_fragment_merges_backwards(self) -> None:
        """A one-line sign-off after a long body joins the body."""
        text = _words(300) + "\n\nApply now."
        chunks = chunk_description(text)
        assert "Apply now." in chunks[-1]
        assert estimate_tokens(chunks[-1]) >= MIN_CHUNK_TOKENS

    def test_a_leading_fragment_merges_forwards(self) -> None:
        """The first chunk has no previous chunk to merge into, so it must go
        the other way rather than survive as a 3-token chunk."""
        text = "Job Title\n\n" + _words(300)
        chunks = chunk_description(text)
        assert "Job Title" in chunks[0]
        assert estimate_tokens(chunks[0]) >= MIN_CHUNK_TOKENS

    def test_wall_of_text_is_split_on_sentences(self) -> None:
        """A description with no paragraph breaks at all, which scraped boards
        produce constantly."""
        text = " ".join(f"Sentence number {i} about backend work." for i in range(200))
        chunks = chunk_description(text)
        assert len(chunks) > 1
        # Split on sentence bounds, so no chunk ends mid-word.
        for chunk in chunks:
            assert not chunk.endswith("Sentenc")

    def test_unsplittable_block_is_emitted_rather_than_cut_mid_word(self) -> None:
        """No sentence punctuation anywhere — better one oversized chunk than a
        fragment cut at an arbitrary character."""
        text = _words(900)
        chunks = chunk_description(text)
        assert len(chunks) == 1
        assert estimate_tokens(chunks[0]) > TARGET_MAX_TOKENS


class TestSectionBoundaries:
    def test_splits_at_a_section_heading(self) -> None:
        text = (
            "About the role\n" + _words(250) + "\n\n"
            "Requirements:\n" + _words(250) + "\n\n"
            "Benefits:\n" + _words(250)
        )
        chunks = chunk_description(text)
        assert len(chunks) >= 3
        # The heading leads its own chunk rather than trailing the previous one.
        assert any(chunk.startswith("Requirements:") for chunk in chunks)

    def test_does_not_split_on_a_sentence_containing_a_heading_word(self) -> None:
        """The length gate: 'requirements' inside prose is not a heading."""
        text = (
            "We have no formal requirements for this role beyond a willingness "
            "to learn and a track record of shipping software. " + _words(60)
        )
        assert len(chunk_description(text)) == 1

    def test_a_heading_alone_does_not_split_a_tiny_chunk(self) -> None:
        """Splitting at every heading would produce a chunk per bullet list."""
        text = "Requirements:\n- Python\n\nBenefits:\n- Health cover"
        assert len(chunk_description(text)) == 1

    def test_all_caps_line_is_a_heading(self) -> None:
        text = "ABOUT US\n" + _words(250) + "\n\nRESPONSIBILITIES\n" + _words(250)
        chunks = chunk_description(text)
        assert len(chunks) >= 2

    def test_bullets_stay_with_their_heading(self) -> None:
        text = "Requirements:\n- Python\n- PostgreSQL\n- Docker\n" + _words(250)
        chunks = chunk_description(text)
        assert "- Python" in chunks[0]


class TestNormalisation:
    def test_windows_line_endings(self) -> None:
        crlf = chunk_description("First para.\r\n\r\nSecond para.")
        lf = chunk_description("First para.\n\nSecond para.")
        assert crlf == lf

    def test_excess_blank_lines_do_not_create_empty_chunks(self) -> None:
        chunks = chunk_description("First.\n\n\n\n\n\nSecond.")
        assert all(chunk.strip() for chunk in chunks)


class TestEstimateTokens:
    def test_scales_with_word_count(self) -> None:
        assert estimate_tokens(_words(100)) > estimate_tokens(_words(50))

    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_overestimates_word_count(self) -> None:
        """Rounding up is the safe direction — chunks come in under target
        rather than overflowing the embedding model's window."""
        assert estimate_tokens(_words(100)) >= 100

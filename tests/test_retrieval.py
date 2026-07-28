"""Unit tests for the pure retrieval logic.

Deliberately free of FAISS, Ollama and the 227 MB index, so they run in a
second and can gate a commit.

    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import (  # noqa: E402
    is_false_positive,
    looks_like_aggregate,
    normalise_dates_in_query,
    reciprocal_rank_fusion,
)


class TestFalsePositives:
    @pytest.mark.parametrize(
        "span",
        ["10 m", "1500 dbar", "35 PSU", "12.5 km", "platform 1902029",
         "float 4901234", "cycle 300", "45N", "30.5 W", "4 knots"],
    )
    def test_measurements_and_ids_rejected(self, span):
        assert is_false_positive(span)

    @pytest.mark.parametrize(
        "span",
        ["January 3rd 2023", "last week", "March 2024", "yesterday", "3rd of May"],
    )
    def test_real_dates_accepted(self, span):
        assert not is_false_positive(span)


class TestDateNormalisation:
    def test_natural_date_becomes_iso(self):
        out = normalise_dates_in_query("temperature on January 3rd, 2023")
        assert "2023-01-03" in out

    def test_iso_dates_left_alone(self):
        query = "temperature on 2023-01-03"
        assert normalise_dates_in_query(query) == query

    def test_query_without_dates_unchanged(self):
        query = "what is the salinity profile of this float"
        assert normalise_dates_in_query(query) == query

    def test_measurement_does_not_disable_other_spans(self):
        """The old code tested the whole query against the false-positive
        patterns, so any mention of a unit stopped every date from parsing."""
        out = normalise_dates_in_query("temperature at 10 m on January 3rd, 2023")
        assert "2023-01-03" in out
        assert "10 m" in out

    def test_coordinates_are_not_parsed_as_dates(self):
        out = normalise_dates_in_query("temperature near 45N 30W")
        assert out == "temperature near 45N 30W"

    def test_longest_span_replaced_first(self):
        out = normalise_dates_in_query("readings from January 3rd, 2023 please")
        assert out.count("2023-01-03") == 1


class TestAggregateDetection:
    @pytest.mark.parametrize(
        "query",
        ["what was the average temperature", "highest salinity recorded",
         "how many floats reported", "total profiles in 2023",
         "show the trend over time", "minimum pressure"],
    )
    def test_aggregates_detected(self, query):
        assert looks_like_aggregate(query)

    @pytest.mark.parametrize(
        "query",
        ["temperature on 2023-01-03 near 45N", "salinity at platform 1902029",
         "what did float 4901234 record"],
    )
    def test_pointwise_queries_not_flagged(self, query):
        assert not looks_like_aggregate(query)


def _doc(row_index, text="x"):
    return Document(page_content=text, metadata={"row_index": row_index})


class TestReciprocalRankFusion:
    def test_document_in_both_lists_outranks_singletons(self):
        dense = [_doc(1), _doc(2), _doc(3)]
        lexical = [_doc(4), _doc(1), _doc(5)]
        fused = reciprocal_rank_fusion([dense, lexical], [0.5, 0.5])
        assert fused[0].metadata["row_index"] == 1

    def test_deduplicates_by_row_index(self):
        fused = reciprocal_rank_fusion([[_doc(1), _doc(2)], [_doc(1)]], [0.5, 0.5])
        assert [d.metadata["row_index"] for d in fused] == [1, 2]

    def test_weights_shift_the_ranking(self):
        dense = [_doc(1)]
        lexical = [_doc(2)]
        assert reciprocal_rank_fusion([dense, lexical], [0.9, 0.1])[0].metadata["row_index"] == 1
        assert reciprocal_rank_fusion([dense, lexical], [0.1, 0.9])[0].metadata["row_index"] == 2

    def test_top_k_truncates(self):
        docs = [[_doc(i) for i in range(10)]]
        assert len(reciprocal_rank_fusion(docs, [1.0], top_k=3)) == 3

    def test_empty_input(self):
        assert reciprocal_rank_fusion([[], []], [0.5, 0.5]) == []

    def test_docs_without_row_index_fall_back_to_content(self):
        a = Document(page_content="alpha", metadata={})
        b = Document(page_content="beta", metadata={})
        fused = reciprocal_rank_fusion([[a, b], [a]], [0.5, 0.5])
        assert len(fused) == 2
        assert fused[0].page_content == "alpha"


class TestSpanRecovery:
    def test_preposition_preserved_around_rewritten_date(self):
        out = normalise_dates_in_query("temperature at 10 m on January 3rd, 2023")
        assert out == "temperature at 10 m on 2023-01-03"

    def test_bare_year_is_not_expanded_to_a_guessed_day(self):
        out = normalise_dates_in_query("profiles collected in 2023")
        assert "2023-07" not in out and "2023-01-01" not in out

    def test_platform_id_survives_rewriting(self):
        out = normalise_dates_in_query("platform 1902029 on March 5th 2024")
        assert "platform 1902029" in out and "2024-03-05" in out


class TestReasoningStrip:
    def test_think_block_removed(self):
        from RAG_main import strip_reasoning
        out = strip_reasoning("<think>let me check the records</think>Temp was 6.9 C")
        assert out == "Temp was 6.9 C"

    def test_multiline_think_block_removed(self):
        from RAG_main import strip_reasoning
        out = strip_reasoning("<think>\nline one\nline two\n</think>\nAnswer here")
        assert out == "Answer here"

    def test_unclosed_opening_tag_left_alone(self):
        from RAG_main import strip_reasoning
        assert "truncated" in strip_reasoning("<think>truncated mid thought")

    def test_stray_closing_tag_drops_preceding_text(self):
        from RAG_main import strip_reasoning
        assert strip_reasoning("reasoning spill</think>Real answer") == "Real answer"

    def test_plain_answer_untouched(self):
        from RAG_main import strip_reasoning
        assert strip_reasoning("Temperature was 6.896 C") == "Temperature was 6.896 C"

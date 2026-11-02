"""Test word-frequency bar labels, values, and input preservation."""

from datetime import datetime

import pandas as pd
import pytest
from matplotlib.figure import Figure

from search_engine import Corpus, Document, SearchEngine
from search_engine.visualizations import plot_word_frequencies


def test_plot_word_frequencies() -> None:
    """Each word has a horizontal bar showing its total occurrences."""
    table = pd.DataFrame(
        {"tf": [2, 2, 1], "df": [1, 2, 1]}, index=["cat", "dog", "fish"],
    )

    figure = plot_word_frequencies(table)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["cat", "dog", "fish"]
    assert [bar.get_width() for bar in axes.patches] == [2, 2, 1]
    assert axes.yaxis_inverted()


def test_plot_preserves_table() -> None:
    """The plot follows the supplied order without modifying the table."""
    table = pd.DataFrame(
        {"tf": [1, 2], "df": [1, 1]}, index=["fish", "cat"],
    )
    original = table.copy()

    figure = plot_word_frequencies(table)

    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["fish", "cat"]
    assert [bar.get_width() for bar in axes.patches] == [1, 2]
    pd.testing.assert_frame_equal(table, original)


def test_plot_selected_statistics(corpus: Corpus) -> None:
    """The chart reuses the limited selection and alphabetical frequency ties."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "dog fish"))
    engine = SearchEngine(corpus)
    freq = engine.freq

    figure = plot_word_frequencies(SearchEngine.get_word_frequencies(freq, 2))

    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["cat", "dog"]
    assert [bar.get_width() for bar in axes.patches] == [2, 2]


def test_plot_empty_table() -> None:
    """An empty selection still returns a figure without bars or word labels."""
    table = pd.DataFrame(columns=["tf", "df"], dtype=int)

    figure = plot_word_frequencies(table)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].patches) == 0
    assert figure.axes[0].get_yticklabels() == []
    assert table.empty
    assert list(table.columns) == ["tf", "df"]


@pytest.mark.parametrize(
    ("n", "expected_words", "expected_values"),
    [
        (2, ["cat", "dog"], [2, 2]),
        (10, ["cat", "dog", "fish"], [2, 2, 1]),
        (0, [], []),
    ],
)
def test_plot_after_tfidf(
    corpus: Corpus,
    n: int,
    expected_words: list[str],
    expected_values: list[int],
) -> None:
    """Charts retain occurrence counts after weighting, including zero-IDF terms."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "dog cat cat"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "fish dog"))
    engine = SearchEngine(corpus)
    document_counts = engine.document_counts
    vocab, freq = engine.vocab, engine.freq
    original_freq = freq.copy()
    idf, mat_TFxIDF = engine.idf, engine.mat_TFxIDF
    table = SearchEngine.get_word_frequencies(freq, n)
    original_table = table.copy()

    figure = plot_word_frequencies(table)

    assert idf[vocab["dog"]["id"]] == 0
    assert mat_TFxIDF[:, vocab["dog"]["id"]].nnz == 0
    assert isinstance(figure, Figure)
    axes = figure.axes[0]
    assert len(axes.patches) == len(expected_words)
    assert [label.get_text() for label in axes.get_yticklabels()] == expected_words
    assert [bar.get_width() for bar in axes.patches] == expected_values
    assert axes.yaxis_inverted()
    pd.testing.assert_frame_equal(freq, original_freq)
    pd.testing.assert_frame_equal(table, original_table)


@pytest.mark.parametrize("text", [None, "... ___"])
def test_plot_without_terms_after_tfidf(corpus: Corpus, text: str | None) -> None:
    """Empty vocabularies still produce a Figure after the full analysis pipeline."""
    if text is not None:
        corpus.add(Document("Empty", "Alice", datetime(2026, 1, 1), "", text))
    engine = SearchEngine(corpus)
    document_counts = engine.document_counts
    vocab, freq = engine.vocab, engine.freq
    idf, mat_TFxIDF = engine.idf, engine.mat_TFxIDF

    figure = plot_word_frequencies(SearchEngine.get_word_frequencies(freq, 10))

    assert idf == []
    assert mat_TFxIDF.shape == (len(corpus), 0)
    assert isinstance(figure, Figure)
    assert len(figure.axes[0].patches) == 0
    assert figure.axes[0].get_yticklabels() == []
    assert freq.empty
    assert list(freq.columns) == ["tf", "df"]

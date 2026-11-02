"""Test word-frequency bar labels, values, and input preservation."""

from datetime import datetime

import pandas as pd
from matplotlib.figure import Figure

from search_engine import Corpus, Document
from search_engine.search.SearchEngine import analyser_corpus, get_word_frequencies
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
    _, freq, _ = analyser_corpus(corpus)

    figure = plot_word_frequencies(get_word_frequencies(freq, 2))

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

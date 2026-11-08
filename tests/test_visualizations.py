"""Test figures built from word frequencies and document dates."""

from datetime import UTC, datetime, timedelta, timezone
from math import log

import pandas as pd
import pytest
from matplotlib.figure import Figure
from wordcloud import WordCloud

from search_engine import Corpus, Document, SearchEngine
from search_engine.visualizations import (
    filter_stopwords,
    plot_temporal_evolution,
    plot_tfidf_word_cloud,
    plot_word_cloud,
    plot_word_frequencies,
)


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


def test_plot_word_cloud() -> None:
    """Create a deterministic cloud from frequencies without modifying them."""
    table = pd.DataFrame({"tf": [3, 2, 1]}, index=["cat", "dog", "fish"])
    original = table.copy()

    figure = plot_word_cloud(table)
    repeated = plot_word_cloud(table)

    assert isinstance(figure, Figure)
    image = figure.axes[0].images[0].get_array()
    assert image.shape == (400, 800, 3)
    assert (image == repeated.axes[0].images[0].get_array()).all()
    pd.testing.assert_frame_equal(table, original)


def test_filter_stopwords() -> None:
    """Remove common words before ranking without changing term counts."""
    table = pd.DataFrame(
        {"tf": [100, 80, 70, 3, 2], "df": [2, 2, 2, 1, 1]},
        index=["The", "that", "a", "neural", "learning"],
    )
    original = table.copy()

    filtered = filter_stopwords(table)
    selected = SearchEngine.get_word_frequencies(filtered, 2)

    assert selected["tf"].to_dict() == {"neural": 3, "learning": 2}
    assert selected["df"].to_dict() == {"neural": 1, "learning": 1}
    pd.testing.assert_frame_equal(table, original)


def test_filter_only_stopwords() -> None:
    """Allow an empty cloud after removing every supplied word."""
    table = pd.DataFrame({"tf": [3, 2, 1]}, index=["the", "that", "a"])

    filtered = filter_stopwords(table)
    figure = plot_word_cloud(filtered)

    assert filtered.empty
    assert list(filtered.columns) == ["tf"]
    assert not figure.axes[0].images
    assert filter_stopwords(filtered).empty


def test_tfidf_word_cloud(corpus: Corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use summed TF-IDF weights and omit ubiquitous words before selection."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "the " * 50 + "cat cat"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "the " * 50 + "fish cat"))
    corpus.add(Document("Third", "Carol", datetime(2026, 1, 1), "", "the dog"))
    engine = SearchEngine(corpus)
    original_freq = engine.freq.copy()
    original_matrix = engine.mat_TFxIDF.copy()
    captured = []
    generate = WordCloud.generate_from_frequencies

    def record_weights(cloud, frequencies, **kwargs):
        """Record weights and generate the real word-cloud image."""
        if not kwargs:
            captured.append(frequencies.copy())
        return generate(cloud, frequencies, **kwargs)

    monkeypatch.setattr(WordCloud, "generate_from_frequencies", record_weights)

    figure = plot_tfidf_word_cloud(engine, 2)

    assert isinstance(figure, Figure) and len(figure.axes[0].images) == 1
    assert list(captured[0]) == ["cat", "dog"]
    assert captured[0] == pytest.approx({"cat": 3 * log(3 / 2), "dog": log(3)})
    pd.testing.assert_frame_equal(engine.freq, original_freq)
    assert (engine.mat_TFxIDF != original_matrix).nnz == 0


@pytest.mark.parametrize("texts,n", [([], 100), (["the", "the"], 100), (["cat", "fish"], 0)])
def test_empty_tfidf_word_cloud(corpus: Corpus, texts: list[str], n: int) -> None:
    """Handle empty corpora, entirely zero weights, and a zero word limit."""
    for number, text in enumerate(texts):
        corpus.add(Document(str(number), "Alice", datetime(2026, 1, 1), "", text))

    figure = plot_tfidf_word_cloud(SearchEngine(corpus), n)

    assert isinstance(figure, Figure) and not figure.axes[0].images


def test_plot_temporal_evolution(corpus: Corpus) -> None:
    """Group UTC days, sort unsorted dates, and include days without documents."""
    dates = [
        datetime(2026, 1, 3, 12, tzinfo=UTC),
        datetime(2026, 1, 1, 12),
        datetime(2026, 1, 2, 1, tzinfo=timezone(timedelta(hours=2))),
    ]
    for number, date in enumerate(dates):
        corpus.add(Document(str(number), "Alice", date, "", "cat"))

    figure = plot_temporal_evolution(corpus.documents.values())

    assert isinstance(figure, Figure)
    line = figure.axes[0].lines[0]
    assert list(line.get_ydata()) == [2, 0, 1]
    assert [date.date().isoformat() for date in line.get_xdata()] == [
        "2026-01-01", "2026-01-02", "2026-01-03",
    ]
    assert [document.date for document in corpus.documents.values()] == dates


def test_empty_visualizations() -> None:
    """Return usable figures when no words or documents are available."""
    cloud = plot_word_cloud(pd.DataFrame(columns=["tf", "df"], dtype=int))
    timeline = plot_temporal_evolution([])

    assert isinstance(cloud, Figure) and not cloud.axes[0].images
    assert isinstance(timeline, Figure) and not timeline.axes[0].lines


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

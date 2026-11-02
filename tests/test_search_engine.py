"""Test text cleaning, vocabulary counts, and frequency displays."""

from datetime import datetime

import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from search_engine import Corpus, Document
from search_engine.search.SearchEngine import (
    analyser_corpus,
    construire_index,
    get_word_frequencies,
    nettoyer_texte,
    stats,
)


def test_nettoyer_texte() -> None:
    """Lowercase text and replace newlines without removing other characters."""
    assert nettoyer_texte("Hello\nWORLD!\tL2") == "hello world!\tl2"


def test_analyser_corpus(corpus: Corpus) -> None:
    """Total occurrences and document frequencies have different values."""
    first = Document("First", "Alice", datetime(2026, 1, 1), "", "Cat cat dog")
    corpus.add(first)
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "dog fish"))

    vocab, freq, document_counts = analyser_corpus(corpus)

    assert vocab == {
        "cat": {"tf": 2, "df": 1},
        "dog": {"tf": 2, "df": 2},
        "fish": {"tf": 1, "df": 1},
    }
    assert list(freq.columns) == ["tf", "df"]
    assert freq["tf"].to_dict() == {"cat": 2, "dog": 2, "fish": 1}
    assert freq["df"].to_dict() == {"cat": 1, "dog": 2, "fish": 1}
    assert document_counts == {0: {"cat": 2, "dog": 1}, 1: {"dog": 1, "fish": 1}}
    assert first.texte == "Cat cat dog"


def test_analyser_corpus_delimiters(corpus: Corpus) -> None:
    """Whitespace and punctuation split terms while letters and digits remain."""
    text = " \tCat,\tDOG\nneural-network don't gpt-4 l2 word_name ÉTÉ... "
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", text))

    vocab, freq, document_counts = analyser_corpus(corpus)

    expected = {
        "cat": 1, "dog": 1, "neural": 1, "network": 1, "don": 1, "t": 1,
        "gpt": 1, "4": 1, "l2": 1, "word": 1, "name": 1, "été": 1,
    }
    assert document_counts == {0: expected}
    assert set(vocab) == set(expected)
    assert freq["tf"].to_dict() == expected
    assert freq["df"].to_dict() == expected


def test_empty_statistics(corpus: Corpus) -> None:
    """Empty corpora and documents without terms retain the table columns."""
    vocab, freq, document_counts = analyser_corpus(corpus)

    assert vocab == {}
    assert freq.empty
    assert list(freq.columns) == ["tf", "df"]
    assert document_counts == {}

    corpus.add(Document("Empty", "Alice", datetime(2026, 1, 1), "", " \n\t...___"))
    vocab, freq, document_counts = analyser_corpus(corpus)

    assert vocab == {}
    assert freq.empty
    assert document_counts == {0: {}}


def test_get_word_frequencies(capsys: pytest.CaptureFixture[str]) -> None:
    """Selection sorts ties alphabetically and returns independent data."""
    freq = pd.DataFrame(
        {"tf": [2, 1, 2], "df": [2, 1, 1]}, index=["dog", "fish", "cat"],
    )

    selected = get_word_frequencies(freq, 2)

    assert list(selected.index) == ["cat", "dog"]
    assert selected.values.tolist() == [[2, 1], [2, 2]]
    assert list(freq.index) == ["dog", "fish", "cat"]
    assert capsys.readouterr().out == ""
    selected.loc["cat", "tf"] = 99
    assert freq.loc["cat", "tf"] == 2


def test_frequency_limits() -> None:
    """Zero, large, and negative limits have explicit selection behavior."""
    freq = pd.DataFrame({"tf": [3], "df": [1]}, index=["cat"])

    assert get_word_frequencies(freq, 0).empty
    assert list(get_word_frequencies(freq, 0).columns) == ["tf", "df"]
    assert list(get_word_frequencies(freq, 10).index) == ["cat"]
    with pytest.raises(ValueError):
        get_word_frequencies(freq, -1)
    with pytest.raises(ValueError):
        stats(freq, -1)


def test_stats_output(capsys: pytest.CaptureFixture[str]) -> None:
    """The display includes the full vocabulary size and the selected counts."""
    freq = pd.DataFrame(
        {"tf": [2, 1, 2], "df": [2, 1, 1]}, index=["dog", "fish", "cat"],
    )

    stats(freq, 2)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Distinct words: 3"
    assert lines[1].split() == ["cat", "2"]
    assert lines[2].split() == ["dog", "2"]
    assert len(lines) == 3


def test_stats_zero_limit(capsys: pytest.CaptureFixture[str]) -> None:
    """A zero limit and an empty table still display their vocabulary sizes."""
    freq = pd.DataFrame({"tf": [3], "df": [1]}, index=["cat"])

    stats(freq, 0)
    assert capsys.readouterr().out == "Distinct words: 1\n"

    stats(freq.iloc[:0], 3)
    assert capsys.readouterr().out == "Distinct words: 0\n"


def test_document_ids_in_statistics(corpus: Corpus) -> None:
    """Per-document counts preserve IDs left after filtering."""
    corpus.add(Document("Short", "Alice", datetime(2026, 1, 1), "", "Short"))
    corpus.add(Document("Long", "Bob", datetime(2026, 1, 1), "", "word " * 20))
    corpus.remove_short_documents()

    _, freq, document_counts = analyser_corpus(corpus)

    assert document_counts == {1: {"word": 20}}
    assert freq.loc["word"].tolist() == [20, 1]


def test_construire_index(corpus: Corpus) -> None:
    """Alphabetical columns hold raw counts and matching matrix statistics."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "dog cat cat"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "fish dog"))
    _, direct_freq, document_counts = analyser_corpus(corpus)

    vocab, freq, mat_TF, doc_ids = construire_index(document_counts)

    assert list(vocab) == ["cat", "dog", "fish"]
    assert vocab == {
        "cat": {"id": 0, "tf": 2, "df": 1},
        "dog": {"id": 1, "tf": 2, "df": 2},
        "fish": {"id": 2, "tf": 1, "df": 1},
    }
    assert isinstance(mat_TF, csr_matrix)
    assert mat_TF.shape == (2, 3)
    assert mat_TF.nnz == 4
    assert mat_TF.toarray().tolist() == [[2, 1, 0], [0, 1, 1]]
    assert mat_TF.sum(axis=0).A1.tolist() == [2, 2, 1]
    assert (mat_TF > 0).sum(axis=0).A1.tolist() == [1, 2, 1]
    assert doc_ids == [0, 1]
    pd.testing.assert_frame_equal(freq, direct_freq.sort_index())
    assert document_counts == {0: {"dog": 1, "cat": 2}, 1: {"fish": 1, "dog": 1}}


def test_index_document_ids(corpus: Corpus) -> None:
    """Rows map to original IDs when filtering leaves a gap."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "cat " * 30))
    corpus.add(Document("Short", "Bob", datetime(2026, 1, 1), "", "short"))
    corpus.add(Document("Last", "Alice", datetime(2026, 1, 1), "", "dog " * 30))
    corpus.remove_short_documents()
    _, _, document_counts = analyser_corpus(corpus)
    reversed_counts = {doc_id: document_counts[doc_id] for doc_id in [2, 0]}

    vocab, _, mat_TF, doc_ids = construire_index(reversed_counts)

    assert list(vocab) == ["cat", "dog"]
    assert doc_ids == [0, 2]
    assert mat_TF.shape == (2, 2)
    assert mat_TF.toarray().tolist() == [[30, 0], [0, 30]]
    assert corpus.documents[doc_ids[1]].titre == "Last"
    assert list(corpus.documents) == [0, 2]


def test_index_empty_corpus(corpus: Corpus) -> None:
    """An empty corpus produces an empty index and a zero-by-zero matrix."""
    _, _, document_counts = analyser_corpus(corpus)

    vocab, freq, mat_TF, doc_ids = construire_index(document_counts)

    assert vocab == {}
    assert freq.empty
    assert list(freq.columns) == ["tf", "df"]
    assert isinstance(mat_TF, csr_matrix)
    assert mat_TF.shape == (0, 0)
    assert mat_TF.nnz == 0
    assert doc_ids == []


def test_index_empty_vocabulary(corpus: Corpus) -> None:
    """A document without words retains a row even with zero columns."""
    corpus.add(Document("Empty", "Alice", datetime(2026, 1, 1), "", "... \n___"))
    _, _, document_counts = analyser_corpus(corpus)

    vocab, freq, mat_TF, doc_ids = construire_index(document_counts)

    assert vocab == {}
    assert freq.empty
    assert list(freq.columns) == ["tf", "df"]
    assert mat_TF.shape == (1, 0)
    assert mat_TF.nnz == 0
    assert doc_ids == [0]


def test_index_empty_document_row(corpus: Corpus) -> None:
    """A document without words contributes zeros and no document frequency."""
    corpus.add(Document("Empty", "Alice", datetime(2026, 1, 1), "", "..."))
    corpus.add(Document("Words", "Bob", datetime(2026, 1, 1), "", "cat cat"))
    _, _, document_counts = analyser_corpus(corpus)

    vocab, freq, mat_TF, doc_ids = construire_index(document_counts)

    assert vocab == {"cat": {"id": 0, "tf": 2, "df": 1}}
    assert freq.loc["cat"].tolist() == [2, 1]
    assert mat_TF.shape == (2, 1)
    assert mat_TF.toarray().tolist() == [[0], [2]]
    assert doc_ids == [0, 1]

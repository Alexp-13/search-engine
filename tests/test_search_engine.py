"""Test text cleaning, vocabulary counts, and frequency displays."""

from datetime import datetime

import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from search_engine import Corpus, Document, SearchEngine


def test_nettoyer_texte() -> None:
    """Lowercase text and replace newlines without removing other characters."""
    assert SearchEngine.nettoyer_texte("Hello\nWORLD!\tL2") == "hello world!\tl2"


def test_analyser_corpus(corpus: Corpus) -> None:
    """Total occurrences and document frequencies have different values."""
    first = Document("First", "Alice", datetime(2026, 1, 1), "", "Cat cat dog")
    corpus.add(first)
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "dog fish"))

    engine = SearchEngine(corpus)
    vocab = engine.freq.to_dict(orient="index")
    freq = engine.freq
    document_counts = engine.document_counts

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

    engine = SearchEngine(corpus)
    vocab = engine.freq.to_dict(orient="index")
    freq = engine.freq
    document_counts = engine.document_counts

    expected = {
        "cat": 1, "dog": 1, "neural": 1, "network": 1, "don": 1, "t": 1,
        "gpt": 1, "4": 1, "l2": 1, "word": 1, "name": 1, "été": 1,
    }
    assert document_counts == {0: expected}
    assert set(vocab) == set(expected)
    assert freq["tf"].to_dict() == expected
    assert freq["df"].to_dict() == expected


def test_index_progress(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """Show document progress without changing the matrices or search results."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "dog fish"))

    quiet = SearchEngine(corpus)
    output = capsys.readouterr()
    assert output.out == output.err == ""

    visible = SearchEngine(corpus, show_progress=True)
    output = capsys.readouterr()
    assert output.out == ""
    assert "Indexing" in output.err
    assert "2/2" in output.err
    assert (quiet.mat_TF != visible.mat_TF).nnz == 0
    assert (quiet.mat_TFxIDF != visible.mat_TFxIDF).nnz == 0
    pd.testing.assert_frame_equal(quiet.search("cat fish", 2), visible.search("cat fish", 2))


def test_empty_statistics(corpus: Corpus) -> None:
    """Empty corpora and documents without terms retain the table columns."""
    engine = SearchEngine(corpus)
    vocab = engine.freq.to_dict(orient="index")
    freq = engine.freq
    document_counts = engine.document_counts

    assert vocab == {}
    assert freq.empty
    assert list(freq.columns) == ["tf", "df"]
    assert document_counts == {}

    corpus.add(Document("Empty", "Alice", datetime(2026, 1, 1), "", " \n\t...___"))
    engine = SearchEngine(corpus)
    vocab = engine.freq.to_dict(orient="index")
    freq = engine.freq
    document_counts = engine.document_counts

    assert vocab == {}
    assert freq.empty
    assert document_counts == {0: {}}


def test_get_word_frequencies(capsys: pytest.CaptureFixture[str]) -> None:
    """Selection sorts ties alphabetically and returns independent data."""
    freq = pd.DataFrame(
        {"tf": [2, 1, 2], "df": [2, 1, 1]}, index=["dog", "fish", "cat"],
    )

    selected = SearchEngine.get_word_frequencies(freq, 2)

    assert list(selected.index) == ["cat", "dog"]
    assert selected.values.tolist() == [[2, 1], [2, 2]]
    assert list(freq.index) == ["dog", "fish", "cat"]
    assert capsys.readouterr().out == ""
    selected.loc["cat", "tf"] = 99
    assert freq.loc["cat", "tf"] == 2


def test_frequency_limits(corpus: Corpus) -> None:
    """Zero, large, and negative limits have explicit selection behavior."""
    freq = pd.DataFrame({"tf": [3], "df": [1]}, index=["cat"])
    engine = SearchEngine(corpus)
    engine.freq = freq

    assert SearchEngine.get_word_frequencies(freq, 0).empty
    assert list(SearchEngine.get_word_frequencies(freq, 0).columns) == ["tf", "df"]
    assert list(SearchEngine.get_word_frequencies(freq, 10).index) == ["cat"]
    with pytest.raises(ValueError):
        SearchEngine.get_word_frequencies(freq, -1)
    with pytest.raises(ValueError):
        engine.stats(-1)


def test_stats_output(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """The display includes the full vocabulary size and the selected counts."""
    freq = pd.DataFrame(
        {"tf": [2, 1, 2], "df": [2, 1, 1]}, index=["dog", "fish", "cat"],
    )
    engine = SearchEngine(corpus)
    engine.freq = freq

    engine.stats(2)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Distinct words: 3"
    assert lines[1].split() == ["cat", "2"]
    assert lines[2].split() == ["dog", "2"]
    assert len(lines) == 3


def test_stats_zero_limit(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """A zero limit and an empty table still display their vocabulary sizes."""
    freq = pd.DataFrame({"tf": [3], "df": [1]}, index=["cat"])
    engine = SearchEngine(corpus)
    engine.freq = freq

    engine.stats(0)
    assert capsys.readouterr().out == "Distinct words: 1\n"

    engine.freq = freq.iloc[:0]
    engine.stats(3)
    assert capsys.readouterr().out == "Distinct words: 0\n"


def test_document_ids_in_statistics(corpus: Corpus) -> None:
    """Per-document counts preserve IDs left after filtering."""
    corpus.add(Document("Short", "Alice", datetime(2026, 1, 1), "", "Short"))
    corpus.add(Document("Long", "Bob", datetime(2026, 1, 1), "", "word " * 20))
    corpus.remove_short_documents()

    engine = SearchEngine(corpus)
    freq = engine.freq
    document_counts = engine.document_counts

    assert document_counts == {1: {"word": 20}}
    assert freq.loc["word"].tolist() == [20, 1]


def test_construire_index(corpus: Corpus) -> None:
    """Alphabetical columns hold raw counts and matching matrix statistics."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "dog cat cat"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "fish dog"))
    engine = SearchEngine(corpus)
    direct_freq = engine.freq
    document_counts = engine.document_counts

    vocab, freq, mat_TF, doc_ids = SearchEngine.construire_index(document_counts)

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
    engine = SearchEngine(corpus)
    document_counts = engine.document_counts
    reversed_counts = {doc_id: document_counts[doc_id] for doc_id in [2, 0]}

    vocab, _, mat_TF, doc_ids = SearchEngine.construire_index(reversed_counts)

    assert list(vocab) == ["cat", "dog"]
    assert doc_ids == [0, 2]
    assert mat_TF.shape == (2, 2)
    assert mat_TF.toarray().tolist() == [[30, 0], [0, 30]]
    assert corpus.documents[doc_ids[1]].titre == "Last"
    assert list(corpus.documents) == [0, 2]


def test_index_empty_corpus(corpus: Corpus) -> None:
    """An empty corpus produces an empty index and a zero-by-zero matrix."""
    engine = SearchEngine(corpus)
    document_counts = engine.document_counts

    vocab, freq, mat_TF, doc_ids = SearchEngine.construire_index(document_counts)

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
    engine = SearchEngine(corpus)
    document_counts = engine.document_counts

    vocab, freq, mat_TF, doc_ids = SearchEngine.construire_index(document_counts)

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
    engine = SearchEngine(corpus)
    document_counts = engine.document_counts

    vocab, freq, mat_TF, doc_ids = SearchEngine.construire_index(document_counts)

    assert vocab == {"cat": {"id": 0, "tf": 2, "df": 1}}
    assert freq.loc["cat"].tolist() == [2, 1]
    assert mat_TF.shape == (2, 1)
    assert mat_TF.toarray().tolist() == [[0], [2]]
    assert doc_ids == [0, 1]


def test_calculer_tfidf() -> None:
    """Known weights produce a second matrix without changing TF statistics."""
    vocab, freq, mat_TF, doc_ids = SearchEngine.construire_index(
        {0: {"cat": 2, "dog": 1}, 1: {"dog": 1, "fish": 1}},
    )
    original_freq = freq.copy()

    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)

    assert idf == pytest.approx([0.6931471805599453, 0, 0.6931471805599453])
    assert isinstance(mat_TFxIDF, csr_matrix)
    assert mat_TFxIDF.shape == mat_TF.shape == (2, 3)
    assert mat_TFxIDF.toarray()[0].tolist() == pytest.approx([1.3862943611198906, 0, 0])
    assert mat_TFxIDF.toarray()[1].tolist() == pytest.approx([0, 0, 0.6931471805599453])
    assert mat_TFxIDF.nnz == 2
    assert mat_TF.toarray().tolist() == [[2, 1, 0], [0, 1, 1]]
    assert vocab == {
        "cat": {"id": 0, "tf": 2, "df": 1},
        "dog": {"id": 1, "tf": 2, "df": 2},
        "fish": {"id": 2, "tf": 1, "df": 1},
    }
    assert doc_ids == [0, 1]
    pd.testing.assert_frame_equal(freq, original_freq)


def test_tfidf_column_ids() -> None:
    """Weights follow column IDs even when vocabulary entries are reordered."""
    vocab, _, mat_TF, doc_ids = SearchEngine.construire_index({
        7: {"common": 1},
        0: {"alpha": 2, "common": 1, "rare": 1},
        2: {"common": 1, "rare": 2},
    })
    reversed_vocab = dict(reversed(list(vocab.items())))

    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, reversed_vocab)

    assert idf == pytest.approx([1.0986122886681098, 0, 0.4054651081081644])
    assert mat_TFxIDF.toarray()[0].tolist() == pytest.approx([2.1972245773362196, 0, 0.4054651081081644])
    assert mat_TFxIDF.toarray()[1].tolist() == pytest.approx([0, 0, 0.8109302162163288])
    assert mat_TFxIDF.toarray()[2].tolist() == [0, 0, 0]
    assert doc_ids == [0, 2, 7]


def test_tfidf_single_document() -> None:
    """Every word in a one-document corpus has zero IDF."""
    vocab, _, mat_TF, _ = SearchEngine.construire_index({0: {"cat": 2, "dog": 1}})

    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)

    assert idf == [0, 0]
    assert mat_TFxIDF.shape == (1, 2)
    assert mat_TFxIDF.nnz == 0
    assert mat_TF.toarray().tolist() == [[2, 1]]


def test_tfidf_empty_corpus() -> None:
    """An empty corpus produces empty weights without evaluating a logarithm."""
    vocab, _, mat_TF, _ = SearchEngine.construire_index({})

    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)

    assert idf == []
    assert isinstance(mat_TFxIDF, csr_matrix)
    assert mat_TFxIDF.shape == (0, 0)
    assert mat_TFxIDF.nnz == 0


def test_tfidf_empty_vocabulary() -> None:
    """Documents without terms retain their rows in the weighted matrix."""
    vocab, _, mat_TF, _ = SearchEngine.construire_index({0: {}, 3: {}})

    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)

    assert idf == []
    assert mat_TFxIDF.shape == (2, 0)
    assert mat_TFxIDF.nnz == 0


def test_tfidf_empty_document_row() -> None:
    """An empty document contributes to N and keeps its zero row."""
    vocab, _, mat_TF, _ = SearchEngine.construire_index({0: {}, 1: {"cat": 2}})

    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)

    assert idf == pytest.approx([0.6931471805599453])
    assert mat_TFxIDF.shape == (2, 1)
    assert mat_TFxIDF.toarray()[0].tolist() == [0]
    assert mat_TFxIDF.toarray()[1].tolist() == pytest.approx([1.3862943611198906])


def test_vectoriser_requete(corpus: Corpus) -> None:
    """Queries share document cleaning, count repetitions, and ignore unknown terms."""
    text = "Cat,\nCAT dog"
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", text))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "dog fish"))
    engine = SearchEngine(corpus)
    counts = engine.document_counts
    vocab, _, mat_TF, _ = SearchEngine.construire_index(counts)
    original_vocab = {word: metadata.copy() for word, metadata in vocab.items()}
    idf, _ = SearchEngine.calculer_tfidf(mat_TF, vocab)
    original_idf = idf.copy()
    reversed_vocab = dict(reversed(list(vocab.items())))

    query = SearchEngine.vectoriser_requete("CAT,\ncat\tDOG fish unknown", reversed_vocab, idf)

    assert isinstance(query, csr_matrix)
    assert query.shape == (1, 3)
    assert query.nnz == 2
    assert query.toarray()[0].tolist() == pytest.approx([1.3862943611198906, 0, 0.6931471805599453])
    assert vocab == original_vocab
    assert idf == original_idf
    assert corpus.documents[0].texte == text


def test_cosine_similarities() -> None:
    """Different document lengths are normalized and a zero row has score zero."""
    matrix = csr_matrix([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]])
    query = csr_matrix([[3.0, 4.0]])
    norms = SearchEngine.calculer_normes(matrix)

    scores = SearchEngine.calculer_similarites(query, matrix, norms)

    assert norms == [5, 2, 0]
    assert scores == pytest.approx([1, 0.8, 0])
    assert matrix.toarray().tolist() == [[3, 4], [0, 2], [0, 0]]
    assert query.toarray().tolist() == [[3, 4]]


def test_ranked_search_with_document_ids(corpus: Corpus) -> None:
    """The complete pipeline ranks real document IDs after filtering."""
    corpus.add(Document("Fish", "Alice", datetime(2026, 1, 1), "", "dog fish " * 15))
    corpus.add(Document("Short", "Bob", datetime(2026, 1, 1), "", "short"))
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "", "cat cat dog " * 10))
    corpus.remove_short_documents()
    engine = SearchEngine(corpus)
    counts = engine.document_counts
    vocab, _, mat_TF, doc_ids = SearchEngine.construire_index(counts)
    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)
    norms = SearchEngine.calculer_normes(mat_TFxIDF)

    query = SearchEngine.vectoriser_requete("cat cat fish", vocab, idf)
    scores = SearchEngine.calculer_similarites(query, mat_TFxIDF, norms)
    results = SearchEngine.classer_documents(scores, doc_ids, 2)

    assert doc_ids == [0, 2]
    assert scores == pytest.approx([0.4472135954999579, 0.8944271909999159])
    assert [doc_id for doc_id, _ in results] == [2, 0]
    assert corpus.documents[results[0][0]].titre == "Cat"


@pytest.mark.parametrize("text", ["", " \n\t...", "unknown", "DOG"])
def test_zero_query_scores(text: str) -> None:
    """Empty, unknown, and zero-IDF queries produce no ranked documents."""
    vocab, _, mat_TF, doc_ids = SearchEngine.construire_index(
        {0: {"cat": 2, "dog": 1}, 1: {"dog": 1, "fish": 1}},
    )
    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)
    norms = SearchEngine.calculer_normes(mat_TFxIDF)

    query = SearchEngine.vectoriser_requete(text, vocab, idf)
    scores = SearchEngine.calculer_similarites(query, mat_TFxIDF, norms)

    assert query.shape == (1, 3)
    assert query.nnz == 0
    assert scores == [0, 0]
    assert SearchEngine.classer_documents(scores, doc_ids, 10) == []


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (1, [(2, 0.8)]),
        (2, [(2, 0.8), (7, 0.8)]),
        (10, [(2, 0.8), (7, 0.8), (9, 0.5)]),
        (0, []),
    ],
)
def test_ranking_limits_and_ties(limit: int, expected: list[tuple[int, float]]) -> None:
    """Positive scores are limited and exact ties use ascending original IDs."""
    scores = [0.5, 0.8, 0.8, 0.0]
    doc_ids = [9, 7, 2, 4]

    assert SearchEngine.classer_documents(scores, doc_ids, limit) == expected
    assert scores == [0.5, 0.8, 0.8, 0.0]
    assert doc_ids == [9, 7, 2, 4]


def test_ranking_invalid_limit_and_empty_results() -> None:
    """Negative limits are rejected and missing positive scores give no results."""
    with pytest.raises(ValueError):
        SearchEngine.classer_documents([0.5], [0], -1)
    assert SearchEngine.classer_documents([], [], 5) == []
    assert SearchEngine.classer_documents([0, 0], [2, 4], 5) == []


def test_ranking_without_rounding() -> None:
    """Close but distinct scores keep their numerical order before display."""
    assert SearchEngine.classer_documents([0.5, 0.500000000001], [1, 9], 2) == [
        (9, 0.500000000001), (1, 0.5),
    ]


@pytest.mark.parametrize("counts", [{}, {0: {}, 2: {}}, {0: {"cat": 2}}])
def test_search_without_weighted_terms(counts: dict[int, dict[str, int]]) -> None:
    """Empty indexes and corpora with only zero-IDF terms remain searchable."""
    vocab, _, mat_TF, doc_ids = SearchEngine.construire_index(counts)
    idf, mat_TFxIDF = SearchEngine.calculer_tfidf(mat_TF, vocab)
    norms = SearchEngine.calculer_normes(mat_TFxIDF)

    query = SearchEngine.vectoriser_requete("cat", vocab, idf)
    scores = SearchEngine.calculer_similarites(query, mat_TFxIDF, norms)

    assert query.shape == (1, len(vocab))
    assert query.nnz == 0
    assert norms == [0] * len(doc_ids)
    assert scores == [0] * len(doc_ids)
    assert SearchEngine.classer_documents(scores, doc_ids, 5) == []


def test_constructor_builds_index(corpus: Corpus) -> None:
    """Instantiation immediately prepares all index data without a search call."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "", "dog fish"))

    engine = SearchEngine(corpus)

    assert engine.corpus is corpus
    assert engine.document_counts == {0: {"cat": 2, "dog": 1}, 1: {"dog": 1, "fish": 1}}
    assert engine.doc_ids == [0, 1]
    assert list(engine.vocab) == ["cat", "dog", "fish"]
    assert [metadata["id"] for metadata in engine.vocab.values()] == [0, 1, 2]
    assert engine.mat_TF.toarray().tolist() == [[2, 1, 0], [0, 1, 1]]
    assert engine.freq["tf"].tolist() == [2, 2, 1]
    assert engine.freq["df"].tolist() == [1, 2, 1]
    assert engine.idf == pytest.approx([0.6931471805599453, 0, 0.6931471805599453])
    assert engine.mat_TFxIDF.shape == (2, 3)
    assert engine.mat_TFxIDF.nnz == 2
    assert engine.normes_documents == pytest.approx([1.3862943611198906, 0.6931471805599453])


def test_search_dataframe(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """Ranked results contain original IDs, metadata, and unrounded scores."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "https://example.test/cat", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "https://example.test/fish", "dog fish"))
    engine = SearchEngine(corpus)

    results = engine.search("cat cat fish", 2)

    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["id", "titre", "url", "score"]
    assert results["id"].tolist() == [0, 1]
    assert results["titre"].tolist() == ["Cat", "Fish"]
    assert results["url"].tolist() == ["https://example.test/cat", "https://example.test/fish"]
    assert results["score"].tolist() == pytest.approx([0.8944271909999159, 0.4472135954999579])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("query", "limit", "expected_ids"),
    [
        ("cat cat fish", 1, [0]),
        ("cat cat fish", 10, [0, 1]),
        ("fish", 5, [1]),
        ("", 5, []),
        ("unknown", 5, []),
        ("DOG", 5, []),
        ("cat", 0, []),
    ],
)
def test_search_limits_and_empty_tables(
    corpus: Corpus, query: str, limit: int, expected_ids: list[int],
) -> None:
    """All query outcomes retain the same DataFrame columns and result limit."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "", "dog fish"))

    results = SearchEngine(corpus).search(query, limit)

    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ["id", "titre", "url", "score"]
    assert results["id"].tolist() == expected_ids
    assert all(score > 0 for score in results["score"])


def test_search_negative_limit(corpus: Corpus) -> None:
    """A negative limit is rejected even with an empty corpus and query."""
    with pytest.raises(ValueError):
        SearchEngine(corpus).search("", -1)


def test_search_ties_and_filtered_ids(corpus: Corpus) -> None:
    """Equal scores use IDs while filtered documents keep their original IDs."""
    corpus.add(Document("Zulu", "Alice", datetime(2026, 1, 1), "", "cat " * 30))
    corpus.add(Document("Short", "Bob", datetime(2026, 1, 1), "", "short"))
    corpus.add(Document("Alpha", "Bob", datetime(2026, 1, 1), "", "cat " * 30))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "", "fish " * 30))
    corpus.remove_short_documents()
    engine = SearchEngine(corpus)

    results = engine.search("cat", 10)

    assert engine.doc_ids == [0, 2, 3]
    assert results["id"].tolist() == [0, 2]
    assert results["titre"].tolist() == ["Zulu", "Alpha"]
    assert results["score"].tolist() == pytest.approx([1, 1])
    assert engine.search("fish", 1)["id"].tolist() == [3]


def test_search_reuses_index(corpus: Corpus) -> None:
    """Repeated queries reuse matrices, vocabulary, and precomputed norms."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "", "dog fish"))
    engine = SearchEngine(corpus)
    matrix = engine.mat_TFxIDF
    norms = engine.normes_documents
    vocab = engine.vocab
    original_freq = engine.freq.copy()

    assert engine.search("cat", 5)["id"].tolist() == [0]
    assert engine.search("fish", 5)["id"].tolist() == [1]

    assert engine.mat_TFxIDF is matrix
    assert engine.normes_documents is norms
    assert engine.vocab is vocab
    pd.testing.assert_frame_equal(engine.freq, original_freq)


def test_rebuild_after_corpus_change(corpus: Corpus) -> None:
    """Constructing another engine indexes newly added corpus documents."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "", "dog fish"))
    old_engine = SearchEngine(corpus)
    corpus.add(Document("Bird", "Alice", datetime(2026, 1, 1), "", "bird"))

    new_engine = SearchEngine(corpus)

    assert old_engine.search("bird", 5).empty
    assert old_engine.doc_ids == [0, 1]
    assert new_engine.doc_ids == [0, 1, 2]
    assert new_engine.search("bird", 5)["id"].tolist() == [2]


def test_corpus_passages_remain_available(corpus: Corpus) -> None:
    """Passages and concordances remain directly accessible on Corpus."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "", "dog fish"))
    engine = SearchEngine(corpus)

    assert engine.search("cat", 5)["id"].tolist() == [0]
    assert len(corpus.search("cat")) == 2
    assert len(corpus.concorde("cat", 2)) == 2
    assert not hasattr(engine, "concorde")
    assert not hasattr(engine, "search_passages")


def test_search_engine_public_imports() -> None:
    """Both package entry points export the same SearchEngine class."""
    from search_engine.search import SearchEngine as PackageEngine
    from search_engine.search.SearchEngine import SearchEngine as ModuleEngine

    assert SearchEngine is PackageEngine is ModuleEngine

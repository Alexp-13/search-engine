"""Test corpus registration, sorting, statistics, and JSONL persistence."""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from search_engine.corpus import Corpus
from search_engine.documents.Document import Document


def test_add_documents(corpus: Corpus) -> None:
    """Documents receive different IDs and share the same author entry."""
    first = Document("First", "Alice", datetime(2026, 1, 1), "", "Hello")
    second = Document("Second", "Alice", datetime(2026, 1, 2), "", "World")

    assert corpus.add(first) == 0
    assert corpus.add(second) == 1
    assert len(corpus) == 2
    assert corpus.documents[0] is first
    assert corpus.authors["Alice"].nb_docs == 2


def test_singleton(corpus: Corpus) -> None:
    """Constructing a second corpus returns the existing instance."""
    another = Corpus("Another name")

    assert another is corpus
    assert another.nom == "Neural networks"


def test_document_stats(corpus: Corpus) -> None:
    """A short example contains three words and two sentences."""
    document = Document(
        "Example", "Alice", datetime(2026, 1, 1), "", "Hello world. Goodbye."
    )
    corpus.add(document)

    assert corpus.document_stats() == [
        {"doc_id": 0, "word_count": 3, "sentence_count": 2}
    ]


def test_sort_by_title(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """Alphabetical display puts Apple before Banana."""
    corpus.add(Document("Banana", "Alice", datetime(2026, 1, 1), "", ""))
    corpus.add(Document("Apple", "Bob", datetime(2026, 1, 1), "", ""))

    corpus.show_by_title(2)

    assert capsys.readouterr().out == "1: Apple\n0: Banana\n"


def test_sort_by_date(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """Date display puts the newer document first."""
    corpus.add(Document("Old", "Alice", datetime(2025, 1, 1), "", ""))
    corpus.add(Document("New", "Bob", datetime(2026, 1, 1), "", ""))

    corpus.show_by_date(2)

    assert capsys.readouterr().out == "1: New\n0: Old\n"


def test_save_and_load(corpus: Corpus, tmp_path: Path) -> None:
    """Saving and loading restores documents, their dates, and their authors."""
    document = Document("Climate", "Alice", datetime(2026, 1, 1), "", "Hello world.")
    corpus.add(document)
    path = tmp_path / "corpus.jsonl"

    corpus.save(path)
    corpus.nom = "Changed name"
    corpus.add(Document("Extra", "Bob", datetime(2026, 1, 2), "", "Extra text."))
    corpus.load(path)

    assert len(corpus) == 1
    assert corpus.nom == "Neural networks"
    assert corpus.id_document == 1
    assert corpus.documents[0].titre == "Climate"
    assert corpus.documents[0].texte == "Hello world."
    assert corpus.documents[0].date == document.date
    assert set(corpus.authors) == {"Alice"}
    assert corpus.authors["Alice"].nb_docs == 1


def test_mixed_documents_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """Both document types and their specific fields survive save/load."""
    from search_engine.documents import ArxivDocument, RedditDocument

    corpus.add(RedditDocument("Post", "Alice", datetime(2026, 1, 1), "", "Text", 4))
    corpus.add(ArxivDocument("Paper", "Bob", datetime(2026, 1, 1), "", "Abstract", ["Carol"]))
    path = tmp_path / "mixed.jsonl"

    corpus.save(path)
    corpus.load(path)

    assert isinstance(corpus.documents[0], RedditDocument)
    assert corpus.documents[0].get_nb_comments() == 4
    assert isinstance(corpus.documents[1], ArxivDocument)
    assert corpus.documents[1].get_coauthors() == ["Carol"]
    assert corpus.authors["Carol"].production[1] is corpus.documents[1]


def test_remove_short_documents(corpus: Corpus) -> None:
    """A 99-character text is removed and a 100-character text is retained."""
    corpus.add(Document("Short", "Alice", datetime(2026, 1, 1), "", "a" * 99))
    corpus.add(Document("Long", "Bob", datetime(2026, 1, 1), "", "b" * 100))

    removed = corpus.remove_short_documents()

    assert removed == 1
    assert list(corpus.documents) == [1]
    assert set(corpus.authors) == {"Bob"}


def test_save_and_load_keeps_next_id_after_filtering(corpus: Corpus, tmp_path: Path) -> None:
    """Reloading preserves the counter even after the highest ID was removed."""
    document = Document("Long", "Alice", datetime(2026, 1, 1), "", "a" * 100)
    corpus.add(document)
    corpus.add(Document("Short", "Bob", datetime(2026, 1, 1), "", "Short"))
    corpus.remove_short_documents()
    path = tmp_path / "filtered.jsonl"

    corpus.save(path)
    corpus.load(path)

    assert list(corpus.documents) == [0]
    assert corpus.id_document == 2
    assert corpus.add(document) == 2


def test_save_and_load_empty_corpus(corpus: Corpus, tmp_path: Path) -> None:
    """An empty corpus still preserves its name and allocation history."""
    corpus.add(Document("Short", "Alice", datetime(2026, 1, 1), "", "Short"))
    corpus.remove_short_documents()
    path = tmp_path / "empty.jsonl"
    corpus.save(path)
    corpus.nom = "Changed name"

    corpus.load(path)

    assert corpus.nom == "Neural networks"
    assert corpus.id_document == 1
    assert corpus.documents == {}
    assert corpus.authors == {}


def test_load_document_only_file(corpus: Corpus, tmp_path: Path) -> None:
    """Older files without metadata remain readable."""
    document = Document("Example", "Alice", datetime(2026, 1, 1), "", "Text")
    path = tmp_path / "older.jsonl"
    path.write_text(json.dumps({"id": 4, **document.to_dict()}) + "\n", encoding="utf-8")

    corpus.load(path)

    assert corpus.nom == "Neural networks"
    assert corpus.documents[4].titre == "Example"
    assert corpus.id_document == 5


def test_load_rejects_counter_collision(corpus: Corpus, tmp_path: Path) -> None:
    """An invalid saved counter is rejected without replacing the active state."""
    document = Document("Example", "Alice", datetime(2026, 1, 1), "", "Text")
    corpus.add(document)
    cached_text = corpus.get_text()
    path = tmp_path / "invalid.jsonl"
    metadata = {"record_type": "corpus", "nom": "Wrong name", "id_document": 0}
    path.write_text(
        json.dumps(metadata) + "\n" + json.dumps({"id": 0, **document.to_dict()}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        corpus.load(path)

    assert corpus.nom == "Neural networks"
    assert corpus.id_document == 1
    assert corpus.documents[0] is document
    assert corpus.get_text() is cached_text
    assert corpus.search("text") == ["Text"]


def test_get_text_is_cached(corpus: Corpus) -> None:
    """Searches reuse the same joined string until the corpus changes."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "Hello"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "World"))

    text = corpus.get_text()

    assert text == "Hello\nWorld"
    assert corpus.search("world") == ["World"]
    assert corpus.get_text() is text


def test_search_literal_keyword_and_case(corpus: Corpus) -> None:
    """Each C++ occurrence is found literally without changing its case."""
    text = "Code in C++. More C++."
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", text))

    assert corpus.search("c++") == [text, text]


def test_search_limits_context(corpus: Corpus) -> None:
    """A passage includes at most 30 characters on either side."""
    text = "a" * 40 + "target" + "b" * 40
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", text))

    assert corpus.search("target") == ["a" * 30 + "target" + "b" * 30]


def test_search_stays_within_documents(corpus: Corpus) -> None:
    """Neither passages nor occurrences cross document boundaries."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "target"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "neighbor"))

    assert corpus.search("target") == ["target"]
    assert corpus.search("neighbor") == ["neighbor"]
    assert corpus.search("target\nneighbor") == []


def test_search_empty_and_missing_keyword(corpus: Corpus) -> None:
    """Empty input, empty corpora, and missing words return no passages."""
    assert corpus.search("") == []
    assert corpus.search("missing") == []

    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "Hello"))

    assert corpus.search("missing") == []


def test_add_invalidates_text_cache(corpus: Corpus) -> None:
    """A document added after a search appears in the next search."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "Old text"))
    corpus.search("old")

    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "New text"))

    assert corpus.get_text() == "Old text\nNew text"
    assert corpus.search("new") == ["New text"]


def test_filter_invalidates_text_cache(corpus: Corpus) -> None:
    """Filtering updates both the joined text and shifted document bounds."""
    retained_text = "target" + "b" * 100
    corpus.add(Document("Short", "Alice", datetime(2026, 1, 1), "", "Short"))
    corpus.add(Document("Long", "Bob", datetime(2026, 1, 1), "", retained_text))
    corpus.get_text()

    assert corpus.remove_short_documents() == 1
    assert corpus.get_text() == retained_text
    assert corpus.search("short") == []
    assert corpus.search("target") == ["target" + "b" * 30]


def test_load_invalidates_text_cache(corpus: Corpus, tmp_path: Path) -> None:
    """Loading replaces cached content rather than retaining later additions."""
    corpus.add(Document("Saved", "Alice", datetime(2026, 1, 1), "", "Saved needle"))
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    corpus.add(Document("Extra", "Bob", datetime(2026, 1, 1), "", "Extra target"))
    corpus.search("target")

    corpus.load(path)

    assert corpus.get_text() == "Saved needle"
    assert corpus.search("target") == []
    assert corpus.search("needle") == ["Saved needle"]


def test_load_empty_file_clears_text_cache(corpus: Corpus, tmp_path: Path) -> None:
    """Loading an empty file removes previously searchable passages."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "target"))
    corpus.search("target")
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    corpus.load(path)

    assert corpus.get_text() == ""
    assert corpus.search("target") == []


def test_concorde_multiple_matches(corpus: Corpus) -> None:
    """Regex alternatives produce the expected rows with original case."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "ab CAT cd dog ef"))

    table = corpus.concorde("cat|dog", 3)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["contexte gauche", "motif trouvé", "contexte droit"]
    assert table.values.tolist() == [
        ["ab ", "CAT", " cd"],
        ["cd ", "dog", " ef"],
    ]
    assert corpus.search("cat|dog") == []


def test_concorde_context_size(corpus: Corpus) -> None:
    """The requested context length applies to both sides of a match."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "abc cat def"))

    assert corpus.concorde("cat", 1).values.tolist() == [[" ", "cat", " "]]
    assert corpus.concorde("cat", 4).values.tolist() == [["abc ", "cat", " def"]]


def test_concorde_stays_within_documents(corpus: Corpus) -> None:
    """Long contexts and regex matches cannot cross into another document."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "cat and dog"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "cat"))

    assert corpus.concorde("cat|dog", 50).values.tolist() == [
        ["", "cat", " and dog"],
        ["cat and ", "dog", ""],
        ["", "cat", ""],
    ]
    assert corpus.concorde(r"dog\s+cat", 10).empty


def test_concorde_zero_context(corpus: Corpus) -> None:
    """Zero context keeps the matched text between two empty strings."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "Before CAT after"))

    assert corpus.concorde("cat", 0).values.tolist() == [["", "CAT", ""]]


def test_concorde_empty_results(corpus: Corpus) -> None:
    """Empty corpora, expressions, and absent matches keep the table schema."""
    empty_corpus = corpus.concorde("cat", 3)
    assert empty_corpus.empty
    assert list(empty_corpus.columns) == ["contexte gauche", "motif trouvé", "contexte droit"]

    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "dog"))

    assert corpus.concorde("cat", 3).empty
    empty_expression = corpus.concorde("", 3)
    assert empty_expression.empty
    assert list(empty_expression.columns) == list(empty_corpus.columns)


def test_concorde_zero_width_regex(corpus: Corpus) -> None:
    """A valid zero-width regex is distinct from an empty expression."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "cat"))

    assert corpus.concorde(r"(?=cat)", 3).values.tolist() == [["", "", "cat"]]
    assert corpus.concorde("", 3).empty


def test_concorde_negative_context(corpus: Corpus) -> None:
    """Negative context lengths are rejected."""
    with pytest.raises(ValueError):
        corpus.concorde("cat", -1)


def test_concorde_invalid_regex(corpus: Corpus) -> None:
    """Malformed regex expressions retain their regex error."""
    with pytest.raises(re.error):
        corpus.concorde("[", 3)


def test_concorde_reuses_text_cache(corpus: Corpus) -> None:
    """Concordances share the text cache and reflect later additions."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "cat here"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "dog there"))
    text = corpus.get_text()

    assert len(corpus.concorde("cat|dog", 3)) == 2
    assert corpus.get_text() is text

    corpus.add(Document("Third", "Eve", datetime(2026, 1, 1), "", "new cat"))
    table = corpus.concorde("cat", 4)

    assert len(table) == 2
    assert table.iloc[-1].tolist() == ["new ", "cat", ""]

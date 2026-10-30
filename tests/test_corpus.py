"""Test corpus registration, sorting, statistics, and JSONL persistence."""

import json
from datetime import datetime
from pathlib import Path

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

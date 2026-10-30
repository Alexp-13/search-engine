"""Test author registration and document-length statistics."""

from datetime import datetime

from search_engine.corpus import Author
from search_engine.documents.Document import Document


def test_add_document() -> None:
    """Adding a document registers it in the author's production."""
    author = Author("Alice")
    document = Document("Climate", "Alice", datetime(2026, 1, 1), "", "Hello")

    author.add(0, document)

    assert author.nb_docs == 1
    assert author.production[0] is document


def test_mean_document_length() -> None:
    """Texts of four and eight characters have an average length of six."""
    author = Author("Alice")
    author.add(0, Document("First", "Alice", datetime(2026, 1, 1), "", "abcd"))
    author.add(1, Document("Second", "Alice", datetime(2026, 1, 1), "", "abcdefgh"))

    assert author.mean_document_length() == 6.0

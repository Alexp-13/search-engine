"""Test document inheritance, source-specific fields, and factory selection."""

from datetime import datetime

import pytest

from search_engine.documents import (
    ArxivDocument,
    Document,
    DocumentFactory,
    RedditDocument,
)


def test_reddit_document() -> None:
    """A Reddit document inherits common fields and exposes its comment count."""
    document = RedditDocument("Post", "Alice", datetime(2026, 1, 1), "", "Text", 3)

    document.set_nb_comments(5)

    assert isinstance(document, Document)
    assert document.getType() == "reddit"
    assert document.get_nb_comments() == 5
    assert "Reddit" in str(document)


def test_arxiv_document() -> None:
    """An arXiv document exposes the main author and its distinct coauthors."""
    document = ArxivDocument("Paper", "Alice", datetime(2026, 1, 1), "", "Abstract", ["Bob"])

    document.set_coauthors(["Bob", "Carol"])

    assert isinstance(document, Document)
    assert document.getType() == "arxiv"
    assert document.get_coauthors() == ["Bob", "Carol"]
    assert document.get_authors() == ["Alice", "Bob", "Carol"]


def test_document_factory() -> None:
    """The source selects the document subclass."""
    date = datetime(2026, 1, 1)
    reddit = DocumentFactory.create("reddit", "Post", "Alice", date, "", "Text")
    arxiv = DocumentFactory.create("arxiv", "Paper", "Bob", date, "", "Abstract")

    assert isinstance(reddit, RedditDocument)
    assert isinstance(arxiv, ArxivDocument)


def test_unknown_factory_source() -> None:
    """An unsupported source produces a clear error."""
    with pytest.raises(ValueError):
        DocumentFactory.create("unknown", "Title", "Alice", datetime(2026, 1, 1), "", "")

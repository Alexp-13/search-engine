"""Test API field mapping using small local responses."""

from datetime import datetime, timezone
from types import SimpleNamespace

from search_engine.collectors import collect_arxiv, collect_reddit
from search_engine.documents import ArxivDocument, RedditDocument


def test_collect_reddit() -> None:
    """Post bodies, comment counts, and deleted authors are mapped correctly."""
    post = SimpleNamespace(
        title="A post", author=None, created_utc=0,
        permalink="/r/example/comments/123/post/", selftext="Hello\nworld", num_comments=2,
    )
    subreddit = SimpleNamespace(search=lambda query, sort, limit: [post])
    client = SimpleNamespace(subreddit=lambda name: subreddit)

    documents = collect_reddit("example", "neural networks", 1, client=client)

    assert len(documents) == 1
    assert isinstance(documents[0], RedditDocument)
    assert documents[0].texte == "Hello world"
    assert documents[0].auteur == "[deleted]"
    assert documents[0].get_nb_comments() == 2


def test_collect_arxiv() -> None:
    """Paper abstracts and all author names are mapped correctly."""
    paper = SimpleNamespace(
        title="A paper", summary="First line\nsecond line",
        authors=[SimpleNamespace(name="Alice"), SimpleNamespace(name="Bob")],
        published=datetime(2026, 1, 1, tzinfo=timezone.utc),
        entry_id="https://arxiv.org/abs/example",
    )
    client = SimpleNamespace(results=lambda search: [paper])

    documents = collect_arxiv("neural networks", 1, client=client)

    assert len(documents) == 1
    assert isinstance(documents[0], ArxivDocument)
    assert documents[0].texte == "First line second line"
    assert documents[0].get_authors() == ["Alice", "Bob"]

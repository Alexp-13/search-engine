"""Manage document collections acquired from Reddit and arXiv."""

from .collectors import collect_arxiv, collect_reddit
from .corpus import Author, Corpus
from .documents import ArxivDocument, Document, DocumentFactory, RedditDocument
from .search import SearchEngine

__all__ = [
    "ArxivDocument",
    "Author",
    "Corpus",
    "Document",
    "DocumentFactory",
    "RedditDocument",
    "SearchEngine",
    "collect_arxiv",
    "collect_reddit",
]

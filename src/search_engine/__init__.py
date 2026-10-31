"""Manage document collections acquired from Reddit and arXiv."""

from .collectors import collect_arxiv, collect_reddit
from .corpus import Author, Corpus
from .documents import ArxivDocument, Document, DocumentFactory, RedditDocument

__all__ = [
    "ArxivDocument",
    "Author",
    "Corpus",
    "Document",
    "DocumentFactory",
    "RedditDocument",
    "collect_arxiv",
    "collect_reddit",
]

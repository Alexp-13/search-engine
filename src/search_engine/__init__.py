"""Manage document collections acquired from Reddit and arXiv."""

from .documents import ArxivDocument, Document, DocumentFactory, RedditDocument

__all__ = [
    "ArxivDocument",
    "Document",
    "DocumentFactory",
    "RedditDocument",
]

"""Expose document representations and their factory."""

from .ArxivDocument import ArxivDocument
from .Document import Document
from .DocumentFactory import DocumentFactory
from .RedditDocument import RedditDocument

__all__ = ["ArxivDocument", "Document", "DocumentFactory", "RedditDocument"]

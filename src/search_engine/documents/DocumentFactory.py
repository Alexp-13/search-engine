"""Construct document objects from a source identifier and metadata."""

from datetime import datetime

from .ArxivDocument import ArxivDocument
from .Document import Document
from .RedditDocument import RedditDocument


class DocumentFactory:
    """Select the document class corresponding to a data source."""

    @staticmethod
    def create(
        source: str, titre: str, auteur: str, date: datetime, url: str, texte: str,
        *, nb_comments: int = 0, coauthors: list[str] | None = None,
    ) -> Document:
        """Create a document with source-specific metadata.

        Args:
            source: reddit, arxiv, or document for an unspecified source.
            titre: Document title.
            auteur: Main author's name.
            date: Publication date and time.
            url: URL of the original document.
            texte: Document body.
            nb_comments: Comment count for Reddit posts.
            coauthors: Additional author names for arXiv papers.

        Returns:
            A document of the class corresponding to source.

        Raises:
            ValueError: If the source or its specific metadata is invalid.
        """
        if source == "reddit":
            return RedditDocument(titre, auteur, date, url, texte, nb_comments)
        if source == "arxiv":
            return ArxivDocument(titre, auteur, date, url, texte, coauthors)
        if source == "document":
            return Document(titre, auteur, date, url, texte)
        raise ValueError(f"Unknown document source: {source!r}.")

"""Define the common metadata and text shared by corpus documents."""

from datetime import datetime


class Document:
    """Represent a document independently of its data source.

    Attributes:
        titre: Document title.
        auteur: Main author's name.
        date: Publication date and time.
        url: URL of the original document.
        texte: Document body.
        type: Source identifier, or document for an unspecified source.
    """

    def __init__(
        self,
        titre: str,
        auteur: str,
        date: datetime,
        url: str,
        texte: str,
    ) -> None:
        """Initialize a document with its metadata and text.

        Args:
            titre: Document title.
            auteur: Main author's name.
            date: Publication date and time.
            url: URL of the original document.
            texte: Document body.
        """
        self.titre = titre
        self.auteur = auteur
        self.date = date
        self.url = url
        self.texte = texte
        self.type = "document"

    def getType(self) -> str:
        """Return the document's source identifier."""
        return self.type

    def get_authors(self) -> list[str]:
        """Return the names of the authors associated with this document."""
        return [self.auteur]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata and text, excluding the corpus ID.

        Dates are ISO 8601 strings. Source-specific fields default to null for
        the comment count and an empty list for coauthors.
        """
        return {
            "titre": self.titre,
            "auteur": self.auteur,
            "date": self.date.isoformat(),
            "url": self.url,
            "texte": self.texte,
            "origine": self.getType(),
            "nb_comments": None,
            "coauthors": [],
        }

    def display(self) -> None:
        """Print all document metadata and the complete text to standard output."""
        print(
            f"Title: {self.titre}\n"
            f"Author: {self.auteur}\n"
            f"Date: {self.date.isoformat()}\n"
            f"URL: {self.url}\n"
            f"Source: {self.getType()}\n"
            f"Text: {self.texte}"
        )

    def __str__(self) -> str:
        """Return the document title as a concise, readable representation."""
        return self.titre

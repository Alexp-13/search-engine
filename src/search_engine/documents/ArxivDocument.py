"""Represent arXiv papers and their coauthors."""

from datetime import datetime

from .Document import Document


class ArxivDocument(Document):
    """Represent an arXiv paper with a main author and additional authors.

    Attributes:
        coauthors: Distinct author names, excluding the main author.
    """

    def __init__(
        self, titre: str, auteur: str, date: datetime, url: str, texte: str,
        coauthors: list[str] | None = None,
    ) -> None:
        """Initialize an arXiv paper.

        Args:
            titre: Paper title.
            auteur: First listed author.
            date: Publication date and time.
            url: URL of the paper's abstract page.
            texte: Paper abstract.
            coauthors: Remaining authors; defaults to an empty list.

        Raises:
            ValueError: If coauthors is not a list of strings.
        """
        super().__init__(titre, auteur, date, url, texte)
        self.type = "arxiv"
        self.set_coauthors([] if coauthors is None else coauthors)

    def get_coauthors(self) -> list[str]:
        """Return a copy of the paper's coauthor names."""
        return self.coauthors.copy()

    def set_coauthors(self, names: list[str]) -> None:
        """Replace the coauthor list, removing duplicates and the main author.

        Args:
            names: List of coauthor names in publication order.

        Raises:
            ValueError: If names is not a list of strings.
        """
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("Coauthors must be a list of strings.")
        self.coauthors = [name for name in dict.fromkeys(names) if name != self.auteur]

    def get_authors(self) -> list[str]:
        """Return the main author followed by the coauthors."""
        return [self.auteur, *self.coauthors]

    def getType(self) -> str:
        """Return the arxiv source identifier."""
        return self.type

    def to_dict(self) -> dict[str, object]:
        """Return common metadata and the arXiv coauthor list."""
        record = super().to_dict()
        record["coauthors"] = self.get_coauthors()
        return record

    def display(self) -> None:
        """Print all document fields and the coauthor names."""
        super().display()
        print(f"Coauthors: {', '.join(self.coauthors) or 'None'}")

    def __str__(self) -> str:
        """Return the title, arXiv source label, and author count."""
        return f"[Arxiv] {self.titre} ({len(self.get_authors())} authors)"

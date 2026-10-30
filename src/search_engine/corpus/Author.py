"""Track the documents associated with an author."""

from ..documents.Document import Document


class Author:
    """Represent an author and their document production.

    Attributes:
        name: Author's name, used as their identifier in a corpus.
        production: Documents indexed by their corpus IDs.
        nb_docs: Number of distinct document IDs in production.
    """

    def __init__(self, name: str) -> None:
        """Initialize an author with no documents.

        Args:
            name: Author's name.
        """
        self.name = name
        self.production: dict[int, Document] = {}
        self.nb_docs = 0

    def add(self, doc_id: int, document: Document) -> None:
        """Register a document under its corpus ID.

        Args:
            doc_id: ID assigned by the corpus.
            document: Document associated with this author. Reusing an ID
                replaces its previous document without increasing the count.
        """
        self.production[doc_id] = document
        self.nb_docs = len(self.production)

    def mean_document_length(self) -> float:
        """Return mean text length in characters, or 0.0 for no documents."""
        if not self.production:
            return 0.0
        return sum(len(doc.texte) for doc in self.production.values()) / self.nb_docs

    def __str__(self) -> str:
        """Return the author's name and document count."""
        return f"{self.name}: {self.nb_docs} documents"

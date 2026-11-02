"""Manage the active corpus and persist it as JSONL through pandas tables."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..documents.Document import Document
from ..documents.DocumentFactory import DocumentFactory
from .Author import Author


class Corpus:
    """Manage one active document collection per Python process.

    Repeated construction returns the same instance without resetting its
    contents or changing its name. Calling load() replaces the collection.

    Attributes:
        nom: Corpus name, assigned at initialization or restored by load().
        documents: Documents indexed by unique nonnegative integer IDs.
        authors: Authors indexed by name.
        id_document: Next available document ID, starting at zero.
    """

    _instance: ClassVar[Corpus | None] = None

    def __new__(cls, nom: str) -> Corpus:
        """Return the existing instance, creating it on the first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, nom: str) -> None:
        """Initialize the active corpus on its first construction only.

        Args:
            nom: Corpus name; ignored if the Singleton already exists.
        """
        # Python calls __init__ even when __new__ returns an existing object.
        if hasattr(self, "documents"):
            return
        self.nom = nom
        self.documents: dict[int, Document] = {}
        self.authors: dict[str, Author] = {}
        self.id_document = 0
        self._text_cache: str | None = None
        self._document_bounds: list[tuple[int, int]] = []

    def add(self, document: Document) -> int:
        """Assign a document ID and register all of its authors.

        Args:
            document: Document to register in the corpus.

        Returns:
            The newly assigned document ID.
        """
        doc_id = self.id_document
        self.documents[doc_id] = document
        for name in document.get_authors():
            if name not in self.authors:
                self.authors[name] = Author(name)
            self.authors[name].add(doc_id, document)
        self.id_document += 1
        self._invalidate_text_cache()
        return doc_id

    @staticmethod
    def _build_authors(documents: dict[int, Document]) -> dict[str, Author]:
        """Build author production dictionaries from a document collection."""
        authors: dict[str, Author] = {}
        for doc_id, document in documents.items():
            for name in document.get_authors():
                if name not in authors:
                    authors[name] = Author(name)
                authors[name].add(doc_id, document)
        return authors

    def remove_short_documents(self, min_length: int = 100) -> int:
        """Remove documents whose text is shorter than min_length characters.

        Authors are rebuilt from the remaining documents. Existing document
        IDs and the next available ID are preserved.

        Args:
            min_length: Minimum text length to retain, including spaces.

        Returns:
            Number of documents removed.

        Raises:
            ValueError: If min_length is negative.
        """
        if min_length < 0:
            raise ValueError("Minimum length must be nonnegative.")
        previous_count = len(self)
        self.documents = {
            doc_id: doc for doc_id, doc in self.documents.items()
            if len(doc.texte) >= min_length
        }
        self.authors = self._build_authors(self.documents)
        self._invalidate_text_cache()
        return previous_count - len(self)

    def author_stats(self, name: str) -> dict[str, int | float]:
        """Return an author's document count and mean text length.

        Args:
            name: Author name registered in this corpus.

        Returns:
            A dictionary with nb_docs and mean_document_length (characters).

        Raises:
            KeyError: If the author is not registered.
        """
        author = self.authors[name]
        return {
            "nb_docs": author.nb_docs,
            "mean_document_length": author.mean_document_length(),
        }

    def document_stats(self) -> list[dict[str, int]]:
        """Count words and sentences in each document.

        Words are separated by spaces and sentences by periods.
        Whitespace-only pieces are ignored.

        Returns:
            A list of dictionaries with doc_id, word_count, and sentence_count,
            ordered by document ID. An empty corpus returns an empty list.
        """
        return [
            {
                "doc_id": doc_id,
                "word_count": sum(bool(word.strip()) for word in doc.texte.split(" ")),
                "sentence_count": sum(
                    bool(sentence.strip()) for sentence in doc.texte.split(".")
                ),
            }
            for doc_id, doc in sorted(self.documents.items())
        ]

    def _invalidate_text_cache(self) -> None:
        """Discard the joined text and its document boundaries."""
        self._text_cache = None
        self._document_bounds = []

    def get_text(self) -> str:
        """Return document texts joined in ID order with newline separators.

        Build and cache the string and document boundaries on the first call.
        Adding, filtering, or successfully loading documents invalidates them.
        Direct edits to document text or the documents dictionary are not
        detected automatically.

        Returns:
            The cached joined text, or an empty string for an empty corpus.
        """
        if self._text_cache is None:
            texts: list[str] = []
            bounds: list[tuple[int, int]] = []
            offset = 0
            for _, document in sorted(self.documents.items()):
                text = document.texte
                texts.append(text)
                bounds.append((offset, offset + len(text)))
                offset += len(text) + 1
            self._text_cache = "\n".join(texts)
            self._document_bounds = bounds
        return self._text_cache

    def _find_matches(
        self, pattern: re.Pattern[str],
    ) -> Iterator[tuple[re.Match[str], int, int]]:
        """Find regex matches within cached document boundaries.

        Args:
            pattern: Compiled expression to search for.

        Yields:
            A match and the start and end positions of its containing document,
            ordered by document ID and occurrence position.
        """
        text = self.get_text()
        for start, end in self._document_bounds:
            for match in pattern.finditer(text, start, end):
                yield match, start, end

    def search(self, mot_cle: str) -> list[str]:
        """Return passages containing a literal keyword, ignoring case.

        Matches may be substrings of words and do not overlap. Search the
        cached joined text within each document's boundaries.

        Args:
            mot_cle: Literal text to find; regex characters are escaped.

        Returns:
            One passage per occurrence, in document ID and occurrence order.
            Passages preserve the original text and include up to 30
            characters on each side, bounded by the containing document.
            An empty keyword or no matches returns an empty list.
        """
        if not mot_cle:
            return []
        pattern = re.compile(re.escape(mot_cle), re.IGNORECASE)
        passages: list[str] = []
        for match, start, end in self._find_matches(pattern):
            left = max(start, match.start() - 30)
            right = min(end, match.end() + 30)
            passages.append(match.string[left:right])
        return passages

    def concorde(self, expression: str, taille_contexte: int) -> pd.DataFrame:
        """Return a concordance table for a case-insensitive regex.

        Contexts preserve the original text and stay within the document
        containing each match. Valid zero-width matches are included.

        Args:
            expression: Regular expression to search for.
            taille_contexte: Maximum number of characters on each side.
                Zero returns only the matched text, with empty contexts.

        Returns:
            A DataFrame with contexte gauche, motif trouvé, and contexte droit
            columns, ordered by document ID and occurrence position.
            Empty expressions and absent matches return an empty table with
            the same columns.

        Raises:
            ValueError: If taille_contexte is negative.
            re.error: If expression is not a valid regular expression.
        """
        if taille_contexte < 0:
            raise ValueError("Context size must be nonnegative.")
        rows: list[tuple[str, str, str]] = []
        if expression:
            pattern = re.compile(expression, re.IGNORECASE)
            for match, start, end in self._find_matches(pattern):
                left = max(start, match.start() - taille_contexte) # stop the context from going beyond the document's start
                right = min(end, match.end() + taille_contexte) # stop the context from going beyond the document's end
                rows.append((
                    match.string[left:match.start()],
                    match.group(),
                    match.string[match.end():right],
                ))
        return pd.DataFrame(
            rows, columns=["contexte gauche", "motif trouvé", "contexte droit"],
        )

    def show_by_date(self, n: int = 10) -> None:
        """Print up to n documents, newest first, with IDs breaking date ties.

        Naive publication dates are interpreted as UTC for sorting.
        Stored dates are left unchanged.

        Args:
            n: Maximum number of documents to print.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("The number of documents must be nonnegative.")

        def date_key(item: tuple[int, Document]) -> datetime:
            """Make naive and timezone-aware dates comparable."""
            date = item[1].date
            return date.replace(tzinfo=UTC) if date.tzinfo is None else date

        documents = sorted(self.documents.items())
        # Stable sorting preserves ascending ID order for equal dates.
        documents.sort(key=date_key, reverse=True)
        for doc_id, document in documents[:n]:
            print(f"{doc_id}: {document}")

    def show_by_title(self, n: int = 10) -> None:
        """Print up to n documents in case-insensitive title order.

        Args:
            n: Maximum number to print; ties are ordered by document ID.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError("The number of documents must be nonnegative.")
        documents = sorted(
            self.documents.items(), key=lambda item: (item[1].titre.casefold(), item[0])
        )
        for doc_id, document in documents[:n]:
            print(f"{doc_id}: {document}")

    def save(self, path: str | Path) -> None:
        """Save corpus metadata and a document DataFrame as UTF-8 JSONL.

        The first record stores the corpus name and next document ID. Remaining
        records contain the documents, with an explicit id column and ISO 8601
        dates. Author associations are reconstructed from these documents.

        Args:
            path: Output file; its parent directory must already exist.

        Raises:
            OSError: If the file cannot be written.
        """
        records = [
            {"id": doc_id, **doc.to_dict()}
            for doc_id, doc in sorted(self.documents.items())
        ]
        table = pd.DataFrame(
            records,
            columns=[
                "id", "titre", "auteur", "date", "url", "texte", "origine",
                "nb_comments", "coauthors",
            ],
            dtype=object,
        )
        metadata = {
            "record_type": "corpus",
            "nom": self.nom,
            "id_document": self.id_document,
        }
        with Path(path).open("w", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            table.to_json(
                output,
                orient="records",
                lines=True,
                date_format="iso",
                force_ascii=False,
            )

    def load(self, path: str | Path) -> Corpus:
        """Restore corpus metadata, documents, and author associations from JSONL.

        Restore the name and next ID from the first metadata record, and read
        documents through a DataFrame. Validate all data before replacing the
        active state. Blank lines are ignored. For older document-only files,
        preserve the current name and derive the next ID from the document IDs.
        An empty legacy file clears the collection and resets its next ID.

        Args:
            path: JSONL file produced by save().

        Returns:
            This Corpus instance, with its restored name, documents, authors,
            and next document ID.

        Raises:
            ValueError: If metadata or documents are invalid, IDs are duplicated,
                or the saved next ID would collide with a document ID.
                The current corpus is unchanged.
            OSError: If the file cannot be read.
        """
        name = self.nom
        next_id: int | None = None
        with Path(path).open(encoding="utf-8") as source:
            first_line = next((line for line in source if line.strip()), "")
            metadata = json.loads(first_line) if first_line else None
            if isinstance(metadata, dict) and metadata.get("record_type") == "corpus":
                if set(metadata) != {"record_type", "nom", "id_document"}:
                    raise ValueError(f"{path}: Invalid corpus metadata fields.")
                if not isinstance(metadata["nom"], str):
                    raise ValueError(f"{path}: Corpus name must be a string.")
                if type(metadata["id_document"]) is not int or metadata["id_document"] < 0:
                    raise ValueError(f"{path}: Next document ID must be a nonnegative integer.")
                name = metadata["nom"]
                next_id = metadata["id_document"]
            else:
                source.seek(0)

            # Preserve textual values until the document parser validates them.
            table = pd.read_json(
                source,
                orient="records",
                lines=True,
                dtype=False,
            )
        documents: dict[int, Document] = {}
        for row_number, record in enumerate(table.to_dict(orient="records"), start=1):
            try:
                doc_id, document = self._parse_record(record)
                if doc_id in documents:
                    raise ValueError(f"Duplicate document ID: {doc_id}.")
            except ValueError as error:
                raise ValueError(f"{path}: record {row_number}: {error}") from error
            documents[doc_id] = document

        minimum_next_id = max(documents, default=-1) + 1
        if next_id is None:
            next_id = minimum_next_id
        elif next_id < minimum_next_id:
            raise ValueError(f"{path}: Next document ID must exceed every saved ID.")

        authors = self._build_authors(documents)
        self.nom = name
        self.documents = documents
        self.authors = authors
        self.id_document = next_id
        self._invalidate_text_cache()
        return self

    @staticmethod
    def _parse_record(record: object) -> tuple[int, Document]:
        """Validate a JSON record and reconstruct its document.

        Args:
            record: Dictionary containing common metadata, origine,
                nb_comments, and coauthors. The date is an ISO 8601 string.

        Returns:
            A tuple containing the document ID and the reconstructed Document.

        Raises:
            ValueError: If fields are missing or unexpected, field types are
                invalid, the ID is negative, or the date cannot be parsed.
        """
        fields = {
            "id", "titre", "auteur", "date", "url", "texte", "origine",
            "nb_comments", "coauthors",
        }
        if not isinstance(record, dict) or record.keys() != fields:
            raise ValueError("Document fields do not match the JSONL schema.")
        if type(record["id"]) is not int or record["id"] < 0:
            raise ValueError("Document ID must be a nonnegative integer.")
        text_fields = {"titre", "auteur", "date", "url", "texte", "origine"}
        if any(not isinstance(record[field], str) for field in text_fields):
            raise ValueError("Document metadata and text must be strings.")
        coauthors = record["coauthors"]
        if not isinstance(coauthors, list) or any(
            not isinstance(name, str) for name in coauthors
        ):
            raise ValueError("Coauthors must be a list of strings.")
        if record["origine"] != "arxiv" and coauthors:
            raise ValueError("Only arXiv documents can have coauthors.")

        comments = record["nb_comments"]
        if record["origine"] == "reddit":
            # A pandas numeric column containing nulls may restore integers as floats.
            if type(comments) is float and comments.is_integer():
                comments = int(comments)
            if type(comments) is not int or comments < 0:
                raise ValueError("Reddit comment counts must be nonnegative integers.")
        else:
            if comments is not None and not (
                type(comments) is float and pd.isna(comments)
            ):
                raise ValueError("Only Reddit documents can have a comment count.")
            comments = 0

        document = DocumentFactory.create(
            source=record["origine"],
            titre=record["titre"],
            auteur=record["auteur"],
            date=datetime.fromisoformat(record["date"]),
            url=record["url"],
            texte=record["texte"],
            nb_comments=comments,
            coauthors=coauthors,
        )
        return record["id"], document

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.documents)

    def __repr__(self) -> str:
        """Return the corpus name and its document and author counts."""
        return (
            f"Corpus(nom={self.nom!r}, documents={len(self)}, "
            f"authors={len(self.authors)})"
        )

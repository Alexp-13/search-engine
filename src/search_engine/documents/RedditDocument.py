"""Represent Reddit posts and their comment counts."""

from datetime import datetime

from .Document import Document


class RedditDocument(Document):
    """Represent a Reddit post with its common metadata and comment count.

    Attributes:
        nb_comments: Nonnegative number of comments on the post.
    """

    def __init__(
        self, titre: str, auteur: str, date: datetime, url: str, texte: str,
        nb_comments: int = 0,
    ) -> None:
        """Initialize a Reddit post.

        Args:
            titre: Post title.
            auteur: Reddit username or a placeholder for a deleted account.
            date: Publication date and time.
            url: URL of the Reddit discussion.
            texte: Post body.
            nb_comments: Nonnegative comment count.

        Raises:
            ValueError: If nb_comments is not a nonnegative integer.
        """
        super().__init__(titre, auteur, date, url, texte)
        self.type = "reddit"
        self.set_nb_comments(nb_comments)

    def get_nb_comments(self) -> int:
        """Return the post's comment count."""
        return self.nb_comments

    def set_nb_comments(self, value: int) -> None:
        """Update the post's comment count.

        Args:
            value: Nonnegative integer comment count.

        Raises:
            ValueError: If value is not a nonnegative integer.
        """
        if type(value) is not int or value < 0:
            raise ValueError("Comment count must be a nonnegative integer.")
        self.nb_comments = value

    def getType(self) -> str:
        """Return the reddit source identifier."""
        return self.type

    def to_dict(self) -> dict[str, object]:
        """Return common metadata and the Reddit comment count."""
        record = super().to_dict()
        record["nb_comments"] = self.nb_comments
        return record

    def display(self) -> None:
        """Print all document fields and the comment count."""
        super().display()
        print(f"Comments: {self.nb_comments}")

    def __str__(self) -> str:
        """Return the title, Reddit source label, and comment count."""
        return f"[Reddit] {self.titre} ({self.nb_comments} comments)"

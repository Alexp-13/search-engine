"""Collect Reddit posts through a read-only PRAW client."""

from datetime import UTC, datetime

import praw

from ..documents import Document, DocumentFactory


def collect_reddit(
    subreddit: str, query: str, limit: int = 25, *, client: praw.Reddit,
) -> list[Document]:
    """Search a subreddit and convert its posts into Reddit documents.

    Args:
        subreddit: Subreddit name without the r/ prefix.
        query: Search keywords.
        limit: Maximum number of posts to retrieve before length filtering.
        client: Authenticated, read-only PRAW client.

    Returns:
        Documents containing post bodies from selftext, normalized whitespace,
        discussion URLs, UTC publication dates, and comment counts.
        Missing authors are represented by [deleted].

    Raises:
        ValueError: If limit is negative or the query/subreddit is empty.
    """
    if limit < 0 or not query.strip() or not subreddit.strip():
        raise ValueError("Use a nonnegative limit and nonempty query/subreddit.")
    if limit == 0:
        return []
    posts = client.subreddit(subreddit).search(query, sort="relevance", limit=limit)
    documents = []
    for post in posts:
        document = DocumentFactory.create(
            source="reddit",
            titre=" ".join(post.title.split()),
            auteur=str(post.author) if post.author is not None else "[deleted]",
            date=datetime.fromtimestamp(post.created_utc, tz=UTC), # Use UTC alias after ruff warning "help: Convert to `datetime.UTC` alias"
            url=f"https://www.reddit.com{post.permalink}",
            texte=" ".join((post.selftext or "").split()),
            nb_comments=post.num_comments,
        )
        documents.append(document)
    return documents

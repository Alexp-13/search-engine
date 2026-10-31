"""Collect paper metadata and abstracts through the arxiv client."""

import arxiv

from ..documents import Document, DocumentFactory


def collect_arxiv(
    query: str, limit: int = 25, *, client: arxiv.Client | None = None,
) -> list[Document]:
    """Search arXiv and convert paper abstracts into documents.

    Args:
        query: arXiv API search expression, such as all:"neural networks".
        limit: Maximum number of papers to retrieve before length filtering.
        client: Optional arxiv client; a default client is created if omitted.

    Returns:
        Documents containing normalized titles and abstracts, publication dates,
        abstract-page URLs, and all author names. The first author is the main
        author; remaining names are coauthors.

    Raises:
        ValueError: If limit is negative or query is empty.
    """
    if limit < 0 or not query.strip():
        raise ValueError("Use a nonnegative limit and a nonempty query.")
    if limit == 0:
        return []
    if client is None:
        client = arxiv.Client(page_size=min(limit, 100))
    search = arxiv.Search(query=query, max_results=limit)
    documents = []
    for paper in client.results(search):
        names = [author.name for author in paper.authors] or ["[unknown]"]
        document = DocumentFactory.create(
            source="arxiv",
            titre=" ".join(paper.title.split()),
            auteur=names[0],
            date=paper.published,
            url=paper.entry_id,
            texte=" ".join(paper.summary.split()),
            coauthors=names[1:],
        )
        documents.append(document)
    return documents

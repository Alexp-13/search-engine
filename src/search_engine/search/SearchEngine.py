"""Analyze text and rank documents with TF-IDF cosine similarity."""

import re
from math import log, sqrt

import pandas as pd
from scipy.sparse import csr_matrix

from ..corpus import Corpus


def nettoyer_texte(texte: str) -> str:
    """Return a lowercase copy with newlines replaced by spaces.

    Args:
        texte: Original text to clean.

    Returns:
        The cleaned text, retaining punctuation, digits, and other whitespace.
    """
    return texte.lower().replace("\n", " ")


def compter_mots(texte: str) -> dict[str, int]:
    """Count terms after lowercasing and splitting on separators.

    Args:
        texte: Document or query text. Whitespace, punctuation, and underscores
            separate terms; Unicode letters and digits are retained.

    Returns:
        A dictionary of term occurrence counts. Empty pieces are ignored,
        and text without terms produces an empty dictionary.
    """
    counts: dict[str, int] = {}
    for word in re.split(r"[\W_]+", nettoyer_texte(texte)):
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts


def analyser_corpus(
    corpus: Corpus,
) -> tuple[dict[str, dict[str, int]], pd.DataFrame, dict[int, dict[str, int]]]:
    """Build vocabulary statistics and word counts for each document.

    Read documents in ID order and clean and split each text once. Whitespace,
    punctuation, and underscores separate terms; empty pieces are ignored.
    Letters and digits are retained. Original documents are not modified.

    Args:
        corpus: Collection whose document texts should be analyzed.

    Returns:
        A tuple containing the vocabulary, the frequency DataFrame, and the
        per-document counts. Vocabulary entries contain tf (total occurrences)
        and df (number of documents containing the word). The DataFrame is
        indexed by word and has the same tf and df columns. Per-document
        dictionaries use the original corpus IDs, including empty dictionaries
        for texts without terms.
    """
    vocab: dict[str, dict[str, int]] = {}
    document_counts: dict[int, dict[str, int]] = {}
    for doc_id, document in sorted(corpus.documents.items()):
        counts = compter_mots(document.texte)
        document_counts[doc_id] = counts
        for word, count in counts.items():
            if word not in vocab:
                vocab[word] = {"tf": 0, "df": 0}
            vocab[word]["tf"] += count
            vocab[word]["df"] += 1
    freq = pd.DataFrame.from_dict(
        vocab, orient="index", columns=["tf", "df"], dtype=int,
    )
    return vocab, freq, document_counts


def construire_index(
    document_counts: dict[int, dict[str, int]],
) -> tuple[dict[str, dict[str, int]], pd.DataFrame, csr_matrix, list[int]]:
    """Build an alphabetical vocabulary and a sparse term-frequency matrix.

    Args:
        document_counts: Word counts produced by analyser_corpus, keyed by
            original document ID. Each inner dictionary contains positive
            occurrence counts, or is empty for a document without terms.
            The input dictionaries are left unchanged.

    Returns:
        A tuple containing vocab, freq, mat_TF, and doc_ids. Vocabulary entries
        have an alphabetical column ID, tf (total occurrences in the corpus),
        and df (number of documents containing the word). The word-indexed
        DataFrame has tf and df columns, both calculated from the matrix.
        Matrix cells contain raw counts for one document and one word.
        Rows follow sorted document IDs; doc_ids maps each row to its original
        ID. Empty documents retain their rows, and an empty vocabulary gives
        a matrix with zero columns.
    """
    words = sorted({word for counts in document_counts.values() for word in counts})
    vocab = {word: {"id": column} for column, word in enumerate(words)}
    doc_ids = sorted(document_counts)
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    for row, doc_id in enumerate(doc_ids):
        for word, count in document_counts[doc_id].items():
            rows.append(row)
            columns.append(vocab[word]["id"])
            values.append(count)
    mat_TF = csr_matrix(
        (values, (rows, columns)), shape=(len(doc_ids), len(vocab)), dtype=int,
    )
    total_counts = mat_TF.sum(axis=0).A1
    document_frequencies = (mat_TF > 0).sum(axis=0).A1
    for word, metadata in vocab.items():
        column = metadata["id"]
        metadata["tf"] = int(total_counts[column])
        metadata["df"] = int(document_frequencies[column])
    freq = pd.DataFrame.from_dict(
        vocab, orient="index", columns=["tf", "df"], dtype=int,
    )
    return vocab, freq, mat_TF, doc_ids


def calculer_tfidf(
    mat_TF: csr_matrix,
    vocab: dict[str, dict[str, int]],
) -> tuple[list[float], csr_matrix]:
    """Return IDF weights and a sparse TF-IDF matrix.

    Args:
        mat_TF: Raw occurrence matrix returned by construire_index. Every row
            counts as a document, including rows without terms.
        vocab: Matching vocabulary with column IDs and positive document
            frequencies in its id and df fields. An empty vocabulary is valid.

    Returns:
        A list of IDF weights in column order and a new CSR matrix named
        mat_TFxIDF. Weights use the natural logarithm log(N / df), without
        smoothing. Terms present in all N documents have zero weight.
        The matrix preserves the input shape and row and column order;
        explicit zeros are removed. With no terms, the weights are empty
        and the result has zero columns. Both inputs are left unchanged.
    """
    document_count = mat_TF.shape[0]
    idf = [0.0] * mat_TF.shape[1]
    for metadata in vocab.values():
        idf[metadata["id"]] = log(document_count / metadata["df"])  # log(N / df)
    mat_TFxIDF = mat_TF.multiply(idf).tocsr()
    mat_TFxIDF.eliminate_zeros()
    return idf, mat_TFxIDF


def vectoriser_requete(
    mots_cles: str,
    vocab: dict[str, dict[str, int]],
    idf: list[float],
) -> csr_matrix:
    """Represent a query in the indexed documents' TF-IDF vocabulary.

    Args:
        mots_cles: Query text, cleaned and counted with compter_mots.
        vocab: Vocabulary returned by construire_index, including column IDs.
        idf: IDF weights returned by calculer_tfidf, in the same column order.

    Returns:
        A CSR row with one column per vocabulary term, containing query counts
        multiplied by their IDF weights. Unknown terms are ignored and zero
        weights are omitted. Empty or zero-weight queries produce a zero row.
        Repeated terms retain their counts. The inputs are left unchanged.
    """
    columns: list[int] = []
    values: list[float] = []
    for word, count in compter_mots(mots_cles).items():
        if word in vocab:
            column = vocab[word]["id"]
            value = count * idf[column]
            if value:
                columns.append(column)
                values.append(value)
    return csr_matrix(
        (values, ([0] * len(columns), columns)),
        shape=(1, len(vocab)), dtype=float,
    )


def calculer_normes(mat_TFxIDF: csr_matrix) -> list[float]:
    """Return document vector lengths in matrix row order.

    Args:
        mat_TFxIDF: Document-by-term matrix whose row norms are needed.

    Returns:
        Euclidean norms, computed as the square root of each row's sum of
        squared weights. Empty rows have norm zero. The matrix is unchanged,
        and the norms can be reused while its weights remain unchanged.
    """
    squared_norms = mat_TFxIDF.multiply(mat_TFxIDF).sum(axis=1).A1
    return [sqrt(value) for value in squared_norms]


def calculer_similarites(
    requete: csr_matrix,
    mat_TFxIDF: csr_matrix,
    normes_documents: list[float],
) -> list[float]:
    """Compute cosine similarities between a query and indexed documents.

    Args:
        requete: Weighted query row returned by vectoriser_requete.
        mat_TFxIDF: Document matrix with the same term columns as the query.
        normes_documents: One precomputed norm per document, in matrix row
            order, returned by calculer_normes for this matrix.

    Returns:
        One cosine score per document in row order. A zero query norm or
        document norm gives score zero. Only a document-sized product vector
        is converted to a dense array; the document matrix stays sparse.
        The inputs are left unchanged.
    """
    query_norm = sqrt(float(requete.multiply(requete).sum()))
    if query_norm == 0:
        return [0.0] * mat_TFxIDF.shape[0]
    products = (mat_TFxIDF @ requete.T).toarray().ravel().tolist()
    return [
        product / (query_norm * document_norm) if document_norm else 0.0
        for product, document_norm in zip(products, normes_documents)
    ]


def classer_documents(
    scores: list[float],
    doc_ids: list[int],
    nombre_documents: int,
) -> list[tuple[int, float]]:
    """Select document IDs and scores in decreasing similarity order.

    Args:
        scores: One similarity score per document matrix row.
        doc_ids: Original document ID corresponding to each score's row.
        nombre_documents: Maximum number of results to return.

    Returns:
        Up to nombre_documents pairs of document ID and score. Only strictly
        positive scores are included; exact ties use increasing document ID.
        Scores are sorted without rounding. A zero limit or no positive
        scores returns an empty list. The inputs are left unchanged.

    Raises:
        ValueError: If nombre_documents is negative.
    """
    if nombre_documents < 0:
        raise ValueError("The number of documents must be nonnegative.")
    classement = [
        (doc_ids[row], score) for row, score in enumerate(scores) if score > 0
    ]
    classement.sort(key=lambda result: (-result[1], result[0]))
    return classement[:nombre_documents]


def get_word_frequencies(freq: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return a copy of the most frequent words and their statistics.

    Args:
        freq: Word-indexed DataFrame containing tf and df columns.
        n: Maximum number of words to return.

    Returns:
        Up to n rows, ordered by decreasing tf with alphabetical ties.
        Zero and empty input return an empty table with the same columns.
        No output is printed and the input table is left unchanged.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("The number of words must be nonnegative.")
    return (
        freq.sort_index()
        .sort_values("tf", ascending=False, kind="stable")
        .head(n)
        .copy()
    )


def stats(freq: pd.DataFrame, n: int) -> None:
    """Print the vocabulary size and the most frequent words with their counts.

    Args:
        freq: Complete word-indexed frequency table with tf and df columns.
        n: Maximum number of words to display. Zero displays only the
            vocabulary size.

    Raises:
        ValueError: If n is negative.
    """
    top_words = get_word_frequencies(freq, n)
    print(f"Distinct words: {len(freq)}")
    if not top_words.empty:
        print(top_words["tf"].to_string())

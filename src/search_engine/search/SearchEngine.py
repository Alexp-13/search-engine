"""Clean corpus texts and compute vocabulary frequencies."""

import re

import pandas as pd

from ..corpus import Corpus


def nettoyer_texte(texte: str) -> str:
    """Return a lowercase copy with newlines replaced by spaces.

    Args:
        texte: Original text to clean.

    Returns:
        The cleaned text, retaining punctuation, digits, and other whitespace.
    """
    return texte.lower().replace("\n", " ")


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
        counts: dict[str, int] = {}
        for word in re.split(r"[\W_]+", nettoyer_texte(document.texte)):
            if word:
                counts[word] = counts.get(word, 0) + 1
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

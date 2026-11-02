"""Analyze text and rank documents with TF-IDF cosine similarity."""

import re
from math import log, sqrt

import pandas as pd
from scipy.sparse import csr_matrix

from ..corpus import Corpus


class SearchEngine:
    """Index a corpus and rank its documents with TF-IDF cosine similarity.

    Construction immediately prepares the index. Recreate the engine after
    adding, removing, loading, or editing corpus documents; changes to the
    referenced corpus are not detected automatically.

    Attributes:
        corpus: Referenced document collection.
        document_counts: Term occurrence counts keyed by original document ID.
        vocab: Alphabetical term metadata with column id, total tf, and df.
        freq: Word-indexed DataFrame of total tf and document frequency df.
        mat_TF: Sparse document-by-term matrix of raw occurrence counts.
        doc_ids: Original document ID corresponding to each matrix row.
        idf: Natural-logarithm IDF weights in vocabulary column order.
        mat_TFxIDF: Sparse matrix of weighted document vectors.
        normes_documents: Precomputed document norms in matrix row order.
    """

    def __init__(self, corpus: Corpus) -> None:
        """Build vocabulary, matrices, weights, and norms from corpus documents.

        Args:
            corpus: Loaded and filtered collection to index. Its documents
                are read once for term counts and are left unchanged.
        """
        self.corpus = corpus
        self.document_counts = {
            doc_id: self.compter_mots(document.texte)
            for doc_id, document in sorted(corpus.documents.items())
        }
        self.vocab, self.freq, self.mat_TF, self.doc_ids = self.construire_index(
            self.document_counts,
        )
        self.idf, self.mat_TFxIDF = self.calculer_tfidf(self.mat_TF, self.vocab)
        self.normes_documents = self.calculer_normes(self.mat_TFxIDF)

    @staticmethod
    def nettoyer_texte(texte: str) -> str:
        """Return a lowercase copy with newlines replaced by spaces.

        Args:
            texte: Original text to clean.

        Returns:
            The cleaned text, retaining punctuation, digits, and other whitespace.
        """
        return texte.lower().replace("\n", " ")

    @staticmethod
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
        for word in re.split(r"[\W_]+", SearchEngine.nettoyer_texte(texte)):
            if word:
                counts[word] = counts.get(word, 0) + 1
        return counts

    @staticmethod
    def construire_index(
        document_counts: dict[int, dict[str, int]],
    ) -> tuple[dict[str, dict[str, int]], pd.DataFrame, csr_matrix, list[int]]:
        """Build an alphabetical vocabulary and a sparse term-frequency matrix.

        Args:
            document_counts: Word occurrence counts keyed by
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

    @staticmethod
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

    @staticmethod
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
        for word, count in SearchEngine.compter_mots(mots_cles).items():
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def stats(self, n: int) -> None:
        """Print the vocabulary size and the most frequent word counts.

        Args:
            n: Maximum number of words to display. Zero displays only the
                vocabulary size. Ties are ordered alphabetically.

        Raises:
            ValueError: If n is negative.
        """
        top_words = self.get_word_frequencies(self.freq, n)
        print(f"Distinct words: {len(self.freq)}")
        if not top_words.empty:
            print(top_words["tf"].to_string())

    def search(self, mots_cles: str, nombre_documents: int) -> pd.DataFrame:
        """Return the highest-scoring documents for a keyword query.

        Args:
            mots_cles: Query text, using the index's cleaning and IDF weights.
            nombre_documents: Maximum number of documents to return.

        Returns:
            A DataFrame with id, titre, url, and score columns in decreasing
            cosine order. Only positive scores are included, with increasing
            document IDs breaking exact ties. Empty, unknown, or zero-weight
            queries and a zero limit return an empty table with those columns.
            Index data and precomputed document norms are reused unchanged.
            Recreate the engine if the referenced corpus has changed.

        Raises:
            ValueError: If nombre_documents is negative.
        """
        if nombre_documents < 0:
            raise ValueError("The number of documents must be nonnegative.")
        columns = ["id", "titre", "url", "score"]
        if nombre_documents == 0:
            return pd.DataFrame(columns=columns)
        requete = self.vectoriser_requete(mots_cles, self.vocab, self.idf)
        scores = self.calculer_similarites(
            requete, self.mat_TFxIDF, self.normes_documents,
        )
        classement = self.classer_documents(scores, self.doc_ids, nombre_documents)
        return pd.DataFrame(
            [
                (
                    doc_id,
                    self.corpus.documents[doc_id].titre,
                    self.corpus.documents[doc_id].url,
                    score,
                )
                for doc_id, score in classement
            ],
            columns=columns,
        )

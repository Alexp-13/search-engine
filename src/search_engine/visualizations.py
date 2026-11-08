"""Build figures from word frequencies, TF-IDF weights, and publication dates."""

from collections.abc import Iterable

import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from wordcloud import STOPWORDS, WordCloud

from .documents import Document
from .search import SearchEngine


def plot_word_frequencies(table: pd.DataFrame) -> Figure:
    """Return a horizontal bar chart of total word occurrences.

    Args:
        table: Word-indexed frequency table selected by the caller. Its tf
            column contains total occurrences. Row order is preserved, with
            the first word at the top. The table is left unchanged.

    Returns:
        A Matplotlib Figure with one horizontal bar per word. An empty table
        produces a figure without bars. Displaying or saving the figure is
        left to the caller.
    """
    figure = Figure(figsize=(8, 5), layout="constrained")
    axes = figure.subplots()
    positions = range(len(table))
    axes.barh(positions, table["tf"])
    axes.set_yticks(positions, labels=table.index)
    axes.invert_yaxis()
    axes.set_xlabel("Occurrences")
    axes.set_ylabel("Words")
    axes.set_title("Most frequent words")
    return figure


def filter_stopwords(table: pd.DataFrame) -> pd.DataFrame:
    """Return frequency rows excluding common English words.

    Args:
        table: Word-indexed frequency table, left unchanged.

    Returns:
        A copy excluding words in WordCloud's English stopword list,
        compared without case sensitivity. Row order, counts, and columns
        are preserved. Filter before selecting the most frequent words.
    """
    keep = [word.casefold() not in STOPWORDS for word in table.index]
    return table.loc[keep].copy()


def plot_word_cloud(table: pd.DataFrame, *, weight_column: str = "tf") -> Figure:
    """Return a word cloud sized from supplied word weights.

    Args:
        table: Word-indexed table, left unchanged.
        weight_column: Column containing word weights; defaults to tf for
            total occurrences. Only positive weights are used.

    Returns:
        A Matplotlib Figure with a reproducible word cloud on a white
        background. Words are sized from the supplied weights without
        tokenizing again or removing stopwords. An empty selection produces
        a figure without an image. Displaying or saving is left to the caller.
    """
    figure = Figure(figsize=(8, 5), layout="constrained")
    axes = figure.subplots()
    frequencies = table.loc[table[weight_column] > 0, weight_column].to_dict()
    if frequencies:
        cloud = WordCloud(
            width=800, height=400, background_color="white",
            max_words=len(frequencies), random_state=42,
        ).generate_from_frequencies(frequencies)
        axes.imshow(cloud.to_array(), interpolation="bilinear")
    axes.set_axis_off()
    axes.set_title("Word cloud")
    return figure


def plot_tfidf_word_cloud(engine: SearchEngine, n: int = 100) -> Figure:
    """Return a word cloud weighted by summed document TF-IDF scores.

    Args:
        engine: Indexed search engine, left unchanged. Reuse its existing
            TF-IDF matrix and vocabulary column IDs.
        n: Maximum number of words to display, defaulting to 100.

    Returns:
        A Matplotlib Figure sized from each word's total TF-IDF weight across
        documents. Keep the n largest positive weights, with alphabetical
        ties. No stopword filtering or additional normalization is applied.
        Terms present in every document have zero weight and are excluded.
        Empty selections produce a figure without an image. Displaying or
        saving the figure is left to the caller.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("The number of words must be nonnegative.")
    weights = engine.mat_TFxIDF.sum(axis=0).A1
    table = pd.DataFrame(
        {"tfidf": [weights[metadata["id"]] for metadata in engine.vocab.values()]},
        index=list(engine.vocab),
    )
    selected = (
        table.loc[table["tfidf"] > 0]
        .sort_index()
        .sort_values("tfidf", ascending=False, kind="stable")
        .head(n)
    )
    figure = plot_word_cloud(selected, weight_column="tfidf")
    figure.axes[0].set_title("TF-IDF word cloud")
    return figure


def plot_temporal_evolution(documents: Iterable[Document]) -> Figure:
    """Return daily document counts ordered by publication date.

    Args:
        documents: Documents to count, left unchanged. Timezone-aware dates
            are converted to UTC; naive dates are interpreted as UTC.

    Returns:
        A Matplotlib Figure showing document counts for every day between
        the first and last publication, including zero for days without
        documents. An empty collection produces a figure without a curve.
        Displaying or saving the figure is left to the caller.
    """
    figure = Figure(figsize=(8, 5), layout="constrained")
    axes = figure.subplots()
    dates = pd.to_datetime([document.date for document in documents], utc=True)
    if len(dates):
        counts = pd.Series(1, index=dates).resample("D").sum()
        axes.plot(counts.index, counts.to_numpy(), marker="o", markersize=3)
        locator = AutoDateLocator()
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    else:
        axes.set_xticks([])
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_ylim(bottom=0)
    axes.set_xlabel("Date (UTC)")
    axes.set_ylabel("Documents")
    axes.set_title("Documents over time")
    axes.grid(axis="y", alpha=0.3)
    return figure

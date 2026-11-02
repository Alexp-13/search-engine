"""Build figures from word-frequency statistics."""

import pandas as pd
from matplotlib.figure import Figure


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

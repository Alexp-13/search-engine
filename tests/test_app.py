"""Check search interactions in the Streamlit interface."""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import extraire_passage
from search_engine import Corpus, Document, SearchEngine

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app_corpus(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Corpus]:
    """Configure a saved two-document corpus and isolate the resource cache."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "https://example.test/cat", "cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "https://example.test/fish", "dog fish"))
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    monkeypatch.setenv("CORPUS_PATH", str(path))
    st.cache_resource.clear()
    yield corpus
    st.cache_resource.clear()


def test_search_and_reuse_index(app_corpus: Corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use current controls, replace results, and reuse the cached index."""
    expected = SearchEngine(app_corpus)
    engines = []
    initialize = SearchEngine.__init__

    def initialize_engine(engine: SearchEngine, corpus: Corpus) -> None:
        """Initialize and record the engine used by the interface."""
        initialize(engine, corpus)
        engines.append(engine)

    monkeypatch.setattr(SearchEngine, "__init__", initialize_engine)
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "Moteur de recherche"
    assert not app.dataframe

    app.text_input[0].set_value("cat fish")
    app.slider[0].set_value(1)
    app.button[0].click().run()
    assert not app.exception
    pd.testing.assert_frame_equal(
        app.dataframe[0].value, expected.search("cat fish", 1), check_dtype=False,
    )
    assert app.expander[0].label == "Tableau des résultats"
    assert len(app.markdown) == 1
    first_result = app.markdown[0].value
    assert first_result.index("Score :") < first_result.index("https://example.test/cat")
    assert first_result.index("https://example.test/cat") < first_result.index(">Cat</a>")
    assert first_result.index(">Cat</a>") < first_result.index("cat cat dog")
    assert "color: #1a73e8" in first_result

    app.text_input[0].set_value("fish").run()
    assert app.markdown[0].value == first_result
    pd.testing.assert_frame_equal(
        app.dataframe[0].value, expected.search("cat fish", 1), check_dtype=False,
    )
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    pd.testing.assert_frame_equal(
        app.dataframe[0].value, expected.search("fish", 1), check_dtype=False,
    )
    assert len(engines) == 1
    assert "https://example.test/fish" in app.markdown[0].value


def test_empty_then_valid_search(app_corpus: Corpus) -> None:
    """Show an empty result, then replace it with a successful search."""
    expected = SearchEngine(app_corpus)
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()
    app.text_input[0].set_value("unknown")
    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe[0].value.empty
    assert app.info[0].value == "Aucun résultat."
    assert not app.markdown

    app.text_input[0].set_value("fish")
    app.slider[0].set_value(2)
    app.button[0].click().run()
    assert not app.exception
    assert not app.info
    assert len(app.dataframe) == 1
    pd.testing.assert_frame_equal(
        app.dataframe[0].value, expected.search("fish", 2), check_dtype=False,
    )


def test_extraire_passage() -> None:
    """Find a whole query word beyond the beginning and retain its real context."""
    texte = "bobcat " + "before " * 40 + "CAT" + " after" * 40
    passage = extraire_passage(texte, "cat")

    assert "CAT" in passage
    assert "bobcat" not in passage
    assert passage.startswith("...") and passage.endswith("...")
    assert extraire_passage("A short text", "unknown") == "A short text"

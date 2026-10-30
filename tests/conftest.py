"""Provide a fresh corpus for tests that modify the Singleton."""

import pytest

from search_engine import Corpus


@pytest.fixture
def corpus() -> Corpus:
    """Return an empty corpus named Neural networks."""
    Corpus._instance = None
    return Corpus("Neural networks")

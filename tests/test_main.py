"""Test the text interface with a locally saved corpus."""

from datetime import datetime
from pathlib import Path

import pytest

from main import main, show_summary
from search_engine import Corpus, Document


def test_summary(corpus: Corpus, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Summary mode reads saved data and prints its document count."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "Hello world."))
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)

    result = main(["--corpus", str(path), "--summary"])

    assert result == 0
    assert "documents=1" in capsys.readouterr().out


def test_author_menu(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Selecting author statistics displays the chosen author's production."""
    corpus.add(Document("Example", "Alice", datetime(2026, 1, 1), "", "Hello"))
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    answers = iter(["4", "Alice", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    result = main(["--corpus", str(path)])

    assert result == 0
    output = capsys.readouterr().out
    assert "Alice: 1 documents" in output
    assert "5.0 characters" in output


def test_summary_without_building_text(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The summary counts the separator without calling get_text."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", "Hello"))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", "World"))
    monkeypatch.setattr(corpus, "get_text", lambda: pytest.fail("Text was built."))

    show_summary(corpus)

    assert "Joined text: 11 characters" in capsys.readouterr().out


def test_summary_empty_corpus(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """An empty corpus has no text or separators."""
    show_summary(corpus)

    assert "Joined text: 0 characters" in capsys.readouterr().out


def test_summary_empty_texts(corpus: Corpus, capsys: pytest.CaptureFixture[str]) -> None:
    """Two empty documents still contribute one newline separator."""
    corpus.add(Document("First", "Alice", datetime(2026, 1, 1), "", ""))
    corpus.add(Document("Second", "Bob", datetime(2026, 1, 1), "", ""))

    show_summary(corpus)

    assert "Joined text: 1 characters" in capsys.readouterr().out

"""Test the text interface with a locally saved corpus."""

from datetime import datetime
from pathlib import Path

import pytest

from main import main
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

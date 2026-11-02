"""Test the text interface with a locally saved corpus."""

from datetime import datetime
from pathlib import Path

import pytest

from main import main, show_summary
from search_engine import Corpus, Document, SearchEngine


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


def test_search_menu(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Searches load saved data, honor limits, and reuse a single engine."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "https://example.test/cat", "Cat cat dog"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "https://example.test/fish", "dog fish"))
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    answers = iter(["6", "cat cat fish", "1", "6", "fish", "", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    engines = []

    def create_engine(collection: Corpus) -> SearchEngine:
        engine = SearchEngine(collection)
        engines.append(engine)
        return engine

    monkeypatch.setattr("main.SearchEngine", create_engine)

    result = main(["--corpus", str(path)])

    output = capsys.readouterr().out
    assert result == 0
    assert len(engines) == 1
    assert "6. Search documents" in output
    assert any(line.split() == ["id", "titre", "url", "score"] for line in output.splitlines())
    assert output.count("https://example.test/cat") == 1
    assert output.count("https://example.test/fish") == 1
    assert "0.894427" in output
    assert "0.447214" not in output


def test_search_menu_invalid_input_and_no_results(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid counts and empty results allow another query in the same menu."""
    corpus.add(Document("Cat", "Alice", datetime(2026, 1, 1), "https://example.test/cat", "cat"))
    corpus.add(Document("Fish", "Bob", datetime(2026, 1, 1), "https://example.test/fish", "fish"))
    path = tmp_path / "corpus.jsonl"
    corpus.save(path)
    answers = iter([
        "6", "cat", "invalid",
        "6", "cat", "-1",
        "6", "unknown", "5",
        "6", "fish", "1",
        "0",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    result = main(["--corpus", str(path)])

    output = capsys.readouterr().out
    assert result == 0
    assert output.count("Invalid input:") == 2
    assert "No results." in output
    assert "https://example.test/fish" in output

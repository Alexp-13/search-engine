"""Check prompt limits, provider requests, and fallback without live API calls."""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
import requests

from search_engine import Document
from search_engine.generation import (
    MAX_DOCUMENT_CHARACTERS,
    GenerationError,
    analyser_documents,
    construire_prompt,
    generer_reponse,
    generer_reponse_groq,
)


@pytest.fixture
def documents() -> list[Document]:
    """Return sixteen documents with distinct texts and one long body."""
    docs = [
        Document(f"Paper {i}", "Alice", datetime(2026, 1, 1),
                 f"https://example.test/{i}", f"Content {i}")
        for i in range(16)
    ]
    docs[0].texte = "a" * (MAX_DOCUMENT_CHARACTERS + 10)
    return docs


@pytest.mark.parametrize("limit", [5, 10, 15])
def test_prompt_document_limits(documents: list[Document], limit: int) -> None:
    """Keep the first ranked documents and mark text truncation explicitly."""
    payload = json.loads(construire_prompt("keywords", documents, nombre_documents=limit))

    assert payload["requete"] == "keywords"
    assert [doc["titre"] for doc in payload["documents"]] == [f"Paper {i}" for i in range(limit)]
    assert payload["documents"][-1]["numero"] == limit
    assert len(payload["documents"][0]["texte"]) == MAX_DOCUMENT_CHARACTERS
    assert payload["documents"][0]["texte_tronque"]


def test_gemini_request(documents: list[Document], monkeypatch: pytest.MonkeyPatch) -> None:
    """Send a private header and both query-mode instructions to Gemini."""
    calls = []

    def post(url, **kwargs):
        """Capture a request and return generated text plus a hidden thought."""
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: {
            "candidates": [{"content": {"parts": [
                {"text": "Internal reasoning", "thought": True}, {"text": "Answer [1]."},
            ]}}],
        })

    monkeypatch.setattr("search_engine.generation.requests.post", post)
    assert generer_reponse("What is it?", documents, api_key="test-key") == "Answer [1]."
    url, request = calls[0]
    assert "test-key" not in url
    assert request["headers"]["x-goog-api-key"] == "test-key"
    instruction = request["json"]["systemInstruction"]["parts"][0]["text"]
    assert "mots-clés" in instruction and "question" in instruction
    assert len(json.loads(request["json"]["contents"][0]["parts"][0]["text"])["documents"]) == 5


def test_gemini_fallback(documents: list[Document], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fall back once after a Groq quota error using the same prompt data."""
    calls = []

    def post(url, **kwargs):
        """Return a quota error for Groq and an answer for Gemini."""
        calls.append((url, kwargs))
        if "api.groq.com" in url:
            return SimpleNamespace(status_code=429)
        return SimpleNamespace(status_code=200, json=lambda: {
            "candidates": [{"content": {"parts": [{"text": "Gemini answer [1]."}]}}],
        })

    monkeypatch.setattr("search_engine.generation.requests.post", post)
    result = analyser_documents("keywords", documents, gemini_key="test-key", groq_key="test-groq")

    assert result.texte == "Gemini answer [1]." and result.fournisseur == "Gemini"
    assert len(calls) == 2
    assert calls[0][1]["headers"]["Authorization"] == "Bearer test-groq"
    assert calls[1][1]["headers"]["x-goog-api-key"] == "test-key"
    assert calls[0][1]["json"]["messages"][1]["content"] == calls[1][1]["json"]["contents"][0]["parts"][0]["text"]


def test_success_does_not_use_fallback(documents: list[Document], monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Gemini unused when Groq returns a usable answer."""
    calls = []

    def post(url, **kwargs):
        """Record the single successful Groq call."""
        calls.append(url)
        return SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "Summary [1]."}}],
        })

    monkeypatch.setattr("search_engine.generation.requests.post", post)
    result = analyser_documents("keywords", documents, gemini_key="test-key", groq_key="test-groq")
    assert result.fournisseur == "Groq" and len(calls) == 1
    assert "api.groq.com" in calls[0]


def test_network_errors_are_readable(documents: list[Document], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail cleanly after both providers without exposing raw request errors."""
    def post(*args, **kwargs):
        """Simulate an error containing information unsuitable for display."""
        raise requests.Timeout("private-request-details")

    monkeypatch.setattr("search_engine.generation.requests.post", post)
    with pytest.raises(GenerationError) as failure:
        analyser_documents("keywords", documents, gemini_key="test-key", groq_key="test-groq")
    assert "Gemini" in str(failure.value) and "Groq" in str(failure.value)
    assert "private-request-details" not in str(failure.value)


def test_context_too_large(documents: list[Document], monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain a provider context-limit error without returning raw API details."""
    monkeypatch.setattr(
        "search_engine.generation.requests.post",
        lambda *args, **kwargs: SimpleNamespace(status_code=413),
    )
    with pytest.raises(GenerationError, match="Réduisez le nombre de documents"):
        generer_reponse_groq("question", documents, api_key="test-key", nombre_documents=15)

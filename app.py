"""Display ranked document search results in a Streamlit interface."""

import os
import re
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from search_engine import Corpus, Document, SearchEngine
from search_engine.generation import (
    DEFAULT_GROQ_MODEL, DEFAULT_MODEL, GenerationError, analyser_documents,
)
from search_engine.visualizations import (
    filter_stopwords, plot_temporal_evolution, plot_tfidf_word_cloud,
    plot_word_cloud, plot_word_frequencies,
)


@st.cache_resource(show_spinner="Chargement du corpus...")
def charger_moteur() -> SearchEngine:
    """Load the configured corpus and cache its search engine.

    Read CORPUS_PATH on the first call, defaulting to data/corpus.100.jsonl.
    Resolve relative paths from this file's directory. The cached engine is
    shared by sessions and used only for consultation. Restart the application
    after changing the configured path or the saved corpus.

    Returns:
        The search engine built from the saved JSONL corpus.

    Raises:
        OSError: If the corpus file cannot be read.
        ValueError: If the saved corpus is invalid.
    """
    path = Path(os.environ.get("CORPUS_PATH", "data/corpus.100.jsonl"))
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    corpus = Corpus("Moteur de recherche").load(path)
    return SearchEngine(corpus)


def clique_bouton() -> None:
    """Read the current controls and replace this session's search results."""
    mots_cles = st.session_state["mots_cles"]
    nombre_documents = st.session_state["nombre_documents"]
    st.session_state["resultats"] = charger_moteur().search(mots_cles, nombre_documents)
    st.session_state["derniere_recherche"] = (mots_cles, nombre_documents)
    for key in (
        "reponse_generation", "erreur_generation", "sources_generation", "fournisseur_generation",
    ):
        st.session_state.pop(key, None)


def analyser_resultats() -> None:
    """Generate an answer from the last search's first five documents.

    Read provider keys from the local environment without requiring a specific
    letter case. GEMINI_MODEL and GROQ_MODEL optionally select models. Keep
    the answer or a readable error in the current session. Each click tries
    Groq once and Gemini once if needed; changing widgets does not generate text.
    """
    for key in (
        "reponse_generation", "erreur_generation", "sources_generation", "fournisseur_generation",
    ):
        st.session_state.pop(key, None)
    resultats = st.session_state["resultats"]
    moteur = charger_moteur()
    documents = [moteur.corpus.documents[doc_id] for doc_id in resultats["id"].head(5)]
    requete = st.session_state["derniere_recherche"][0]
    load_dotenv(Path(__file__).resolve().parent / ".env")
    keys = {
        name.casefold(): value for name, value in os.environ.items()
        if name.casefold() in ("gemini_api_key", "groq_api_key")
    }
    try:
        with st.spinner("Analyse des documents..."):
            reponse = analyser_documents(
                requete, documents, gemini_key=keys.get("gemini_api_key", ""),
                groq_key=keys.get("groq_api_key", ""),
                gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            )
    except (ValueError, GenerationError) as error:
        st.session_state["erreur_generation"] = str(error)
    else:
        st.session_state["reponse_generation"] = reponse.texte
        st.session_state["fournisseur_generation"] = f"{reponse.fournisseur} ({reponse.modele})"
        st.session_state["sources_generation"] = [document.url for document in documents]


def extraire_passage(texte: str, mots_cles: str) -> str:
    """Extract context around the first query term in a document.

    Args:
        texte: Original document text.
        mots_cles: Query whose terms are identified with the engine's tokenizer.

    Returns:
        Up to 100 characters on each side of the first matching word, ignoring
        case. If no word matches, return the first 200 characters. Whitespace
        is collapsed and omitted text is indicated by ellipses.
    """
    termes = SearchEngine.compter_mots(mots_cles)
    debut, fin = 0, min(len(texte), 200)
    for match in re.finditer(r"[^\W_]+", texte):
        if match.group().lower() in termes:
            debut = max(0, match.start() - 100)
            fin = min(len(texte), match.end() + 100)
            break
    passage = " ".join(texte[debut:fin].split())
    return ("..." if debut else "") + passage + ("..." if fin < len(texte) else "")


def afficher_resultat(document: Document, score: float, mots_cles: str) -> None:
    """Display a score, URL, blue title link, and document passage.

    Args:
        document: Document returned by the ranked search.
        score: Its cosine similarity score, displayed with six significant digits.
        mots_cles: Query used to select the displayed passage.
    """
    titre = escape(document.titre)
    url = escape(document.url, quote=True)
    if document.url.lower().startswith(("https://", "http://")):
        titre = (
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'style="color: #1a73e8; text-decoration: none">{titre}</a>'
        )
    passage = escape(extraire_passage(document.texte, mots_cles))
    st.markdown(
        '<div style="margin-bottom: 1.5rem">'
        f'<div style="font-size: 0.85rem">Score : {score:.6g}</div>'
        f'<div style="font-size: 0.85rem; overflow-wrap: anywhere">{url}</div>'
        f'<div style="font-size: 1.25rem; color: #1a73e8">{titre}</div>'
        f'<p style="margin: 0">{passage}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render search controls, results, and collapsible corpus visualizations."""
    st.title("Moteur de recherche")
    try:
        moteur = charger_moteur()
    except (OSError, ValueError):
        st.error("Impossible de charger le corpus. Vérifiez le fichier puis redémarrez l’application.")
        return

    st.text_input("Mots clés", key="mots_cles")
    st.slider(
        "Nombre d’articles à extraire", min_value=1,
        max_value=max(10, len(moteur.doc_ids)), value=5, key="nombre_documents",
    )
    st.button("Rechercher", on_click=clique_bouton)

    sortie = st.empty()
    resultats = st.session_state.get("resultats")
    with sortie.container():
        if resultats is not None:
            with st.expander("Tableau des résultats"):
                tableau = resultats.copy()
                tableau["coauthors"] = [
                    moteur.corpus.documents[doc_id].to_dict()["coauthors"]
                    for doc_id in resultats["id"]
                ]
                st.dataframe(tableau, hide_index=True)

        with st.expander("Statistiques du corpus", expanded=False):
            corpus = moteur.corpus
            st.text(f"Corpus : {corpus.nom}")
            documents, auteurs, mots = st.columns(3)
            documents.metric("Documents", len(corpus))
            auteurs.metric("Auteurs", len(corpus.authors))
            mots.metric("Mots distincts", len(moteur.freq))
            sources = Counter(document.getType() for document in corpus.documents.values())
            st.dataframe(
                pd.DataFrame(sorted(sources.items()), columns=["origine", "documents"]),
                hide_index=True,
            )
            st.dataframe(
                pd.DataFrame(
                    corpus.document_stats(),
                    columns=["doc_id", "word_count", "sentence_count"],
                ),
                hide_index=True,
            )
            st.dataframe(SearchEngine.get_word_frequencies(moteur.freq, 10))

        with st.expander("Graphique des mots les plus fréquents", expanded=False):
            frequences = SearchEngine.get_word_frequencies(moteur.freq, 10)
            st.pyplot(plot_word_frequencies(frequences))

        with st.expander("Nuage de mots", expanded=False):
            frequences_nuage = SearchEngine.get_word_frequencies(moteur.freq, 100)
            st.pyplot(plot_word_cloud(frequences_nuage))

        with st.expander("Nuage de mots sans mots courants", expanded=False):
            frequences_filtrees = filter_stopwords(moteur.freq)
            frequences_nuage = SearchEngine.get_word_frequencies(frequences_filtrees, 100)
            st.pyplot(plot_word_cloud(frequences_nuage))

        with st.expander("Nuage de mots TF-IDF", expanded=False):
            st.pyplot(plot_tfidf_word_cloud(moteur))

        with st.expander("Évolution temporelle des documents", expanded=False):
            st.pyplot(plot_temporal_evolution(moteur.corpus.documents.values()))

        if resultats is not None:
            if resultats.empty:
                st.info("Aucun résultat.")
            else:
                st.button("Analyser les résultats", on_click=analyser_resultats)
                if "erreur_generation" in st.session_state:
                    st.error(st.session_state["erreur_generation"])
                if "reponse_generation" in st.session_state:
                    st.caption(st.session_state["fournisseur_generation"])
                    st.markdown(st.session_state["reponse_generation"])
                    for number, url in enumerate(st.session_state["sources_generation"], start=1):
                        st.caption(f"Source [{number}] : {url}")
                mots_cles = st.session_state["derniere_recherche"][0]
                for resultat in resultats.itertuples(index=False):
                    afficher_resultat(
                        moteur.corpus.documents[resultat.id], resultat.score, mots_cles,
                    )


if __name__ == "__main__":
    main()

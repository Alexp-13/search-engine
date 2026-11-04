"""Display ranked document search results in a Streamlit interface."""

import os
import re
from html import escape
from pathlib import Path

import streamlit as st

from search_engine import Corpus, Document, SearchEngine
from search_engine.visualizations import plot_word_frequencies


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
    """Render search controls, results, and a collapsible word-frequency chart."""
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
                st.dataframe(resultats, hide_index=True)

        with st.expander("Graphique des mots les plus fréquents", expanded=False):
            frequences = SearchEngine.get_word_frequencies(moteur.freq, 10)
            st.pyplot(plot_word_frequencies(frequences))

        if resultats is not None:
            if resultats.empty:
                st.info("Aucun résultat.")
            else:
                mots_cles = st.session_state["derniere_recherche"][0]
                for resultat in resultats.itertuples(index=False):
                    afficher_resultat(
                        moteur.corpus.documents[resultat.id], resultat.score, mots_cles,
                    )


if __name__ == "__main__":
    main()

# Search Engine

A Python search engine for Reddit and arXiv documents, with TF-IDF search,
a Streamlit interface, and document-based AI answers.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Alexp-13/search-engine.git
cd search-engine
uv sync
```

## Versions

`v1`, `v2`, and `v3` are Git tags.

### Version 1

```bash
git switch --detach v1
uv sync
uv run main.py
```

Explore documents and authors through the terminal menu.

### Version 2

```bash
git switch --detach v2
uv sync
uv run main.py
```

Choose option **6** to search. V1 and V2 load `data/corpus.demo.jsonl` by default.

### Version 3

```bash
git switch --detach v3
uv sync
uv run streamlit run app.py
```

Open the URL shown in the terminal. The default corpus is `data/corpus.100.jsonl`.
To use another corpus, stop the app with `Ctrl+C` and launch:

```bash
CORPUS_PATH=data/corpus.1000.jsonl uv run streamlit run app.py
CORPUS_PATH=data/corpus.10000.jsonl uv run streamlit run app.py
```

For AI answers, configure `GROQ_API_Key` in `.env` and optionally
`GEMINI_API_Key` as fallback; see `.env.example`. Search works without API keys.
After searching, click **Analyser les résultats**.

Example question for the default corpus:

> How many questions does ScienceVision contain, how many are assigned to each domain ?

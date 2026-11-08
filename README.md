# Search Engine

A Python search engine for Reddit and arXiv documents, with TF-IDF search,
a Streamlit interface, and document-based AI answers.

The Streamlit application is deployed online:

- [Application](https://perez-alexandre.dev/search/)
- [Technical documentation](https://perez-alexandre.dev/search/docs/)

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Alexp-13/search-engine.git
cd search-engine
uv sync
```

For collection or AI answers, create `.env` and fill in the required keys:

```bash
cp -n .env.example .env
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

Return to the latest development version:

```bash
git switch main
uv sync
```

## Command-line usage

Load another corpus, or print its statistics and exit:

```bash
uv run main.py --corpus data/corpus.1000.jsonl
uv run main.py --corpus data/corpus.1000.jsonl --summary
```

Optional collection of up to 25 publications per source. Configure the Reddit
credentials in `.env` first:

```bash
uv run main.py --collect \
    --query "neural networks" \
    --subreddit learnmachinelearning \
    --limit 25 \
    --corpus data/corpus.local.jsonl
```

## Tests and notebook

Run these commands from the repository root on `main`:

```bash
uv run pytest -q
uv run jupyter execute search-engine.ipynb
```

## Generate documentation

Generate HTML documentation from the docstrings into `docs/api/`:

```bash
uv run python -m pdoc search_engine main app \
    --docformat google \
    -o docs/api
```

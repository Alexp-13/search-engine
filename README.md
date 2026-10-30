# Search Engine

A search engine built from scratch in Python for indexing and retrieving documents from Reddit and arXiv.

This project is developed as part of an M1 programming course. It focuses on implementing the core information retrieval pipeline without relying on ready-made search engine libraries.

## Current status

Work in progress.

## Data sources

- Reddit, using PRAW
- arXiv, using the arXiv Python API

## Setup

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/Alexp-13/search-engine.git
cd search-engine
uv sync
```

Create a local .env file based on .env.example:

```bash
cp .env.example .env
```

Then configure your Reddit API credentials.

## Project structure

```
src/search_engine/   Core Python package
scripts/             Utility and data acquisition scripts
data/                Local corpus data
docs/                Architecture and design documentation
tests/               Tests
```

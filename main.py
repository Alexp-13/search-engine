"""Collect, save, and explore a document corpus through a text interface."""

import argparse
import os
from collections import Counter
from pathlib import Path

import arxiv
import praw
from dotenv import load_dotenv
from prawcore import PrawcoreException
from requests.exceptions import RequestException

from search_engine import Corpus
from search_engine.collectors import collect_arxiv, collect_reddit


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for corpus paths, collection options, and summary mode."""
    parser = argparse.ArgumentParser(description="Collect and explore a document corpus.")
    parser.add_argument(
        "--corpus", type=Path,
        default=Path(os.getenv("DATA_PATH", "data")) / "corpus.demo.jsonl",
        help="JSONL file to load or save (default: data/corpus.demo.jsonl).",
    )
    parser.add_argument("--query", default="neural networks", help="English topic keywords.")
    parser.add_argument("--subreddit", default="learnmachinelearning")
    parser.add_argument("--limit", type=int, default=25, help="Maximum records per source.")
    parser.add_argument("--collect", action="store_true", help="Collect both sources and save.")
    parser.add_argument("--summary", action="store_true", help="Display statistics and exit.")
    return parser


def create_reddit_client() -> praw.Reddit:
    """Create a read-only Reddit client from environment variables.

    Returns:
        A PRAW client using REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and
        REDDIT_USER_AGENT.

    Raises:
        ValueError: If any required credential is missing or a placeholder.
    """
    keys = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
    missing = [key for key in keys if not os.getenv(key) or os.environ[key].startswith("your_")]
    if missing:
        raise ValueError(f"Configure {', '.join(missing)} in .env before collecting.")
    client = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    client.read_only = True
    return client


def collect_corpus(args: argparse.Namespace) -> Corpus:
    """Collect both sources, remove short texts, and save/reload their corpus.

    Args:
        args: Parsed options containing query, subreddit, limit, and corpus path.

    Returns:
        The saved corpus with source-specific metadata and author associations.

    Raises:
        ValueError: If no documents of at least 100 characters were collected.
    """
    with create_reddit_client() as reddit_client:
        reddit_documents = collect_reddit(
            args.subreddit, args.query, args.limit, client=reddit_client
        )
    arxiv_documents = collect_arxiv(f'all:"{args.query}"', args.limit)
    corpus = Corpus(args.query)
    for document in [*reddit_documents, *arxiv_documents]:
        corpus.add(document)
    removed = corpus.remove_short_documents(100)
    if not len(corpus):
        raise ValueError("No documents of at least 100 characters were collected.")
    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.save(args.corpus)
    corpus.load(args.corpus)
    print(f"Collected {len(reddit_documents)} Reddit posts and {len(arxiv_documents)} arXiv papers.")
    print(f"Removed {removed} documents shorter than 100 characters.")
    print(f"Saved and reloaded {args.corpus}.")
    return corpus


def show_summary(corpus: Corpus) -> None:
    """Print corpus size, source counts, per-document counts, and joined text size.

    Compute the joined length without building or populating the text cache.

    Args:
        corpus: Collection whose statistics should be displayed.
    """
    print(repr(corpus))
    sources = Counter(doc.getType() for doc in corpus.documents.values())
    for source, count in sorted(sources.items()):
        print(f"{source}: {count} documents")
    for row in corpus.document_stats():
        print(f"{row['doc_id']}: {row['word_count']} words, {row['sentence_count']} sentences")
    text_length = sum(len(doc.texte) for doc in corpus.documents.values())
    text_length += max(len(corpus) - 1, 0)
    print(f"Joined text: {text_length} characters")


def run_menu(corpus: Corpus) -> None:
    """Read menu choices and display documents or author statistics.

    Args:
        corpus: Collection to explore. Entering 0 exits the menu.
    """
    while True:
        print("\n1. Newest documents\n2. Documents by title\n3. Corpus statistics")
        print("4. Author statistics\n5. Document details\n0. Quit")
        choice = input("Choice: ").strip()
        try:
            if choice == "0":
                return
            if choice in {"1", "2"}:
                count = int(input("Number of documents [10]: ").strip() or "10")
                if choice == "1":
                    corpus.show_by_date(count)
                else:
                    corpus.show_by_title(count)
            elif choice == "3":
                show_summary(corpus)
            elif choice == "4":
                name = input("Author name: ").strip()
                if name not in corpus.authors:
                    print(f"Unknown author: {name}")
                    continue
                stats = corpus.author_stats(name)
                print(f"{name}: {stats['nb_docs']} documents")
                print(f"Mean document length: {stats['mean_document_length']:.1f} characters")
            elif choice == "5":
                doc_id = int(input("Document ID: "))
                if doc_id not in corpus.documents:
                    print(f"Unknown document ID: {doc_id}")
                    continue
                corpus.documents[doc_id].display()
            else:
                print("Choose an option from 0 to 5.")
        except ValueError as error:
            print(f"Invalid input: {error}")


def main(argv: list[str] | None = None) -> int:
    """Run collection or load a saved corpus and display the text interface.

    Args:
        argv: Command-line arguments; defaults to the process arguments.

    Returns:
        Zero on success or an interrupted menu, and one on a data/API error.
    """
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        print("The per-source limit must be nonnegative.")
        return 1
    try:
        if args.collect:
            corpus = collect_corpus(args)
        else:
            corpus = Corpus(args.query).load(args.corpus)
        show_summary(corpus)
        if not args.collect and not args.summary:
            run_menu(corpus)
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
    except (
        OSError, ValueError, PrawcoreException, praw.exceptions.PRAWException,
        arxiv.ArxivError, RequestException,
    ) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

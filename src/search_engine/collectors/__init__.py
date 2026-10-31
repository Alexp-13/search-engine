"""Expose Reddit and arXiv document collectors."""

from .arxiv import collect_arxiv
from .reddit import collect_reddit

__all__ = ["collect_arxiv", "collect_reddit"]

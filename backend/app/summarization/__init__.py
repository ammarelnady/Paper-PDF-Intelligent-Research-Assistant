<<<<<<< HEAD
"""Reserved for Salma's summarization implementation."""
=======
"""
Paper summarization package: overall summaries, section summaries, and prompts.
"""

from app.summarization import prompts
from app.summarization.summarizer import (
    PaperSummarizer,
    summarize_paper,
    summarize_section,
)

__all__ = [
    "PaperSummarizer",
    "summarize_paper",
    "summarize_section",
    "prompts",
]
>>>>>>> origin/main

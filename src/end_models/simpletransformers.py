"""Backward-compatible entry point for the RoBERTa end model.

Prefer importing from `src.end_models.roberta_classifier`.
"""

from src.end_models.roberta_classifier import run_paper_end_model

if __name__ == "__main__":
    run_paper_end_model()

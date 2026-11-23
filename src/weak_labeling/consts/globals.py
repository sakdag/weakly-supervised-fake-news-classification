"""Lazy spaCy loader so importing weak-labeling modules does not require the model."""

from __future__ import annotations

from typing import Any, Optional

_NLP: Optional[Any] = None


def get_nlp():
    """Return the shared spaCy English pipeline, loading it on first use."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


class _NLPProxy:
    """Backward-compatible `NLP` object that loads spaCy on first attribute access."""

    def __getattr__(self, item):
        return getattr(get_nlp(), item)

    def __call__(self, *args, **kwargs):
        return get_nlp()(*args, **kwargs)


NLP = _NLPProxy()

import spacy
from snorkel.preprocess.nlp import SpacyPreprocessor

NLP = spacy.load("en_core_web_sm")

spacy_content = SpacyPreprocessor(
    text_field="content", doc_field="spacy_content", memoize=True
)
spacy_title = SpacyPreprocessor(
    text_field="title", doc_field="spacy_title", memoize=True
)

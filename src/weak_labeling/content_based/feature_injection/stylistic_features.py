import numpy as np
import pandas as pd
from tqdm import tqdm

from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.consts.globals import NLP


def append_stylistic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize empty columns to fill while iterating
    tqdm.pandas()

    df[
        [
            ContentFeatures.CONTENT_WORDS_PER_SENTENCE,
            ContentFeatures.CONTENT_STOPWORDS_PER_SENTENCE,
        ]
    ] = df.progress_apply(
        lambda df_row: append_stylistic_features_helper(df_row),
        axis=1,
        result_type="expand",
    )

    return df


def append_stylistic_features_helper(row):
    content_spacy = NLP(row.content)
    num_of_words = len(row.content.split())
    num_of_sentences = sum(1 for dummy in content_spacy.sents)

    num_of_stopwords = 0
    for token in content_spacy:
        if token.is_stop:
            num_of_stopwords += 1

    # Word per sentence
    content_word_per_sentence = num_of_words / num_of_sentences
    content_stopwords_per_sentence = num_of_stopwords / num_of_sentences

    return content_word_per_sentence, content_stopwords_per_sentence

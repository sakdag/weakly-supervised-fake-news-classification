from statistics import mean

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
            ContentFeatures.CONTENT_WORD_COUNT,
            ContentFeatures.CONTENT_WORDS_PER_SENTENCE,
            ContentFeatures.CONTENT_STOPWORDS_PER_SENTENCE,
            ContentFeatures.CONTENT_UNIQUE_WORDS_RATIO,
            ContentFeatures.CONTENT_UNIQUE_WORDS_COUNT,
            ContentFeatures.CONTENT_AVERAGE_SENTENCE_LENGTH,
            ContentFeatures.TITLE_WORD_COUNT,
        ]
    ] = df.progress_apply(
        lambda df_row: append_stylistic_features_helper(df_row),
        axis=1,
        result_type="expand",
    )

    return df


def append_stylistic_features_helper(row):
    # Content
    content_spacy = NLP(row.content)
    content_word_count = len(content_spacy)

    num_of_stopwords = 0
    for token in content_spacy:
        if token.is_stop:
            num_of_stopwords += 1

    sentence_lengths = []
    num_of_sentences = 0
    for sentence in content_spacy.sents:
        num_of_sentences += 1
        sentence_lengths.append(len(sentence))

    words_lower = [word.text.lower() for word in content_spacy]
    unique_words = set(words_lower)

    # Word per sentence
    content_word_per_sentence = content_word_count / num_of_sentences
    content_stopwords_per_sentence = num_of_stopwords / num_of_sentences
    content_unique_words_ratio = len(unique_words) / content_word_count
    content_unique_words_count = len(unique_words)
    content_average_sentence_length = mean(sentence_lengths)

    # Title
    title_spacy = NLP(row.title)
    title_word_count = len(title_spacy)

    return (
        content_word_count,
        content_word_per_sentence,
        content_stopwords_per_sentence,
        content_unique_words_ratio,
        content_unique_words_count,
        content_average_sentence_length,
        title_word_count,
    )

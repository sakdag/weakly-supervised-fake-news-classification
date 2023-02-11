from statistics import mean
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from dask.diagnostics import ProgressBar

from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.consts.globals import NLP
import dask.dataframe as dd

from src.weak_labeling.consts.spacy_consts import (
    ADJECTIVE,
    NOUN,
    PRONOUN,
    PROPER_NOUN,
    NUMBER,
    CARDINAL_NUMBER,
    VERB,
    VERB_PRESENT_PARTICIPLE,
    VERB_PAST_PARTICIPLE,
    VERB_THIRD_PERSON,
    DETERMINER,
    ADPOSITION,
    ADVERB,
    COMPARATIVE_ADVERB,
    SUPERLATIVE_ADVERB,
    PERSONAL_PRONOUN,
    POSSESSIVE_PRONOUN,
    INTERJECTION,
    SYMBOL,
    WH_DETERMINER,
    VERB_MODAL,
    EXISTENTIAL,
    PUNCTUATION,
    PERIOD,
    QUESTION_MARK,
    COMMA,
    EXCLAMATION_POINT,
    COLON,
    SEMICOLON,
    PARENTHESES_OPEN,
    PARENTHESES_CLOSE,
    QUOTATION_MARK,
    SPACE,
)


def append_stylistic_features(df: pd.DataFrame, features=List[str]) -> pd.DataFrame:
    if not features:
        return df

    # Initialize empty columns to fill while iterating
    tqdm.pandas()

    res = df.progress_apply(
        append_stylistic_features_helper,
        axis=1,
        result_type="expand",
        features_to_apply=features,
    )

    res.columns = features

    full = df.merge(res, left_index=True, right_index=True)
    print(full)

    return full


def append_stylistic_features_helper(row, **kwargs):
    features = kwargs["features_to_apply"]

    # Content
    content_spacy = NLP(row.content)

    word_lengths = []

    num_of_stopwords = 0
    num_of_words = 0

    # Pos tagging features
    num_of_adjectives = 0
    num_of_nouns = 0
    num_of_proper_nouns = 0
    num_of_cardinal_numbers = 0
    num_of_verbs = 0
    num_of_present_participle_verbs = 0
    num_of_past_participle_verbs = 0
    num_of_third_person_verbs = 0
    num_of_determiners = 0
    num_of_adpositions = 0
    num_of_adverbs = 0
    num_of_personal_pronouns = 0
    num_of_possessive_pronouns = 0
    num_of_verb_modals = 0
    num_of_interjections = 0
    num_of_pronouns = 0
    num_of_existentials = 0
    num_of_wh_determiners = 0
    num_of_comparative_adverbs = 0
    num_of_superlative_adverbs = 0
    num_of_symbols = 0

    # Punctuation features
    num_of_punctuations = 0
    num_of_periods = 0
    num_of_question_marks = 0
    num_of_exclamation_points = 0
    num_of_commas = 0
    num_of_colons = 0
    num_of_semicolons = 0
    num_of_parentheses_openers = 0
    num_of_parentheses_closers = 0
    num_of_quotation_marks = 0

    for token in content_spacy:
        if token.is_stop:
            num_of_stopwords += 1
        if token.pos_ == NOUN:
            num_of_nouns += 1
        elif token.pos_ == PROPER_NOUN:
            num_of_proper_nouns += 1
        elif token.pos_ == NUMBER:
            if token.tag_ == CARDINAL_NUMBER:
                num_of_cardinal_numbers += 1
        elif token.pos_ == DETERMINER:
            num_of_determiners += 1
        elif token.pos_ == ADPOSITION:
            num_of_adpositions += 1
        elif token.pos_ == INTERJECTION:
            num_of_interjections += 1
        elif token.pos_ == SYMBOL:
            num_of_symbols += 1
        elif token.pos_ == ADJECTIVE:
            num_of_adjectives += 1
            if token.tag_ == WH_DETERMINER:
                num_of_wh_determiners += 1
        elif token.pos_ == VERB:
            num_of_verbs += 1
            if token.tag_ == VERB_PRESENT_PARTICIPLE:
                num_of_present_participle_verbs += 1
            elif token.tag_ == VERB_PAST_PARTICIPLE:
                num_of_past_participle_verbs += 1
            elif token.tag_ == VERB_THIRD_PERSON:
                num_of_third_person_verbs += 1
            elif token.tag_ == VERB_MODAL:
                num_of_verb_modals += 1
        elif token.pos_ == ADVERB:
            num_of_adverbs += 1
            if token.tag_ == COMPARATIVE_ADVERB:
                num_of_comparative_adverbs += 1
            elif token.tag_ == SUPERLATIVE_ADVERB:
                num_of_superlative_adverbs += 1
            elif token.tag_ == EXISTENTIAL:
                num_of_existentials += 1
        elif token.pos_ == PRONOUN:
            num_of_pronouns += 1
            if token.tag_ == PERSONAL_PRONOUN:
                num_of_personal_pronouns += 1
            elif token.tag_ == POSSESSIVE_PRONOUN:
                num_of_possessive_pronouns += 1
        elif token.pos_ == PUNCTUATION:
            num_of_punctuations += 1
            if token.tag_ == PERIOD:
                num_of_periods += 1
            elif token.tag_ == QUESTION_MARK:
                num_of_question_marks += 1
            elif token.tag_ == COMMA:
                num_of_commas += 1
            elif token.tag_ == EXCLAMATION_POINT:
                num_of_exclamation_points += 1
            elif token.tag_ == COLON:
                num_of_colons += 1
            elif token.tag_ == SEMICOLON:
                num_of_semicolons += 1
            elif token.tag_ == PARENTHESES_OPEN:
                num_of_parentheses_openers += 1
            elif token.tag_ == PARENTHESES_CLOSE:
                num_of_parentheses_closers += 1
            elif token.tag_ == QUOTATION_MARK:
                num_of_quotation_marks += 1

        if token.pos_ not in [PUNCTUATION, SYMBOL, SPACE]:
            num_of_words += 1
            word_lengths.append(len(token.text))

    title_spacy = NLP(row.title)
    num_of_title_punctuations = 0
    num_of_title_words = 0
    for title_token in title_spacy:
        if title_token.pos_ == PUNCTUATION:
            num_of_title_punctuations += 1
        if title_token.pos_ not in [PUNCTUATION, SYMBOL, SPACE]:
            num_of_title_words += 1

    sentence_lengths = []
    num_of_sentences = 0
    for sentence in content_spacy.sents:
        num_of_sentences += 1
        sentence_lengths.append(len(sentence))

    words_lower = [word.text.lower() for word in content_spacy]
    unique_words = set(words_lower)

    features_to_return = []

    # content_word_count = len(content_spacy)
    if ContentFeatures.CONTENT_WORD_COUNT in features:
        features_to_return.append(num_of_words)
    if ContentFeatures.TITLE_WORD_COUNT in features:
        features_to_return.append(num_of_title_words)
    if ContentFeatures.CONTENT_UNIQUE_WORDS_COUNT in features:
        features_to_return.append(len(unique_words))

    if ContentFeatures.CONTENT_WORDS_PER_SENTENCE in features:
        features_to_return.append(num_of_words / num_of_sentences)
    if ContentFeatures.CONTENT_STOPWORDS_RATIO in features:
        features_to_return.append(num_of_stopwords / num_of_words)
    if ContentFeatures.CONTENT_UNIQUE_WORDS_RATIO in features:
        features_to_return.append(len(unique_words) / num_of_words)
    if ContentFeatures.CONTENT_AVERAGE_SENTENCE_LENGTH in features:
        features_to_return.append(mean(sentence_lengths))
    if ContentFeatures.CONTENT_AVERAGE_WORD_LENGTH in features:
        features_to_return.append(mean(word_lengths))

    if ContentFeatures.CONTENT_NOUN_RATIO in features:
        features_to_return.append(num_of_nouns / num_of_words)
    if ContentFeatures.CONTENT_PROPER_NOUN_RATIO in features:
        features_to_return.append(num_of_proper_nouns / num_of_words)
    if ContentFeatures.CONTENT_CARDINAL_NUMBER_RATIO in features:
        features_to_return.append(num_of_cardinal_numbers / num_of_words)
    if ContentFeatures.CONTENT_DETERMINER_RATIO in features:
        features_to_return.append(num_of_determiners / num_of_words)
    if ContentFeatures.CONTENT_ADPOSITION_RATIO in features:
        features_to_return.append(num_of_adpositions / num_of_words)
    if ContentFeatures.CONTENT_INTERJECTION_RATIO in features:
        features_to_return.append(num_of_interjections / num_of_words)
    if ContentFeatures.CONTENT_SYMBOL_RATIO in features:
        features_to_return.append(num_of_symbols / num_of_words)

    if ContentFeatures.CONTENT_ADJECTIVE_RATIO in features:
        features_to_return.append(num_of_adjectives / num_of_words)
    if ContentFeatures.CONTENT_WH_DETERMINER_RATIO in features:
        features_to_return.append(num_of_wh_determiners / num_of_words)

    if ContentFeatures.CONTENT_VERB_RATIO in features:
        features_to_return.append(num_of_verbs / num_of_words)
    if ContentFeatures.CONTENT_VERB_PRESENT_PARTICIPLE_RATIO in features:
        features_to_return.append(num_of_present_participle_verbs / num_of_words)
    if ContentFeatures.CONTENT_VERB_PAST_PARTICIPLE_RATIO in features:
        features_to_return.append(num_of_past_participle_verbs / num_of_words)
    if ContentFeatures.CONTENT_VERB_THIRD_PERSON_RATIO in features:
        features_to_return.append(num_of_third_person_verbs / num_of_words)
    if ContentFeatures.CONTENT_VERB_MODAL_RATIO in features:
        features_to_return.append(num_of_verb_modals / num_of_words)

    if ContentFeatures.CONTENT_ADVERB_RATIO in features:
        features_to_return.append(num_of_adverbs / num_of_words)
    if ContentFeatures.CONTENT_COMPARATIVE_ADVERB_RATIO in features:
        features_to_return.append(num_of_comparative_adverbs / num_of_words)
    if ContentFeatures.CONTENT_SUPERLATIVE_ADVERB_RATIO in features:
        features_to_return.append(num_of_superlative_adverbs / num_of_words)
    if ContentFeatures.CONTENT_EXISTENTIAL_RATIO in features:
        features_to_return.append(num_of_existentials / num_of_words)

    if ContentFeatures.CONTENT_PRONOUN_RATIO in features:
        features_to_return.append(num_of_pronouns / num_of_words)
    if ContentFeatures.CONTENT_PERSONAL_PRONOUN_RATIO in features:
        features_to_return.append(num_of_personal_pronouns / num_of_words)
    if ContentFeatures.CONTENT_POSSESSIVE_PRONOUN_RATIO in features:
        features_to_return.append(num_of_possessive_pronouns / num_of_words)

    if ContentFeatures.CONTENT_PUNCTUATION_RATIO in features:
        features_to_return.append(num_of_punctuations / num_of_words)
    if ContentFeatures.CONTENT_PERIOD_RATIO in features:
        features_to_return.append(num_of_periods / num_of_words)
    if ContentFeatures.CONTENT_QUESTION_MARK_RATIO in features:
        features_to_return.append(num_of_question_marks / num_of_words)
    if ContentFeatures.CONTENT_EXCLAMATION_POINT_RATIO in features:
        features_to_return.append(num_of_exclamation_points / num_of_words)
    if ContentFeatures.CONTENT_COMMA_RATIO in features:
        features_to_return.append(num_of_commas / num_of_words)
    if ContentFeatures.CONTENT_COLON_RATIO in features:
        features_to_return.append(num_of_colons / num_of_words)
    if ContentFeatures.CONTENT_SEMICOLON_RATIO in features:
        features_to_return.append(num_of_semicolons / num_of_words)
    if ContentFeatures.CONTENT_PARENTHESES_OPEN_RATIO in features:
        features_to_return.append(num_of_parentheses_openers / num_of_words)
    if ContentFeatures.CONTENT_PARENTHESES_CLOSE_RATIO in features:
        features_to_return.append(num_of_parentheses_closers / num_of_words)
    if ContentFeatures.CONTENT_QUOTATION_MARK_RATIO in features:
        features_to_return.append(num_of_quotation_marks / num_of_words)

    if ContentFeatures.TITLE_PUNCTUATION_RATIO in features:
        features_to_return.append(num_of_title_punctuations / num_of_title_words)

    return tuple(features_to_return)


def append_stylistic_features_parallel(ddf: dd.DataFrame) -> pd.DataFrame:
    # Initialize empty columns to fill while iterating
    tqdm.pandas()
    ProgressBar().register()

    res = ddf.apply(
        append_stylistic_features_helper,
        axis=1,
        result_type="expand",
        meta={
            0: float,
            1: float,
            2: float,
            3: float,
            4: float,
            5: float,
            6: float,
        },
    )

    res.columns = [
        ContentFeatures.CONTENT_WORD_COUNT,
        ContentFeatures.CONTENT_WORDS_PER_SENTENCE,
        ContentFeatures.CONTENT_STOPWORDS_RATIO,
        ContentFeatures.CONTENT_UNIQUE_WORDS_RATIO,
        ContentFeatures.CONTENT_UNIQUE_WORDS_COUNT,
        ContentFeatures.CONTENT_AVERAGE_SENTENCE_LENGTH,
        ContentFeatures.TITLE_WORD_COUNT,
    ]

    full = ddf.merge(res, left_index=True, right_index=True)

    final_res = full.compute()
    print(final_res)
    print(len(final_res))

    return final_res

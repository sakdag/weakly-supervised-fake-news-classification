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
    MODAL,
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
    AUXILIARY,
)


def append_stylistic_features(df: pd.DataFrame, features=List[str]) -> pd.DataFrame:
    if not features:
        return df

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

    features_to_return = []
    features_to_return.extend(get_content_features(features, row.content))
    features_to_return.extend(get_title_features(features, row.title))

    return tuple(features_to_return)


def get_title_features(features, title):
    res = []

    # Content
    title_spacy = NLP(title)

    word_lengths_title = []

    num_of_stopwords_title = 0
    num_of_words_title = 0

    # Pos tagging features
    num_of_adjectives_title = 0
    num_of_nouns_title = 0
    num_of_proper_nouns_title = 0
    num_of_cardinal_numbers_title = 0
    num_of_verbs_title = 0
    num_of_present_participle_verbs_title = 0
    num_of_past_participle_verbs_title = 0
    num_of_third_person_verbs_title = 0
    num_of_determiners_title = 0
    num_of_adpositions_title = 0
    num_of_adverbs_title = 0
    num_of_personal_pronouns_title = 0
    num_of_possessive_pronouns_title = 0
    num_of_modals_title = 0
    num_of_interjections_title = 0
    num_of_pronouns_title = 0
    num_of_existentials_title = 0
    num_of_wh_determiners_title = 0
    num_of_comparative_adverbs_title = 0
    num_of_superlative_adverbs_title = 0
    num_of_symbols_title = 0

    # Punctuation features
    num_of_punctuations_title = 0

    for token in title_spacy:
        if token.is_stop:
            num_of_stopwords_title += 1
        if token.pos_ == NOUN:
            num_of_nouns_title += 1
        elif token.pos_ == PROPER_NOUN:
            num_of_proper_nouns_title += 1
        elif token.pos_ == NUMBER:
            if token.tag_ == CARDINAL_NUMBER:
                num_of_cardinal_numbers_title += 1
        elif token.pos_ == DETERMINER:
            num_of_determiners_title += 1
        elif token.pos_ == ADPOSITION:
            num_of_adpositions_title += 1
        elif token.pos_ == INTERJECTION:
            num_of_interjections_title += 1
        elif token.pos_ == SYMBOL:
            num_of_symbols_title += 1
        elif token.pos_ == ADJECTIVE:
            num_of_adjectives_title += 1
        elif token.pos_ == VERB:
            num_of_verbs_title += 1
            if token.tag_ == VERB_PRESENT_PARTICIPLE:
                num_of_present_participle_verbs_title += 1
            elif token.tag_ == VERB_PAST_PARTICIPLE:
                num_of_past_participle_verbs_title += 1
            elif token.tag_ == VERB_THIRD_PERSON:
                num_of_third_person_verbs_title += 1
        elif token.pos_ == ADVERB:
            num_of_adverbs_title += 1
            if token.tag_ == COMPARATIVE_ADVERB:
                num_of_comparative_adverbs_title += 1
            elif token.tag_ == SUPERLATIVE_ADVERB:
                num_of_superlative_adverbs_title += 1
        elif token.pos_ == PRONOUN:
            num_of_pronouns_title += 1
            if token.tag_ == PERSONAL_PRONOUN:
                num_of_personal_pronouns_title += 1
            elif token.tag_ == POSSESSIVE_PRONOUN:
                num_of_possessive_pronouns_title += 1
            elif token.tag_ == EXISTENTIAL:
                num_of_existentials_title += 1
            if token.tag_ == WH_DETERMINER:
                num_of_wh_determiners_title += 1
        elif token.pos_ == AUXILIARY:
            if token.tag_ == MODAL:
                num_of_modals_title += 1
        elif token.pos_ == PUNCTUATION:
            num_of_punctuations_title += 1

        if token.pos_ not in [PUNCTUATION, SYMBOL, SPACE]:
            num_of_words_title += 1
            word_lengths_title.append(len(token.text))

    words_lower = [word.text.lower() for word in title_spacy]
    unique_words_title = set(words_lower)

    if ContentFeatures.TITLE_WORD_COUNT in features:
        res.append(num_of_words_title)

    if ContentFeatures.TITLE_STOPWORDS_RATIO in features:
        res.append(num_of_stopwords_title / num_of_words_title)
    if ContentFeatures.TITLE_UNIQUE_WORDS_RATIO in features:
        res.append(len(unique_words_title) / num_of_words_title)
    if ContentFeatures.TITLE_AVERAGE_WORD_LENGTH in features:
        res.append(mean(word_lengths_title))

    if ContentFeatures.TITLE_NOUN_RATIO in features:
        res.append(num_of_nouns_title / num_of_words_title)
    if ContentFeatures.TITLE_PROPER_NOUN_RATIO in features:
        res.append(num_of_proper_nouns_title / num_of_words_title)
    if ContentFeatures.TITLE_CARDINAL_NUMBER_RATIO in features:
        res.append(num_of_cardinal_numbers_title / num_of_words_title)
    if ContentFeatures.TITLE_DETERMINER_RATIO in features:
        res.append(num_of_determiners_title / num_of_words_title)
    if ContentFeatures.TITLE_ADPOSITION_RATIO in features:
        res.append(num_of_adpositions_title / num_of_words_title)
    if ContentFeatures.TITLE_INTERJECTION_RATIO in features:
        res.append(num_of_interjections_title / num_of_words_title)
    if ContentFeatures.TITLE_SYMBOL_RATIO in features:
        res.append(num_of_symbols_title / num_of_words_title)

    if ContentFeatures.TITLE_ADJECTIVE_RATIO in features:
        res.append(num_of_adjectives_title / num_of_words_title)
    if ContentFeatures.TITLE_WH_DETERMINER_RATIO in features:
        res.append(num_of_wh_determiners_title / num_of_words_title)

    if ContentFeatures.TITLE_VERB_RATIO in features:
        res.append(num_of_verbs_title / num_of_words_title)
    if ContentFeatures.TITLE_VERB_PRESENT_PARTICIPLE_RATIO in features:
        res.append(num_of_present_participle_verbs_title / num_of_words_title)
    if ContentFeatures.TITLE_VERB_PAST_PARTICIPLE_RATIO in features:
        res.append(num_of_past_participle_verbs_title / num_of_words_title)
    if ContentFeatures.TITLE_VERB_THIRD_PERSON_RATIO in features:
        res.append(num_of_third_person_verbs_title / num_of_words_title)
    if ContentFeatures.TITLE_MODAL_RATIO in features:
        res.append(num_of_modals_title / num_of_words_title)

    if ContentFeatures.TITLE_ADVERB_RATIO in features:
        res.append(num_of_adverbs_title / num_of_words_title)
    if ContentFeatures.TITLE_COMPARATIVE_ADVERB_RATIO in features:
        res.append(num_of_comparative_adverbs_title / num_of_words_title)
    if ContentFeatures.TITLE_SUPERLATIVE_ADVERB_RATIO in features:
        res.append(num_of_superlative_adverbs_title / num_of_words_title)
    if ContentFeatures.TITLE_EXISTENTIAL_RATIO in features:
        res.append(num_of_existentials_title / num_of_words_title)

    if ContentFeatures.TITLE_PRONOUN_RATIO in features:
        res.append(num_of_pronouns_title / num_of_words_title)
    if ContentFeatures.TITLE_PERSONAL_PRONOUN_RATIO in features:
        res.append(num_of_personal_pronouns_title / num_of_words_title)
    if ContentFeatures.TITLE_POSSESSIVE_PRONOUN_RATIO in features:
        res.append(num_of_possessive_pronouns_title / num_of_words_title)

    if ContentFeatures.TITLE_PUNCTUATION_RATIO in features:
        res.append(num_of_punctuations_title / num_of_words_title)

    return res


def get_content_features(features, content):
    res = []

    # Content
    content_spacy = NLP(content)

    word_lengths_content = []

    num_of_stopwords_content = 0
    num_of_words_content = 0

    # Pos tagging features
    num_of_adjectives_content = 0
    num_of_nouns_content = 0
    num_of_proper_nouns_content = 0
    num_of_cardinal_numbers_content = 0
    num_of_verbs_content = 0
    num_of_present_participle_verbs_content = 0
    num_of_past_participle_verbs_content = 0
    num_of_third_person_verbs_content = 0
    num_of_determiners_content = 0
    num_of_adpositions_content = 0
    num_of_adverbs_content = 0
    num_of_personal_pronouns_content = 0
    num_of_possessive_pronouns_content = 0
    num_of_modals_content = 0
    num_of_interjections_content = 0
    num_of_pronouns_content = 0
    num_of_existentials_content = 0
    num_of_wh_determiners_content = 0
    num_of_comparative_adverbs_content = 0
    num_of_superlative_adverbs_content = 0
    num_of_symbols_content = 0

    # Punctuation features
    num_of_punctuations_content = 0
    num_of_periods_content = 0
    num_of_question_marks_content = 0
    num_of_exclamation_points_content = 0
    num_of_commas_content = 0
    num_of_colons_content = 0
    num_of_semicolons_content = 0
    num_of_parentheses_openers_content = 0
    num_of_parentheses_closers_content = 0
    num_of_quotation_marks_content = 0

    for token in content_spacy:
        if token.is_stop:
            num_of_stopwords_content += 1
        if token.pos_ == NOUN:
            num_of_nouns_content += 1
        elif token.pos_ == PROPER_NOUN:
            num_of_proper_nouns_content += 1
        elif token.pos_ == NUMBER:
            if token.tag_ == CARDINAL_NUMBER:
                num_of_cardinal_numbers_content += 1
        elif token.pos_ == DETERMINER:
            num_of_determiners_content += 1
        elif token.pos_ == ADPOSITION:
            num_of_adpositions_content += 1
        elif token.pos_ == INTERJECTION:
            num_of_interjections_content += 1
        elif token.pos_ == SYMBOL:
            num_of_symbols_content += 1
        elif token.pos_ == ADJECTIVE:
            num_of_adjectives_content += 1
        elif token.pos_ == VERB:
            num_of_verbs_content += 1
            if token.tag_ == VERB_PRESENT_PARTICIPLE:
                num_of_present_participle_verbs_content += 1
            elif token.tag_ == VERB_PAST_PARTICIPLE:
                num_of_past_participle_verbs_content += 1
            elif token.tag_ == VERB_THIRD_PERSON:
                num_of_third_person_verbs_content += 1
        elif token.pos_ == ADVERB:
            num_of_adverbs_content += 1
            if token.tag_ == COMPARATIVE_ADVERB:
                num_of_comparative_adverbs_content += 1
            elif token.tag_ == SUPERLATIVE_ADVERB:
                num_of_superlative_adverbs_content += 1
        elif token.pos_ == PRONOUN:
            num_of_pronouns_content += 1
            if token.tag_ == PERSONAL_PRONOUN:
                num_of_personal_pronouns_content += 1
            elif token.tag_ == POSSESSIVE_PRONOUN:
                num_of_possessive_pronouns_content += 1
            elif token.tag_ == EXISTENTIAL:
                num_of_existentials_content += 1
            if token.tag_ == WH_DETERMINER:
                num_of_wh_determiners_content += 1
        elif token.pos_ == AUXILIARY:
            if token.tag_ == MODAL:
                num_of_modals_content += 1
        elif token.pos_ == PUNCTUATION:
            num_of_punctuations_content += 1
            if token.text == PERIOD:
                num_of_periods_content += 1
            elif token.text == QUESTION_MARK:
                num_of_question_marks_content += 1
            elif token.text == COMMA:
                num_of_commas_content += 1
            elif token.text == EXCLAMATION_POINT:
                num_of_exclamation_points_content += 1
            elif token.text == COLON:
                num_of_colons_content += 1
            elif token.text == SEMICOLON:
                num_of_semicolons_content += 1
            elif token.text == PARENTHESES_OPEN:
                num_of_parentheses_openers_content += 1
            elif token.text == PARENTHESES_CLOSE:
                num_of_parentheses_closers_content += 1
            elif token.text == QUOTATION_MARK:
                num_of_quotation_marks_content += 1

        if token.pos_ not in [PUNCTUATION, SYMBOL, SPACE]:
            num_of_words_content += 1
            word_lengths_content.append(len(token.text))

    sentence_lengths_content = []
    num_of_sentences_content = 0
    for sentence in content_spacy.sents:
        num_of_sentences_content += 1
        sentence_lengths_content.append(len(sentence))

    words_lower = [word.text.lower() for word in content_spacy]
    unique_words_content = set(words_lower)

    if ContentFeatures.CONTENT_WORD_COUNT in features:
        res.append(num_of_words_content)
    if ContentFeatures.CONTENT_UNIQUE_WORDS_COUNT in features:
        res.append(len(unique_words_content))

    if ContentFeatures.CONTENT_WORDS_PER_SENTENCE in features:
        res.append(num_of_words_content / num_of_sentences_content)
    if ContentFeatures.CONTENT_STOPWORDS_RATIO in features:
        res.append(num_of_stopwords_content / num_of_words_content)
    if ContentFeatures.CONTENT_UNIQUE_WORDS_RATIO in features:
        res.append(len(unique_words_content) / num_of_words_content)
    if ContentFeatures.CONTENT_AVERAGE_SENTENCE_LENGTH in features:
        res.append(mean(sentence_lengths_content))
    if ContentFeatures.CONTENT_AVERAGE_WORD_LENGTH in features:
        res.append(mean(word_lengths_content))

    if ContentFeatures.CONTENT_NOUN_RATIO in features:
        res.append(num_of_nouns_content / num_of_words_content)
    if ContentFeatures.CONTENT_PROPER_NOUN_RATIO in features:
        res.append(num_of_proper_nouns_content / num_of_words_content)
    if ContentFeatures.CONTENT_CARDINAL_NUMBER_RATIO in features:
        res.append(num_of_cardinal_numbers_content / num_of_words_content)
    if ContentFeatures.CONTENT_DETERMINER_RATIO in features:
        res.append(num_of_determiners_content / num_of_words_content)
    if ContentFeatures.CONTENT_ADPOSITION_RATIO in features:
        res.append(num_of_adpositions_content / num_of_words_content)
    if ContentFeatures.CONTENT_INTERJECTION_RATIO in features:
        res.append(num_of_interjections_content / num_of_words_content)
    if ContentFeatures.CONTENT_SYMBOL_RATIO in features:
        res.append(num_of_symbols_content / num_of_words_content)

    if ContentFeatures.CONTENT_ADJECTIVE_RATIO in features:
        res.append(num_of_adjectives_content / num_of_words_content)
    if ContentFeatures.CONTENT_WH_DETERMINER_RATIO in features:
        res.append(num_of_wh_determiners_content / num_of_words_content)

    if ContentFeatures.CONTENT_VERB_RATIO in features:
        res.append(num_of_verbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_VERB_PRESENT_PARTICIPLE_RATIO in features:
        res.append(num_of_present_participle_verbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_VERB_PAST_PARTICIPLE_RATIO in features:
        res.append(num_of_past_participle_verbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_VERB_THIRD_PERSON_RATIO in features:
        res.append(num_of_third_person_verbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_MODAL_RATIO in features:
        res.append(num_of_modals_content / num_of_words_content)

    if ContentFeatures.CONTENT_ADVERB_RATIO in features:
        res.append(num_of_adverbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_COMPARATIVE_ADVERB_RATIO in features:
        res.append(num_of_comparative_adverbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_SUPERLATIVE_ADVERB_RATIO in features:
        res.append(num_of_superlative_adverbs_content / num_of_words_content)
    if ContentFeatures.CONTENT_EXISTENTIAL_RATIO in features:
        res.append(num_of_existentials_content / num_of_words_content)

    if ContentFeatures.CONTENT_PRONOUN_RATIO in features:
        res.append(num_of_pronouns_content / num_of_words_content)
    if ContentFeatures.CONTENT_PERSONAL_PRONOUN_RATIO in features:
        res.append(num_of_personal_pronouns_content / num_of_words_content)
    if ContentFeatures.CONTENT_POSSESSIVE_PRONOUN_RATIO in features:
        res.append(num_of_possessive_pronouns_content / num_of_words_content)

    if ContentFeatures.CONTENT_PUNCTUATION_RATIO in features:
        res.append(num_of_punctuations_content / num_of_words_content)
    if ContentFeatures.CONTENT_PERIOD_RATIO in features:
        res.append(num_of_periods_content / num_of_words_content)
    if ContentFeatures.CONTENT_QUESTION_MARK_RATIO in features:
        res.append(num_of_question_marks_content / num_of_words_content)
    if ContentFeatures.CONTENT_EXCLAMATION_POINT_RATIO in features:
        res.append(num_of_exclamation_points_content / num_of_words_content)
    if ContentFeatures.CONTENT_COMMA_RATIO in features:
        res.append(num_of_commas_content / num_of_words_content)
    if ContentFeatures.CONTENT_COLON_RATIO in features:
        res.append(num_of_colons_content / num_of_words_content)
    if ContentFeatures.CONTENT_SEMICOLON_RATIO in features:
        res.append(num_of_semicolons_content / num_of_words_content)
    if ContentFeatures.CONTENT_PARENTHESES_OPEN_RATIO in features:
        res.append(num_of_parentheses_openers_content / num_of_words_content)
    if ContentFeatures.CONTENT_PARENTHESES_CLOSE_RATIO in features:
        res.append(num_of_parentheses_closers_content / num_of_words_content)
    if ContentFeatures.CONTENT_QUOTATION_MARK_RATIO in features:
        res.append(num_of_quotation_marks_content / num_of_words_content)

    return res


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

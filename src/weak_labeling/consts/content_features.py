class ContentFeatures:
    # Stylistic count features
    CONTENT_WORD_COUNT = "content_word_count"
    TITLE_WORD_COUNT = "title_word_count"
    CONTENT_UNIQUE_WORDS_COUNT = "content_unique_words_count"

    # Stylistic ratio features
    CONTENT_WORDS_PER_SENTENCE = "content_words_per_sentence"
    CONTENT_STOPWORDS_RATIO = "content_stopwords_ratio"
    CONTENT_UNIQUE_WORDS_RATIO = "content_unique_words_ratio"
    CONTENT_AVERAGE_SENTENCE_LENGTH = "content_average_sentence_length"
    CONTENT_AVERAGE_WORD_LENGTH = "content_average_word_length"

    # Pos tagging features
    CONTENT_NOUN_RATIO = "content_noun_ratio"
    CONTENT_PROPER_NOUN_RATIO = "content_proper_noun_ratio"
    CONTENT_CARDINAL_NUMBER_RATIO = "content_cardinal_number_ratio"
    CONTENT_DETERMINER_RATIO = "content_determiner_ratio"
    CONTENT_ADPOSITION_RATIO = "content_adposition_ratio"
    CONTENT_INTERJECTION_RATIO = "content_interjection_ratio"
    CONTENT_SYMBOL_RATIO = "content_symbol_ratio"

    # Adjective
    CONTENT_ADJECTIVE_RATIO = "content_adjective_ratio"
    CONTENT_WH_DETERMINER_RATIO = "content_wh_determiner_ratio"

    # Verb
    CONTENT_VERB_RATIO = "content_verb_ratio"
    CONTENT_VERB_PRESENT_PARTICIPLE_RATIO = "content_verb_present_participle_ratio"
    CONTENT_VERB_PAST_PARTICIPLE_RATIO = "content_verb_past_participle_ratio"
    CONTENT_VERB_THIRD_PERSON_RATIO = "content_verb_third_person_ratio"
    CONTENT_VERB_MODAL_RATIO = "content_modal_ratio"

    # Adverbs
    CONTENT_ADVERB_RATIO = "content_adverb_ratio"
    CONTENT_COMPARATIVE_ADVERB_RATIO = "content_comparative_adverb_ratio"
    CONTENT_SUPERLATIVE_ADVERB_RATIO = "content_superlative_adverb_ratio"
    CONTENT_EXISTENTIAL_RATIO = "content_existential_ratio"

    # Pronoun
    CONTENT_PRONOUN_RATIO = "content_pronoun_ratio"
    CONTENT_PERSONAL_PRONOUN_RATIO = "content_personal_pronoun_ratio"
    CONTENT_POSSESSIVE_PRONOUN_RATIO = "content_possessive_pronoun_ratio"

    # Punctuation features
    CONTENT_PUNCTUATION_RATIO = "content_punctuation_ratio"
    CONTENT_PERIOD_RATIO = "content_period_ratio"
    CONTENT_QUESTION_MARK_RATIO = "content_question_mark_ration"
    CONTENT_EXCLAMATION_POINT_RATIO = "content_exclamation_point_ratio"
    CONTENT_COMMA_RATIO = "content_comma_ratio"
    CONTENT_COLON_RATIO = "content_colon_ratio"
    CONTENT_SEMICOLON_RATIO = "content_semicolon_ratio"
    CONTENT_PARENTHESES_OPEN_RATIO = "content_parentheses_open_ratio"
    CONTENT_PARENTHESES_CLOSE_RATIO = "content_parentheses_close_ratio"
    CONTENT_QUOTATION_MARK_RATIO = "content_quotation_mark_ratio"

    # Title punctuation
    TITLE_PUNCTUATION_RATIO = "content_punctuation_ratio"

    # Readability features
    CONTENT_GUNNING_FOG_INDEX = "content_gunning_fog_index"
    CONTENT_AUTOMATED_READABILITY_INDEX = "content_automated_readability_index"
    CONTENT_FLESCH_KINCAID_INDEX = "content_flesch_kincaid_index"

    # Calculated features
    CONTENT_TITLE_BODY_SIMILARITY = "content_title_body_similarity"

    @classmethod
    def get_all(cls):
        all_features = []
        all_features.extend(cls.get_stylistic_count_features())
        all_features.extend(cls.get_stylistic_ratio_features())
        all_features.extend(cls.get_readability_features())
        return all_features

    @classmethod
    def get_stylistic_count_features(cls):
        return [
            cls.CONTENT_WORD_COUNT,
            cls.TITLE_WORD_COUNT,
            cls.CONTENT_UNIQUE_WORDS_COUNT,
        ]

    @classmethod
    def get_stylistic_ratio_features(cls):
        return [
            cls.CONTENT_WORDS_PER_SENTENCE,
            cls.CONTENT_STOPWORDS_RATIO,
            cls.CONTENT_UNIQUE_WORDS_RATIO,
            cls.CONTENT_AVERAGE_SENTENCE_LENGTH,
            cls.CONTENT_AVERAGE_WORD_LENGTH,
        ]

    @classmethod
    def get_pos_tagging_features(cls):
        return [
            cls.CONTENT_NOUN_RATIO,
            cls.CONTENT_PROPER_NOUN_RATIO,
            cls.CONTENT_CARDINAL_NUMBER_RATIO,
            cls.CONTENT_DETERMINER_RATIO,
            cls.CONTENT_ADPOSITION_RATIO,
            cls.CONTENT_INTERJECTION_RATIO,
            cls.CONTENT_SYMBOL_RATIO,
            cls.CONTENT_ADJECTIVE_RATIO,
            cls.CONTENT_WH_DETERMINER_RATIO,
            cls.CONTENT_VERB_RATIO,
            cls.CONTENT_VERB_PRESENT_PARTICIPLE_RATIO,
            cls.CONTENT_VERB_PAST_PARTICIPLE_RATIO,
            cls.CONTENT_VERB_THIRD_PERSON_RATIO,
            cls.CONTENT_VERB_MODAL_RATIO,
            cls.CONTENT_ADVERB_RATIO,
            cls.CONTENT_COMPARATIVE_ADVERB_RATIO,
            cls.CONTENT_SUPERLATIVE_ADVERB_RATIO,
            cls.CONTENT_EXISTENTIAL_RATIO,
            cls.CONTENT_PRONOUN_RATIO,
            cls.CONTENT_PERSONAL_PRONOUN_RATIO,
            cls.CONTENT_POSSESSIVE_PRONOUN_RATIO,
        ]

    @classmethod
    def get_punctuation_features(cls):
        return [
            cls.CONTENT_PUNCTUATION_RATIO,
            cls.CONTENT_PERIOD_RATIO,
            cls.CONTENT_QUESTION_MARK_RATIO,
            cls.CONTENT_EXCLAMATION_POINT_RATIO,
            cls.CONTENT_COMMA_RATIO,
            cls.CONTENT_COLON_RATIO,
            cls.CONTENT_SEMICOLON_RATIO,
            cls.CONTENT_PARENTHESES_OPEN_RATIO,
            cls.CONTENT_PARENTHESES_CLOSE_RATIO,
            cls.CONTENT_QUOTATION_MARK_RATIO,
            cls.TITLE_PUNCTUATION_RATIO,
        ]

    @classmethod
    def get_readability_features(cls):
        return [
            cls.CONTENT_GUNNING_FOG_INDEX,
            cls.CONTENT_AUTOMATED_READABILITY_INDEX,
            cls.CONTENT_FLESCH_KINCAID_INDEX,
        ]

    @classmethod
    def get_calculated_features(cls):
        return [
            cls.CONTENT_TITLE_BODY_SIMILARITY,
        ]

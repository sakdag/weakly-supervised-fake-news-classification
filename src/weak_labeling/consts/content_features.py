class ContentFeatures:
    # Stylistic count features
    CONTENT_WORD_COUNT = "content_word_count"
    TITLE_WORD_COUNT = "title_word_count"

    # Stylistic ratio features
    CONTENT_WORDS_PER_SENTENCE = "content_words_per_sentence"
    CONTENT_STOPWORDS_PER_SENTENCE = "content_stopwords_per_sentence"
    CONTENT_UNIQUE_WORDS_RATIO = "content_unique_words_ratio"
    CONTENT_UNIQUE_WORDS_COUNT = "content_unique_words_count"
    CONTENT_AVERAGE_SENTENCE_LENGTH = "content_average_sentence_length"

    # Readability features
    CONTENT_GUNNING_FOG_INDEX = "content_gunning_fog_index"
    CONTENT_AUTOMATED_READABILITY_INDEX = "content_automated_readability_index"
    CONTENT_FLESH_KINCAID_INDEX = "content_flesh_kincaid_index"

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
        ]

    @classmethod
    def get_stylistic_ratio_features(cls):
        return [
            cls.CONTENT_WORDS_PER_SENTENCE,
            cls.CONTENT_STOPWORDS_PER_SENTENCE,
            cls.CONTENT_UNIQUE_WORDS_RATIO,
            cls.CONTENT_UNIQUE_WORDS_COUNT,
            cls.CONTENT_AVERAGE_SENTENCE_LENGTH,
        ]

    @classmethod
    def get_readability_features(cls):
        return [
            cls.CONTENT_GUNNING_FOG_INDEX,
            cls.CONTENT_AUTOMATED_READABILITY_INDEX,
            cls.CONTENT_FLESH_KINCAID_INDEX,
        ]

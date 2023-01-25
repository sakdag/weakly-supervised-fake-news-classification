class ContentFeatures:
    CONTENT_WORD_COUNT = "content_word_count"
    CONTENT_WORDS_PER_SENTENCE = "content_words_per_sentence"
    CONTENT_STOPWORDS_PER_SENTENCE = "content_stopwords_per_sentence"
    CONTENT_UNIQUE_WORDS_RATIO = "content_unique_words_ratio"
    CONTENT_UNIQUE_WORDS_COUNT = "content_unique_words_count"
    CONTENT_AVERAGE_SENTENCE_LENGTH = "content_average_sentence_length"
    TITLE_WORD_COUNT = "title_word_count"

    @staticmethod
    def get_all():
        all_features = []
        for var in vars(ContentFeatures):
            if "__" not in var and var != "get_all":
                all_features.append(getattr(ContentFeatures, var))
        return all_features

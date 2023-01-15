class ContentFeatures:
    CONTENT_WORDS_PER_SENTENCE = "content_words_per_sentence"
    CONTENT_STOPWORDS_PER_SENTENCE = "content_stopwords_per_sentence"

    @staticmethod
    def get_all():
        all_features = []
        for var in vars(ContentFeatures):
            if "__" not in var and var != "get_all":
                all_features.append(getattr(ContentFeatures, var))
        return all_features

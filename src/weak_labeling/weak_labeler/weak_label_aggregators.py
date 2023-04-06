class WeakLabelAggregators:
    SNORKEL_MAJORITY_VOTE = "SNORKEL_MAJORITY_VOTE"
    SNORKEL_LABEL_MODEL = "SNORKEL_LABEL_MODEL"
    HYPER_LABEL_MODEL = "HYPER_LABEL_MODEL"

    @classmethod
    def get_active_aggregators(cls):
        return [
            cls.SNORKEL_MAJORITY_VOTE,
            cls.SNORKEL_LABEL_MODEL,
            cls.HYPER_LABEL_MODEL,
        ]

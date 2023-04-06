class FeatureColumns:
    CONTENT = "content"
    TITLE = "title"

    @staticmethod
    def get_all():
        all_features = []
        for var in vars(FeatureColumns):
            if "__" not in var and var != "get_all":
                all_features.append(getattr(FeatureColumns, var))
        return all_features


class ModelFeatures:
    XGBOOST = "xgboost"
    ADABOOST = "adaboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    NAIVE_BAYES = "naive_bayes"

    @classmethod
    def get_active_features(cls):
        return [
            cls.XGBOOST,
            cls.ADABOOST,
            cls.LOGISTIC_REGRESSION,
            cls.RANDOM_FOREST,
            cls.NAIVE_BAYES,
        ]

    @staticmethod
    def get_all():
        all_features = []
        for var in vars(ModelFeatures):
            if "__" not in var and var != "get_all":
                all_features.append(getattr(ModelFeatures, var))
        return all_features

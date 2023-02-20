class ModelFeatures:
    XGBOOST = "xgboost"
    ADABOOST = "adaboost"
    LOGISTIC_REGRESSION = "logistic_regression"

    @staticmethod
    def get_all():
        all_features = []
        for var in vars(ModelFeatures):
            if "__" not in var and var != "get_all":
                all_features.append(getattr(ModelFeatures, var))
        return all_features

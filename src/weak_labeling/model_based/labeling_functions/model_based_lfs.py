from snorkel.labeling import LabelingFunction

from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns


def make_model_lf(model_feature, column):
    def model_lf(x):
        return x[f"model_{model_feature}_{column}_label"]

    return LabelingFunction(
        name=f"lf_model_{model_feature}_{column}",
        f=model_lf,
    )


def get_model_based_lfs():
    lfs = []
    for model_feature in ModelFeatures.get_active_features():
        for column in FeatureColumns.get_all():
            lfs.append(make_model_lf(model_feature, column))
    return lfs

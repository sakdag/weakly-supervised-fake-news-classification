from snorkel.labeling import LabelingFunction

from src.weak_labeling.consts.model_features import ModelFeatures


def make_model_lf(model_feature):
    def model_lf(x):
        return x[f"model_{model_feature}_label"]

    return LabelingFunction(
        name=f"lf_model_{model_feature}",
        f=model_lf,
    )


def get_model_based_lfs():
    lfs = []
    for model_feature in ModelFeatures.get_all():
        lfs.append(make_model_lf(model_feature))
    return lfs

from src.weak_labeling.model_based.xgboost.xgboost_lf import lf_xgboost_classifier


def get_model_based_lfs():
    return [lf_xgboost_classifier]

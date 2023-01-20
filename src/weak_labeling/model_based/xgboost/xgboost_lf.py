import datetime

import pandas as pd
from snorkel.labeling import labeling_function

from src.weak_labeling.model_based.xgboost.xgboost_preprocess import (
    preprocess_for_xgboost_test,
)
from src.weak_labeling.model_based.xgboost.xgboost_train import load_xgb

XGB = load_xgb(date=datetime.date(2023, 1, 13))


@labeling_function()
def lf_xgboost_classifier(x):
    test_data = pd.Series([x.content])
    features = preprocess_for_xgboost_test(test_data)
    xgb_pred = XGB.predict(features)
    return xgb_pred[0]

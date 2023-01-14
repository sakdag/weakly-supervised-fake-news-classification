import datetime

import pandas as pd
from keras_preprocessing import sequence
from snorkel.labeling import labeling_function

from src.classifier.xgboost.preprocessing import xgboost_tokenizer, ROBERTA_TOKENIZER
from src.classifier.xgboost.xgboost_classifier import load_xgb

XGB = load_xgb(date=datetime.date(2023, 1, 13))


@labeling_function()
def lf_xgboost_classifier(x):
    test_data = pd.Series([x.content])
    features = xgboost_tokenizer(ROBERTA_TOKENIZER, test_data)
    padded_features = sequence.pad_sequences(features, maxlen=500)
    xgb_pred = XGB.predict(padded_features)
    return xgb_pred[0]

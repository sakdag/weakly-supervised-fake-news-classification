import os
import pickle
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.classifier.xgboost.preprocessing import preprocess_for_xgboost_train
from src.preprocessing.consts import RawDatasets
from src.preprocessing.read_raw_data import read_raw_data
from src.utils.path_utils import get_project_root


def train_xgb(df: pd.DataFrame, use_saved=True):
    roberta_train, roberta_test, y_test, y_train = preprocess_for_xgboost_train(df)

    # warnings.filterwarnings(action='ignore', category=UserWarning)
    print(f"Use saved XGB model: {use_saved}")
    xgb_file_name = os.path.join(
        get_project_root(),
        f"model/xgboost/xgb-nela-covid-{str(datetime.now().date())}.pkl",
    )
    if use_saved:
        xgb = pickle.load(open(xgb_file_name, "rb"))
    else:
        xgb = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.15,
            max_depth=9,
            eval_metric="auc",
            use_label_encoder=False,
            objective="binary:logistic",
        )
        xgb.fit(roberta_train, y_train)
        pickle.dump(xgb, open(xgb_file_name, "wb"))

    xgb_pred = xgb.predict(roberta_test)
    xgb_score = accuracy_score(y_test, xgb_pred)
    xgb_roc = roc_auc_score(y_test, xgb_pred)
    print("The accuracy of XGBOOST is: %0.2f" % xgb_score)
    print("The roc_auc score of XGBOOST is: %0.2f" % xgb_roc)
    cr = classification_report(y_test, xgb_pred)
    print(cr)


def load_xgb(date: datetime.date):
    xgb_file_name = os.path.join(
        get_project_root(), f"model/xgboost/xgb-nela-covid-{str(date)}.pkl"
    )
    return pickle.load(open(xgb_file_name, "rb"))


if __name__ == "__main__":
    covid_df: pd.DataFrame = read_raw_data(RawDatasets.COVID_DATASET_SAMPLED)
    train_xgb(covid_df, use_saved=False)

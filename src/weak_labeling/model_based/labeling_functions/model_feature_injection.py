import datetime

import pandas as pd

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    save_to_csv,
    read_csv,
)
from src.weak_labeling.model_based.adaboost.adaboost import (
    train_adaboost_classifier,
    get_adaboost_labels,
)
from src.weak_labeling.model_based.xgboost.xgboost_preprocess import (
    preprocess_for_xgboost_test,
)
from src.weak_labeling.model_based.xgboost.xgboost_train import load_xgb


def inject_xgboost_label(df_to_inject: pd.DataFrame) -> pd.DataFrame:
    xgb = load_xgb(date=datetime.date(2023, 1, 22))
    test_data = df_to_inject.content
    features = preprocess_for_xgboost_test(test_data)
    xgb_pred = xgb.predict(features)
    df_to_inject["model_xgboost_label"] = xgb_pred
    return df_to_inject


def inject_adaboost_label(df_to_inject: pd.DataFrame) -> pd.DataFrame:
    adaboost_classifier, tfidf_vectorizer = train_adaboost_classifier(
        train_with=Datasets.NELA_COVID_2021
    )
    adb_pred = get_adaboost_labels(adaboost_classifier, tfidf_vectorizer, df_to_inject)
    df_to_inject["model_adaboost_label"] = adb_pred
    return df_to_inject


def inject_model_features(df_to_inject: pd.DataFrame) -> pd.DataFrame:
    df_to_inject = inject_xgboost_label(df_to_inject)
    df_to_inject = inject_adaboost_label(df_to_inject)
    return df_to_inject


def run_model_feature_injection(dataset: Dataset, zero: bool = False):
    if zero:
        df = read_csv(dataset.sampled_path)
    else:
        df = read_csv(dataset.injected_path)
    print(df.head())

    injected_df = inject_model_features(df)
    print(injected_df.head())

    save_to_csv(injected_df, dataset.injected_path)


if __name__ == "__main__":
    run_model_feature_injection(
        dataset=Datasets.NELA_ELECTIONS_2020, zero=False
    )
    run_model_feature_injection(
        dataset=Datasets.NELA_COVID_2021, zero=False
    )
    run_model_feature_injection(
        dataset=Datasets.NELA_COVID_2020, zero=False
    )
    run_model_feature_injection(
        dataset=Datasets.NELA_CAPITOL_2021, zero=False
    )

import datetime

import pandas as pd

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    save_to_csv,
    read_csv,
)
from src.weak_labeling.consts.model_features import ModelFeatures
from src.weak_labeling.model_based.adaboost.adaboost import (
    get_adaboost_labels,
)
from src.weak_labeling.model_based.logistic_regression.logistic_regression import (
    get_logistic_regression_labels,
)
from src.weak_labeling.model_based.utils import load_model
from src.weak_labeling.model_based.xgboost.xgboost_model import (
    preprocess_for_xgboost_test,
)


def inject_xgboost_label(
    df_to_inject: pd.DataFrame, train_dataset: Dataset, identifier: str
) -> pd.DataFrame:
    xgb = load_model(
        model_feature=ModelFeatures.XGBOOST,
        dataset=train_dataset,
        identifier=identifier,
    )
    test_data = df_to_inject.content
    features = preprocess_for_xgboost_test(test_data)
    xgb_pred = xgb.predict(features)
    df_to_inject["model_xgboost_label"] = xgb_pred
    return df_to_inject


def inject_adaboost_label(
    df_to_inject: pd.DataFrame, train_dataset: Dataset, identifier: str
) -> pd.DataFrame:
    adaboost_classifier = load_model(
        model_feature=ModelFeatures.ADABOOST,
        dataset=train_dataset,
        identifier=identifier,
    )
    tfidf_vectorizer = load_model(
        model_feature="tfidf_vectorizer",
        dataset=train_dataset,
        identifier=f"adaboost-{identifier}",
    )
    adb_pred = get_adaboost_labels(adaboost_classifier, tfidf_vectorizer, df_to_inject)
    df_to_inject["model_adaboost_label"] = adb_pred
    return df_to_inject


def inject_logistic_regression_label(
    df_to_inject: pd.DataFrame, train_dataset: Dataset, identifier: str
) -> pd.DataFrame:
    lr_classifier = load_model(
        model_feature=ModelFeatures.LOGISTIC_REGRESSION,
        dataset=train_dataset,
        identifier=identifier,
    )
    tfidf_vectorizer = load_model(
        model_feature="tfidf_vectorizer",
        dataset=train_dataset,
        identifier=f"logistic_regression-{identifier}",
    )
    adb_pred = get_logistic_regression_labels(
        lr_classifier, tfidf_vectorizer, df_to_inject
    )
    df_to_inject["model_logistic_regression_label"] = adb_pred
    return df_to_inject


def inject_model_features(
    df_to_inject: pd.DataFrame, train_dataset: Dataset, identifier: str
) -> pd.DataFrame:
    # df_to_inject = inject_xgboost_label(
    #     df_to_inject, train_dataset=train_dataset, identifier=identifier
    # )
    # df_to_inject = inject_adaboost_label(
    #     df_to_inject, train_dataset=train_dataset, identifier=identifier
    # )
    df_to_inject = inject_logistic_regression_label(
        df_to_inject, train_dataset=train_dataset, identifier=identifier
    )
    return df_to_inject


def run_model_feature_injection(
    dataset: Dataset, train_dataset: Dataset, identifier: str, zero: bool = False
):
    if zero:
        df = read_csv(dataset.sampled_path)
    else:
        df = read_csv(dataset.injected_path)
    print(df.head())

    injected_df = inject_model_features(
        df, train_dataset=train_dataset, identifier=identifier
    )
    print(injected_df.head())

    save_to_csv(injected_df, dataset.injected_path)


if __name__ == "__main__":
    datasets = [
        Datasets.NELA_ELECTIONS_2020,
        # Datasets.NELA_COVID_2021,
        # Datasets.NELA_COVID_2020,
        # Datasets.NELA_CAPITOL_2021,
    ]
    for dataset in datasets:
        run_model_feature_injection(
            dataset=dataset,
            train_dataset=Datasets.NELA_COVID_2021,
            identifier="test1",
            zero=False,
        )

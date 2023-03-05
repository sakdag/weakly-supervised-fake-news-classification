import pandas as pd

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    save_to_csv,
    read_csv,
)
from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns
from src.weak_labeling.model_based.adaboost.adaboost import (
    get_adaboost_labels,
)
from src.weak_labeling.model_based.logistic_regression.logistic_regression import (
    get_logistic_regression_labels,
)
from src.weak_labeling.model_based.naive_bayes.naive_bayes import get_naive_bayes_labels
from src.weak_labeling.model_based.random_forest.random_forest import get_random_forest_labels
from src.weak_labeling.model_based.utils import load_model
from src.weak_labeling.model_based.xgboost.xgboost_model import (
    get_xgboost_labels,
)


def inject_model_feature(
    df_to_inject: pd.DataFrame,
    train_dataset: Dataset,
    model_feature: str = ModelFeatures.XGBOOST,
    feature_column: str = FeatureColumns.CONTENT,
    identifier: str = "default",
) -> pd.DataFrame:
    classifier = load_model(
        model_feature=model_feature,
        feature_column=feature_column,
        dataset=train_dataset,
        identifier=identifier,
    )
    predictions = None
    if model_feature.startswith("xgboost"):
        predictions = get_xgboost_labels(
            xgboost_classifier=classifier,
            df=df_to_inject,
            feature_column=feature_column
        )
    elif model_feature.startswith("adaboost"):
        predictions = get_adaboost_labels(
            adaboost_classifier=classifier,
            df=df_to_inject,
            feature_column=feature_column
        )
    elif model_feature.startswith("logistic_regression"):
        predictions = get_logistic_regression_labels(
            lr_classifier=classifier,
            df=df_to_inject,
            feature_column=feature_column
        )
    elif model_feature.startswith("random_forest"):
        predictions = get_random_forest_labels(
            random_forest_classifier=classifier,
            df=df_to_inject,
            feature_column=feature_column
        )
    elif model_feature.startswith("naive_bayes"):
        predictions = get_naive_bayes_labels(
            naive_bayes_classifier=classifier,
            df=df_to_inject,
            feature_column=feature_column
        )
    else:
        print(f"Unknown model_feature is provided: {model_feature}")

    if predictions is not None:
        df_to_inject[f"model_{model_feature}_{feature_column}_label"] = predictions
    return df_to_inject


def inject_model_features(
    df_to_inject: pd.DataFrame, train_dataset: Dataset, identifier: str
) -> pd.DataFrame:
    for feature in ModelFeatures.get_active_features():
        for column in FeatureColumns.get_all():
            print(f"Inject model_feature train_dataset: {train_dataset.name}, feature: {feature}, feature_column: {column}")
            df_to_inject = inject_model_feature(
                df_to_inject=df_to_inject,
                train_dataset=train_dataset,
                model_feature=feature,
                feature_column=column,
                identifier=identifier
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

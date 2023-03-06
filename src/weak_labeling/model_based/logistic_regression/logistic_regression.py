import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_csv
from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns
from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
from src.weak_labeling.model_based.utils import get_model_file_name
from sklearn.model_selection import train_test_split


def train_logistic_regression_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)

    x_train, x_test, y_train, y_test = train_test_split(df, df["label"], test_size=0.20)

    train_vectors = get_tfidf_vectors(x_train, feature_column)
    test_vectors = get_tfidf_vectors(x_test, feature_column)

    lr_classifier = LogisticRegression()
    lr_classifier.fit(train_vectors, y_train)
    lr_pred = lr_classifier.predict(test_vectors)
    cr = classification_report(y_test, lr_pred)
    print(cr)

    logistic_regression_file_name = get_model_file_name(
        ModelFeatures.LOGISTIC_REGRESSION, feature_column, dataset, save_identifier
    )
    pickle.dump(lr_classifier, open(logistic_regression_file_name, "wb"))


def get_logistic_regression_labels(
    lr_classifier,
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
) -> pd.Series:
    test_df = df[["title", "content", "label"]].copy()
    test_vectors = get_tfidf_vectors(test_df, feature_column)

    y_test_pred = lr_classifier.predict(test_vectors)
    cr = classification_report(test_df["label"], y_test_pred)
    print(cr)
    return y_test_pred


if __name__ == "__main__":
    datasets = [
        Datasets.NELA_COVID_2021,
        Datasets.NELA_ELECTIONS_SUBSET_2020,
    ]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_logistic_regression_model(
                dataset, save_identifier="test1", feature_column=column
            )

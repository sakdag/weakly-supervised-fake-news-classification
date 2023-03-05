import pickle
from typing import Tuple

import pandas as pd
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_raw_data, read_csv
from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns
from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
from src.weak_labeling.model_based.utils import get_model_file_name

stop = stopwords.words("english")


def train_random_forest_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
    optimize: bool = False,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df, df["label"], test_size=0.20
    )

    train_vectors = get_tfidf_vectors(x_train, feature_column)
    test_vectors = get_tfidf_vectors(x_test, feature_column)

    if optimize:
        params = get_best_random_forest_parameters(train_vectors, y_train)
    else:
        params = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 1,
            "min_samples_split": 5,
            "max_features": "sqrt"
        }

    dts = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_split=params["min_samples_split"],
        max_features=params["max_features"],
        random_state=42
    )
    dts.fit(train_vectors, y_train)
    y_pred = dts.predict(test_vectors)
    cr = classification_report(y_test, y_pred)
    print(cr)

    file_name = get_model_file_name(
        ModelFeatures.RANDOM_FOREST, feature_column, dataset, save_identifier
    )
    pickle.dump(dts, open(file_name, "wb"))


def get_best_random_forest_parameters(
    tfidf_train,
    y_train,
) -> dict:
    rfc = RandomForestClassifier()

    parameters = {
        "n_estimators": [90, 100, 115, 130],
        "max_depth": range(2, 20),
        "min_samples_leaf": range(1, 10),
        "min_samples_split": range(2, 10),
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(rfc, parameters, verbose=3, scoring="f1", n_jobs=-1)
    grid_search.fit(tfidf_train, y_train)
    best_params = grid_search.best_params_

    print(f"Best params: {grid_search.best_params_}")
    return best_params


def get_random_forest_labels(
    random_forest_classifier,
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
) -> pd.Series:
    test_df = df[["title", "content", "label"]].copy()
    test_vectors = get_tfidf_vectors(test_df, feature_column)

    y_test_pred = random_forest_classifier.predict(test_vectors)
    cr = classification_report(test_df["label"], y_test_pred)
    print(cr)
    return y_test_pred


if __name__ == "__main__":
    datasets = [
        Datasets.NELA_COVID_2021,
        # Datasets.NELA_ELECTIONS_SUBSET_2020,
    ]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_random_forest_model(
                dataset,
                save_identifier="test1",
                feature_column=column
            )

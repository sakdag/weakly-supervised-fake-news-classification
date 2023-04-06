import pickle

import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_csv, balanced_undersample
from src.utils.time_utils import report_time
from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns
from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
from src.weak_labeling.model_based.utils import get_model_file_name

stop = stopwords.words("english")


def train_adaboost_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
    undersample: bool = False,
    optimize: bool = False,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)

    if undersample:
        df = balanced_undersample(df, min(df["label"].value_counts()) * 2)
        print(f"Undersampled, dataframe size: {len(df)}")

    x_train, x_test, y_train, y_test = train_test_split(df, df["label"], test_size=0.20)

    train_vectors = get_tfidf_vectors(x_train, feature_column)
    test_vectors = get_tfidf_vectors(x_test, feature_column)

    if optimize:
        params = get_best_adaboost_parameters(train_vectors, y_train)
    else:
        params = {
            "base_estimator__max_depth": 10,
            "base_estimator__min_samples_leaf": 1,
            "learning_rate": 1.0,
            "n_estimators": 5,
        }

    abc = AdaBoostClassifier(
        DecisionTreeClassifier(
            max_depth=params["base_estimator__max_depth"],
            min_samples_leaf=params["base_estimator__min_samples_leaf"],
        ),
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        random_state=1,
    )
    abc.fit(train_vectors, y_train)
    y_pred = abc.predict(test_vectors)
    cr = classification_report(y_test, y_pred)
    print(cr)

    adaboost_file_name = get_model_file_name(
        ModelFeatures.ADABOOST, feature_column, dataset, save_identifier
    )
    pickle.dump(abc, open(adaboost_file_name, "wb"))


def get_best_adaboost_parameters(
    tfidf_train,
    y_train,
) -> dict:
    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

    parameters = {
        "base_estimator__max_depth": [i for i in range(2, 11, 2)],
        "base_estimator__min_samples_leaf": [5, 10],
        "n_estimators": [10, 50, 250, 1000],
        "learning_rate": [0.01, 0.1],
    }

    grid_search = GridSearchCV(abc, parameters, verbose=3, scoring="f1", n_jobs=-1)
    grid_search.fit(tfidf_train, y_train)
    best_params = grid_search.best_params_

    print(f"Best params: {grid_search.best_params_}")
    return best_params


def get_adaboost_labels(
    adaboost_classifier,
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
) -> pd.Series:
    test_df = df[["title", "content", "label"]].copy()
    test_vectors = get_tfidf_vectors(test_df, feature_column)

    y_test_pred = adaboost_classifier.predict(test_vectors)
    cr = classification_report(test_df["label"], y_test_pred)
    print(cr)
    return y_test_pred


@report_time
def run():
    datasets = [
        # Datasets.NELA_ELECTIONS_2020,
        # Datasets.NELA_COVID_2021,
        Datasets.NELA_ELECTIONS_SUBSET_2020,
        Datasets.NELA_COVID_COMBINED,
    ]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_adaboost_model(
                dataset,
                save_identifier="final",
                feature_column=column,
                undersample=True,
                optimize=False,
            )


if __name__ == "__main__":
    run()

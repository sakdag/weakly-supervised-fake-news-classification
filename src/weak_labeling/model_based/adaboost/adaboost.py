import pickle
from typing import Tuple

import pandas as pd
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_raw_data, read_csv
from src.weak_labeling.consts.model_features import ModelFeatures
from src.weak_labeling.model_based.utils import get_model_file_name

stop = stopwords.words("english")


def train_adaboost_model(
    dataset: Dataset,
    save_identifier: str,
    optimize: bool = False,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)

    df["title_without_stopwords"] = df["title"].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )
    df["content_without_stopwords"] = df["content"].apply(
        lambda x: " ".join([word for word in x.split()[:200] if word not in (stop)])
    )

    y = df.label
    df["x_train"] = df["title_without_stopwords"] + df["content_without_stopwords"]
    x = df["x_train"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    tfidf_vect = TfidfVectorizer(stop_words="english")
    tfidf_train = tfidf_vect.fit_transform(x_train)
    tfidf_test = tfidf_vect.transform(x_test)

    # Save vectorizer
    tfidf_vect_file_name = get_model_file_name(
        "tfidf_vectorizer", dataset, save_identifier
    )
    pickle.dump(tfidf_vect, open(tfidf_vect_file_name, "wb"))

    if optimize:
        params = get_best_adaboost_parameters(tfidf_train, y_train)
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
    abc.fit(tfidf_train, y_train)
    y_pred = abc.predict(tfidf_test)
    abc_score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy: %0.3f" % abc_score)

    adaboost_file_name = get_model_file_name(
        ModelFeatures.ADABOOST, dataset, save_identifier
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
    adaboost_classifier, tfidf_vectorizer, df: pd.DataFrame
) -> pd.Series:
    test_df = df[["title", "content", "label"]].copy()
    test_df["title_without_stopwords"] = test_df["title"].apply(
        lambda x: " ".join([word for word in x.split() if word not in (stop)])
    )
    test_df["content_without_stopwords"] = test_df["content"].apply(
        lambda x: " ".join([word for word in x.split()[:200] if word not in (stop)])
    )

    y_test = test_df.label
    test_df["x_train"] = (
        test_df["title_without_stopwords"] + test_df["content_without_stopwords"]
    )
    x_test = test_df["x_train"]

    tfidf_test = tfidf_vectorizer.transform(x_test)

    y_test_pred = adaboost_classifier.predict(tfidf_test)
    adab_score = metrics.accuracy_score(y_test, y_test_pred)

    print("accuracy: %0.3f" % adab_score)
    return y_test_pred


if __name__ == "__main__":
    train_adaboost_model(
        Datasets.NELA_COVID_2021, save_identifier="test1", optimize=False
    )

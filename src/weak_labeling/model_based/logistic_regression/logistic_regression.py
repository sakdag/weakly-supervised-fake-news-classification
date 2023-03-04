import pickle

import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_csv, balanced_undersample
from src.weak_labeling.consts.model_features import ModelFeatures
from src.weak_labeling.model_based.utils import get_model_file_name
from sklearn.model_selection import train_test_split


def train_logistic_regression_model(
    dataset: Dataset,
    save: bool,
    save_identifier: str,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)

    df = balanced_undersample(df, n=2000)

    sentences = df["content"].values
    y = df["label"].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000
    )

    tfidf_vect = TfidfVectorizer(stop_words="english")
    tfidf_train = tfidf_vect.fit_transform(sentences_train)
    tfidf_test = tfidf_vect.transform(sentences_test)

    # Save vectorizer
    if save:
        tfidf_vect_file_name = get_model_file_name(
            "tfidf_vectorizer", dataset, f"logistic_regression-{save_identifier}"
        )
        pickle.dump(tfidf_vect, open(tfidf_vect_file_name, "wb"))

    lr_classifier = LogisticRegression()
    lr_classifier.fit(tfidf_train, y_train)
    score = lr_classifier.score(tfidf_test, y_test)

    print("Accuracy:", score)

    if save:
        logistic_regression_file_name = get_model_file_name(
            ModelFeatures.LOGISTIC_REGRESSION, dataset, save_identifier
        )
        pickle.dump(lr_classifier, open(logistic_regression_file_name, "wb"))


def get_logistic_regression_labels(
    lr_classifier, tfidf_vectorizer, df: pd.DataFrame
) -> pd.Series:
    sentences = df["content"].values
    y_test_actual = df["label"].values

    tfidf_test = tfidf_vectorizer.transform(sentences)

    y_test_pred = lr_classifier.predict(tfidf_test)
    lr_score = metrics.accuracy_score(y_test_actual, y_test_pred)

    print("accuracy: %0.3f" % lr_score)
    return y_test_pred


if __name__ == "__main__":
    # train_logistic_regression_model(
    #     Datasets.NELA_COVID_2021, save=True, save_identifier="test1"
    # )
    train_logistic_regression_model(
        Datasets.NELA_ELECTIONS_SUBSET_2020, save=True, save_identifier="test1"
    )

import pickle

import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import (
    read_raw_data,
    read_csv,
    balanced_undersample,
)
from src.utils.time_utils import report_time
from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns
from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
from src.weak_labeling.model_based.utils import get_model_file_name

stop = stopwords.words("english")


def train_naive_bayes_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
    undersample: bool = False,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)

    if undersample:
        df = balanced_undersample(df, min(df["label"].value_counts()) * 2)
        print(f"Undersampled, dataframe size: {len(df)}")

    x_train, x_test, y_train, y_test = train_test_split(df, df["label"], test_size=0.20)

    train_vectors = get_tfidf_vectors(x_train, feature_column)
    test_vectors = get_tfidf_vectors(x_test, feature_column)

    nb = MultinomialNB()
    nb.fit(train_vectors, y_train)
    y_pred = nb.predict(test_vectors)
    cr = classification_report(y_test, y_pred)
    print(cr)

    file_name = get_model_file_name(
        ModelFeatures.NAIVE_BAYES, feature_column, dataset, save_identifier
    )
    pickle.dump(nb, open(file_name, "wb"))


def get_naive_bayes_labels(
    naive_bayes_classifier,
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
) -> pd.Series:
    test_df = df[["title", "content", "label"]].copy()
    test_vectors = get_tfidf_vectors(test_df, feature_column)

    y_test_pred = naive_bayes_classifier.predict(test_vectors)
    cr = classification_report(test_df["label"], y_test_pred)
    print(cr)
    return y_test_pred


@report_time
def run():
    datasets = [
        # Datasets.NELA_ELECTIONS_2020,
        Datasets.NELA_COVID_SUBSET,
        # Datasets.NELA_ELECTIONS_SUBSET_2020,
        # Datasets.NELA_COVID_COMBINED,
    ]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_naive_bayes_model(
                dataset,
                save_identifier="final",
                feature_column=column,
                undersample=True,
            )


if __name__ == "__main__":
    run()

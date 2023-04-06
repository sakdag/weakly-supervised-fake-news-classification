import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.cleanup import clean_text
from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_csv
from src.weak_labeling.consts.model_features import FeatureColumns
from src.weak_labeling.model_based.utils import get_any_file_name


def save_tfidf_vectorizer(
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
    max_features: int = -1,
):
    tfidf_vectorizer = TfidfVectorizer()
    if max_features > 0:
        tfidf_vectorizer.max_features = max_features
    text_column = df[feature_column]
    text_column_cleaned = text_column.apply(lambda x: clean_text(x))
    tfidf_vectorizer.fit(text_column_cleaned)

    tfidf_vect_file_name = get_any_file_name(
        "tfidf_vectorizer",
        f"generic-{feature_column}{f'-{max_features}' if max_features > 0 else ''}",
    )
    print(f"Saving TFIDF vectorizer to path: {tfidf_vect_file_name}")
    pickle.dump(tfidf_vectorizer, open(tfidf_vect_file_name, "wb"))


def load_tfidf_vectorizer(
    feature_column: str = FeatureColumns.CONTENT, max_features: int = -1
):
    tfidf_vect_file_name = get_any_file_name(
        "tfidf_vectorizer",
        f"generic-{feature_column}{f'-{max_features}' if max_features > 0 else ''}",
    )
    print(f"Loading TFIDF vectorizer from path: {tfidf_vect_file_name}")
    return pickle.load(open(tfidf_vect_file_name, "rb"))


if __name__ == "__main__":
    datasets = [
        Datasets.NELA_ELECTIONS_2020,
        Datasets.NELA_COVID_2021,
        Datasets.NELA_COVID_2020,
        Datasets.NELA_CAPITOL_2021,
        # Datasets.NELA_ELECTIONS_SUBSET_2020,
    ]
    big_df = read_csv(datasets[0].sampled_path)
    for dataset in datasets[1:]:
        big_df = pd.concat([big_df, read_csv(dataset.sampled_path)])
    print(f"Total df size: {len(big_df)}")

    save_tfidf_vectorizer(big_df, "title")
    save_tfidf_vectorizer(big_df, "content")

"""Shared train / predict helpers for classical weak-labeling models."""

from __future__ import annotations

import pickle
from typing import Any, Callable, Optional

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.preprocessing.consts import Dataset
from src.preprocessing.read_raw_data import balanced_undersample, read_csv
from src.weak_labeling.consts.model_features import FeatureColumns
from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
from src.weak_labeling.model_based.utils import get_model_file_name


def train_and_save_model(
    dataset: Dataset,
    save_identifier: str,
    model_feature: str,
    build_model: Callable[[], Any],
    feature_column: str = FeatureColumns.CONTENT,
    undersample: bool = False,
    test_size: float = 0.20,
) -> Any:
    """Train a sklearn-compatible classifier on TF-IDF features and pickle it."""
    df: pd.DataFrame = read_csv(dataset.injected_path)

    if undersample:
        df = balanced_undersample(df, min(df["label"].value_counts()) * 2)
        print(f"Undersampled, dataframe size: {len(df)}")

    x_train, x_test, y_train, y_test = train_test_split(
        df, df["label"], test_size=test_size
    )

    train_vectors = get_tfidf_vectors(x_train, feature_column)
    test_vectors = get_tfidf_vectors(x_test, feature_column)

    model = build_model()
    model.fit(train_vectors, y_train)

    predictions = model.predict(test_vectors)
    print(classification_report(y_test, predictions))

    model_path = get_model_file_name(
        model_feature, feature_column, dataset, save_identifier
    )
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    return model


def predict_labels(
    classifier: Any,
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
    report: bool = True,
) -> pd.Series:
    """Predict labels for a dataframe and optionally print a classification report."""
    test_df = df[["title", "content", "label"]].copy()
    test_vectors = get_tfidf_vectors(test_df, feature_column)
    y_pred = classifier.predict(test_vectors)
    if report:
        print(classification_report(test_df["label"], y_pred))
    return y_pred

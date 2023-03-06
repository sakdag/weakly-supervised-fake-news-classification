import pandas as pd

from src.preprocessing.cleanup import clean_text
from src.preprocessing.tfidf_vectorizer import load_tfidf_vectorizer
from src.weak_labeling.consts.model_features import FeatureColumns


def get_tfidf_vectors(
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
    max_features: int = -1,
):
    text_col = df[feature_column]
    cleaned_text_col = text_col.apply(lambda t: clean_text(t))
    tfidf_vectorizer = load_tfidf_vectorizer(feature_column, max_features)
    return tfidf_vectorizer.transform(cleaned_text_col)

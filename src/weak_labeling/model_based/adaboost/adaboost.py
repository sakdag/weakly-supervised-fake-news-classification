from typing import Tuple

import pandas as pd
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_raw_data

stop = stopwords.words("english")


def train_adaboost_classifier(
    train_with: Datasets = Datasets.NELA_COVID_2021_DATASET_SAMPLED,
) -> Tuple[AdaBoostClassifier, TfidfVectorizer]:
    df = read_raw_data(dataset=train_with)

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

    Adab = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=10), n_estimators=5, random_state=1
    )
    Adab.fit(tfidf_train, y_train)
    y_pred3 = Adab.predict(tfidf_test)
    ABscore = metrics.accuracy_score(y_test, y_pred3)
    print("accuracy: %0.3f" % ABscore)

    return Adab, tfidf_vect


def get_adaboost_labels(
    adab: AdaBoostClassifier, tfidf_vectorizer: TfidfVectorizer, df: pd.DataFrame
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

    y_test_pred = adab.predict(tfidf_test)
    adab_score = metrics.accuracy_score(y_test, y_test_pred)

    print("accuracy: %0.3f" % adab_score)
    df["model_adaboost_label"] = y_test_pred
    return y_test_pred

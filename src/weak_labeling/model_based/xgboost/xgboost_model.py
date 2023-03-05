import pickle

import pandas as pd
import transformers
from keras_preprocessing import sequence
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.preprocessing.cleanup import cleanup_df
from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_csv
from src.weak_labeling.consts.model_features import ModelFeatures, FeatureColumns
from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
from src.weak_labeling.model_based.utils import get_model_file_name
from sklearn.model_selection import GridSearchCV, train_test_split

ROBERTA_TOKENIZER = transformers.RobertaTokenizer.from_pretrained(
    "roberta-base-openai-detector"
)


def xgboost_tokenizer(tokenizer_name, docs):
    features = []
    for doc in docs:
        tokens = tokenizer_name.tokenize(doc)
        ids = tokenizer_name.convert_tokens_to_ids(tokens)
        features.append(ids)
    return features


# Legacy, not used right now
def preprocess_for_xgboost_test(test_data: pd.Series):
    features = xgboost_tokenizer(ROBERTA_TOKENIZER, test_data)
    padded_features = sequence.pad_sequences(features, maxlen=500)
    return padded_features


# Legacy, not used right now
def preprocess_for_xgboost_train(df: pd.DataFrame):
    df_cleaned = cleanup_df(df)
    # df.sample(n=5000).reset_index(drop=True)

    X, y = df_cleaned["content"], df_cleaned["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    roberta_train_features = xgboost_tokenizer(ROBERTA_TOKENIZER, X_train)
    roberta_test_features = xgboost_tokenizer(ROBERTA_TOKENIZER, X_test)

    roberta_train = sequence.pad_sequences(roberta_train_features, maxlen=500)
    roberta_test = sequence.pad_sequences(roberta_test_features, maxlen=500)

    return roberta_train, roberta_test, y_test, y_train


def get_best_xgb_parameters(
    roberta_train,
    y_train,
):
    parameters = {
        "max_depth": range(2, 10, 1),
        "n_estimators": range(60, 1000, 188),
        "learning_rate": [0.15, 0.1, 0.01, 0.05],
    }
    xgb = XGBClassifier(
        eval_metric="auc",
        use_label_encoder=False,
        objective="binary:logistic",
        seed=42,
    )
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=parameters,
        scoring="roc_auc",
        n_jobs=10,
        cv=10,
        verbose=1,
    )
    grid_search.fit(roberta_train, y_train)
    best_params = grid_search.best_params_

    print(f"Best params: {grid_search.best_params_}")
    return best_params


def train_xgb_model(
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

    # warnings.filterwarnings(action='ignore', category=UserWarning)
    if optimize:
        params = get_best_xgb_parameters(train_vectors, y_train)
    else:
        params = {"learning_rate": 0.15, "max_depth": 9, "n_estimators": 1000}

    xgb = XGBClassifier(
        **params,
        eval_metric="auc",
        use_label_encoder=False,
        objective="binary:logistic",
    )
    xgb.fit(train_vectors, y_train)

    xgb_pred = xgb.predict(test_vectors)
    cr = classification_report(y_test, xgb_pred)
    print(cr)

    xgb_file_name = get_model_file_name(ModelFeatures.XGBOOST, feature_column, dataset, save_identifier)
    pickle.dump(xgb, open(xgb_file_name, "wb"))

def get_xgboost_labels(
    xgboost_classifier,
    df: pd.DataFrame,
    feature_column: str = FeatureColumns.CONTENT,
) -> pd.Series:
    test_df = df[["title", "content", "label"]].copy()
    test_vectors = get_tfidf_vectors(test_df, feature_column)

    y_test_pred = xgboost_classifier.predict(test_vectors)
    cr = classification_report(test_df["label"], y_test_pred)
    print(cr)
    return y_test_pred


if __name__ == "__main__":
    datasets = [
        # Datasets.NELA_COVID_2021,
        Datasets.NELA_ELECTIONS_SUBSET_2020,
    ]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_xgb_model(
                dataset,
                save_identifier="test1",
                feature_column=column
            )

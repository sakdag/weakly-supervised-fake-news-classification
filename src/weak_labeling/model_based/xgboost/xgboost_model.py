import pickle

import pandas as pd
import transformers
from keras_preprocessing import sequence
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.preprocessing.cleanup import cleanup_df
from src.preprocessing.consts import Datasets, Dataset
from src.preprocessing.read_raw_data import read_csv
from src.weak_labeling.consts.model_features import ModelFeatures
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


def preprocess_for_xgboost_test(test_data: pd.Series):
    features = xgboost_tokenizer(ROBERTA_TOKENIZER, test_data)
    padded_features = sequence.pad_sequences(features, maxlen=500)
    return padded_features


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
    optimize: bool = False,
):
    df: pd.DataFrame = read_csv(dataset.injected_path)
    roberta_train, roberta_test, y_test, y_train = preprocess_for_xgboost_train(df)

    # warnings.filterwarnings(action='ignore', category=UserWarning)
    xgb_file_name = get_model_file_name(ModelFeatures.XGBOOST, dataset, save_identifier)
    if optimize:
        params = get_best_xgb_parameters(roberta_train, y_train)
    else:
        params = {"learning_rate": 0.15, "max_depth": 9, "n_estimators": 1000}

    xgb = XGBClassifier(
        **params,
        eval_metric="auc",
        use_label_encoder=False,
        objective="binary:logistic",
    )
    xgb.fit(roberta_train, y_train)
    pickle.dump(xgb, open(xgb_file_name, "wb"))

    xgb_pred = xgb.predict(roberta_test)
    xgb_score = accuracy_score(y_test, xgb_pred)
    xgb_roc = roc_auc_score(y_test, xgb_pred)
    print("The accuracy of XGBOOST is: %0.2f" % xgb_score)
    print("The roc_auc score of XGBOOST is: %0.2f" % xgb_roc)
    cr = classification_report(y_test, xgb_pred)
    print(cr)


if __name__ == "__main__":
    # train_xgb_model(Datasets.NELA_COVID_2021, save_identifier="test1", optimize=False)
    train_xgb_model(
        Datasets.NELA_ELECTIONS_SUBSET_2020, save_identifier="test1", optimize=False
    )

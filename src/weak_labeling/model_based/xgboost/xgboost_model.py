from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.preprocessing.consts import Dataset, Datasets
from src.utils.time_utils import report_time
from src.weak_labeling.consts.model_features import FeatureColumns, ModelFeatures
from src.weak_labeling.model_based.common.trainer import (
    predict_labels,
    train_and_save_model,
)


def get_best_xgb_parameters(train_vectors, y_train):
    parameters = {
        "max_depth": range(2, 10, 1),
        "n_estimators": [60, 150, 250, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
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
    grid_search.fit(train_vectors, y_train)
    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_params_


def train_xgb_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
    undersample: bool = False,
    optimize: bool = False,
):
    if optimize:
        from sklearn.model_selection import train_test_split

        from src.preprocessing.read_raw_data import balanced_undersample, read_csv
        from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors

        df = read_csv(dataset.injected_path)
        if undersample:
            df = balanced_undersample(df, min(df["label"].value_counts()) * 2)
            print(f"Undersampled, dataframe size: {len(df)}")
        x_train, _, y_train, _ = train_test_split(df, df["label"], test_size=0.20)
        train_vectors = get_tfidf_vectors(x_train, feature_column)
        params = get_best_xgb_parameters(train_vectors, y_train)
    else:
        params = {"learning_rate": 0.15, "max_depth": 9, "n_estimators": 1000}

    def build_model():
        return XGBClassifier(
            **params,
            eval_metric="auc",
            use_label_encoder=False,
            objective="binary:logistic",
        )

    return train_and_save_model(
        dataset=dataset,
        save_identifier=save_identifier,
        model_feature=ModelFeatures.XGBOOST,
        build_model=build_model,
        feature_column=feature_column,
        undersample=undersample,
    )


def get_xgboost_labels(
    xgboost_classifier,
    df,
    feature_column: str = FeatureColumns.CONTENT,
):
    return predict_labels(xgboost_classifier, df, feature_column)


@report_time
def run():
    datasets = [Datasets.NELA_COVID_SUBSET]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_xgb_model(
                dataset,
                save_identifier="final",
                feature_column=column,
                undersample=True,
                optimize=False,
            )


if __name__ == "__main__":
    run()

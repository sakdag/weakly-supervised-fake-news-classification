from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.consts import Dataset, Datasets
from src.utils.time_utils import report_time
from src.weak_labeling.consts.model_features import FeatureColumns, ModelFeatures
from src.weak_labeling.model_based.common.trainer import (
    predict_labels,
    train_and_save_model,
)


def get_best_adaboost_parameters(tfidf_train, y_train) -> dict:
    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    parameters = {
        "base_estimator__max_depth": [i for i in range(2, 11, 2)],
        "base_estimator__min_samples_leaf": [5, 10],
        "n_estimators": [10, 50, 250, 1000],
        "learning_rate": [0.01, 0.1],
    }
    grid_search = GridSearchCV(abc, parameters, verbose=3, scoring="f1", n_jobs=-1)
    grid_search.fit(tfidf_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_params_


def train_adaboost_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
    undersample: bool = False,
    optimize: bool = False,
):
    if optimize:
        from src.preprocessing.read_raw_data import balanced_undersample, read_csv
        from src.weak_labeling.model_based.common.get_vectors import get_tfidf_vectors
        from sklearn.model_selection import train_test_split

        df = read_csv(dataset.injected_path)
        if undersample:
            df = balanced_undersample(df, min(df["label"].value_counts()) * 2)
            print(f"Undersampled, dataframe size: {len(df)}")
        x_train, _, y_train, _ = train_test_split(df, df["label"], test_size=0.20)
        train_vectors = get_tfidf_vectors(x_train, feature_column)
        params = get_best_adaboost_parameters(train_vectors, y_train)
    else:
        params = {
            "base_estimator__max_depth": 10,
            "base_estimator__min_samples_leaf": 1,
            "learning_rate": 1.0,
            "n_estimators": 50,
        }

    def build_model():
        return AdaBoostClassifier(
            DecisionTreeClassifier(
                max_depth=params["base_estimator__max_depth"],
                min_samples_leaf=params["base_estimator__min_samples_leaf"],
            ),
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=1,
        )

    return train_and_save_model(
        dataset=dataset,
        save_identifier=save_identifier,
        model_feature=ModelFeatures.ADABOOST,
        build_model=build_model,
        feature_column=feature_column,
        undersample=undersample,
    )


def get_adaboost_labels(
    adaboost_classifier,
    df,
    feature_column: str = FeatureColumns.CONTENT,
):
    return predict_labels(adaboost_classifier, df, feature_column)


@report_time
def run():
    datasets = [Datasets.NELA_COVID_SUBSET]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_adaboost_model(
                dataset,
                save_identifier="final",
                feature_column=column,
                undersample=True,
                optimize=False,
            )


if __name__ == "__main__":
    run()

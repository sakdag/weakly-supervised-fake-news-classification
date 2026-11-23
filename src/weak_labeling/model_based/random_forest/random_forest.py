from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.preprocessing.consts import Dataset, Datasets
from src.utils.time_utils import report_time
from src.weak_labeling.consts.model_features import FeatureColumns, ModelFeatures
from src.weak_labeling.model_based.common.trainer import (
    predict_labels,
    train_and_save_model,
)


def get_best_random_forest_parameters(tfidf_train, y_train) -> dict:
    rfc = RandomForestClassifier()
    parameters = {
        "n_estimators": [90, 100, 115, 130],
        "max_depth": range(2, 20),
        "min_samples_leaf": range(1, 10),
        "min_samples_split": range(2, 10),
        "max_features": ["sqrt", "log2"],
    }
    grid_search = GridSearchCV(rfc, parameters, verbose=3, scoring="f1", n_jobs=-1)
    grid_search.fit(tfidf_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_params_


def train_random_forest_model(
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
        params = get_best_random_forest_parameters(train_vectors, y_train)
    else:
        params = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 1,
            "min_samples_split": 5,
            "max_features": "sqrt",
        }

    def build_model():
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            random_state=42,
        )

    return train_and_save_model(
        dataset=dataset,
        save_identifier=save_identifier,
        model_feature=ModelFeatures.RANDOM_FOREST,
        build_model=build_model,
        feature_column=feature_column,
        undersample=undersample,
    )


def get_random_forest_labels(
    random_forest_classifier,
    df,
    feature_column: str = FeatureColumns.CONTENT,
):
    return predict_labels(random_forest_classifier, df, feature_column)


@report_time
def run():
    datasets = [Datasets.NELA_COVID_SUBSET]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_random_forest_model(
                dataset,
                save_identifier="final",
                feature_column=column,
                undersample=True,
                optimize=False,
            )


if __name__ == "__main__":
    run()

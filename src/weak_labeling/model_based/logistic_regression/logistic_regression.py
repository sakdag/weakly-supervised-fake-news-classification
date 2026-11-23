from sklearn.linear_model import LogisticRegression

from src.preprocessing.consts import Dataset, Datasets
from src.utils.time_utils import report_time
from src.weak_labeling.consts.model_features import FeatureColumns, ModelFeatures
from src.weak_labeling.model_based.common.trainer import (
    predict_labels,
    train_and_save_model,
)


def train_logistic_regression_model(
    dataset: Dataset,
    save_identifier: str,
    feature_column: str = FeatureColumns.CONTENT,
    undersample: bool = False,
):
    return train_and_save_model(
        dataset=dataset,
        save_identifier=save_identifier,
        model_feature=ModelFeatures.LOGISTIC_REGRESSION,
        build_model=LogisticRegression,
        feature_column=feature_column,
        undersample=undersample,
    )


def get_logistic_regression_labels(
    lr_classifier,
    df,
    feature_column: str = FeatureColumns.CONTENT,
):
    return predict_labels(lr_classifier, df, feature_column)


@report_time
def run():
    datasets = [Datasets.NELA_COVID_SUBSET]
    for dataset in datasets:
        for column in FeatureColumns.get_all():
            train_logistic_regression_model(
                dataset,
                save_identifier="final",
                feature_column=column,
                undersample=True,
            )


if __name__ == "__main__":
    run()

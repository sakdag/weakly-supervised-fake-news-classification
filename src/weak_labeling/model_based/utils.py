import os
import pickle

from src.preprocessing.consts import Dataset
from src.utils.path_utils import get_project_root


def get_model_file_name(classifier: str, feature_column: str, dataset: Dataset, identifier: str) -> str:
    return os.path.join(
        get_project_root(),
        f"model/weak_labeling/{classifier}/{dataset.name}-{feature_column}-{identifier}.pkl",
    )


def get_any_file_name(folder: str, name: str) -> str:
    return os.path.join(
        get_project_root(),
        f"model/weak_labeling/{folder}/{name}.pkl",
    )


def load_model(model_feature: str, feature_column: str, dataset: Dataset, identifier: str):
    xgb_file_name = get_model_file_name(model_feature, feature_column, dataset, identifier)
    return pickle.load(open(xgb_file_name, "rb"))

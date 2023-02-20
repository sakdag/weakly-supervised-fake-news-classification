import os
import pickle

from src.preprocessing.consts import Dataset
from src.utils.path_utils import get_project_root


def get_model_file_name(classifier: str, dataset: Dataset, identifier: str) -> str:
    return os.path.join(
        get_project_root(),
        f"model/weak_labeling/{classifier}/{dataset.name}-{identifier}.pkl",
    )


def load_model(model_feature: str, dataset: Dataset, identifier: str):
    xgb_file_name = get_model_file_name(model_feature, dataset, identifier)
    return pickle.load(open(xgb_file_name, "rb"))

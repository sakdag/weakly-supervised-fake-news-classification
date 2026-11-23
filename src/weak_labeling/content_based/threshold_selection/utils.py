import json
import os

from src.preprocessing.consts import Dataset
from src.utils.path_utils import get_project_root

THRESHOLDS_JSON_PREFIX = "content_feature_thresholds"


def get_thresholds_json(dataset: Dataset) -> dict:
    thresholds_json_path = os.path.join(
        get_project_root(), f"{THRESHOLDS_JSON_PREFIX}/{dataset.name}-thresholds.json"
    )
    print(f"Getting thresholds json, path: {thresholds_json_path}")
    with open(thresholds_json_path) as handle:
        return json.load(handle)


def update_and_save_thresholds_json(updated_thresholds: dict, dataset: Dataset):
    old_thresholds = get_thresholds_json(dataset)
    old_thresholds.update(updated_thresholds)

    thresholds_json_path = os.path.join(
        get_project_root(), f"{THRESHOLDS_JSON_PREFIX}/{dataset.name}-thresholds.json"
    )
    with open(thresholds_json_path, "w") as handle:
        json.dump(old_thresholds, handle, indent=2)

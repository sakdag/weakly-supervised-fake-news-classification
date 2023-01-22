import json
import os

from src.utils.path_utils import get_project_root

THRESHOLDS_JSON_PATH = (
    "src/weak_labeling/content_based/threshold_selection/lf_thresholds.json"
)


def get_thresholds_json() -> dict:
    thresholds_json_path = os.path.join(get_project_root(), THRESHOLDS_JSON_PATH)
    f = open(thresholds_json_path)
    return json.load(f)

def update_and_save_thresholds_json(updated_thresholds: dict):
    old_thresholds = get_thresholds_json()
    old_thresholds.update(updated_thresholds)

    thresholds_json_path = os.path.join(get_project_root(), THRESHOLDS_JSON_PATH)
    f = open(thresholds_json_path, "w")
    f.write(json.dumps(old_thresholds, indent=2))

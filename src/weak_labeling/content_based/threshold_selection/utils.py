import json
import os

from src.utils.path_utils import get_project_root

THRESHOLDS_JSON_PATH = (
    "src/weak_labeling/content_based/threshold_selection/lf_thresholds.json"
)


def get_thresholds_json():
    thresholds_json_path = os.path.join(get_project_root(), THRESHOLDS_JSON_PATH)
    f = open(thresholds_json_path)
    return json.load(f)

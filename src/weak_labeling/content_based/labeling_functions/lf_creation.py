from src.weak_labeling.consts.labels import Labels
from src.weak_labeling.content_based.threshold_selection.utils import (
    get_thresholds_json,
)
from snorkel.labeling import LabelingFunction


def make_upper_real_lf(feature, threshold):
    def upper_real(x):
        return Labels.REAL if x[feature] > threshold else Labels.ABSTAIN

    return LabelingFunction(
        name=f"lf_{feature}_upper_real",
        f=upper_real,
    )


def make_upper_fake_lf(feature, threshold):
    def upper_fake(x):
        return Labels.FAKE if x[feature] > threshold else Labels.ABSTAIN

    return LabelingFunction(
        name=f"lf_{feature}_upper_fake",
        f=upper_fake,
    )


def make_lower_real_lf(feature, threshold):
    def lower_real(x):
        return Labels.REAL if x[feature] < threshold else Labels.ABSTAIN

    return LabelingFunction(
        name=f"lf_{feature}_lower_real",
        f=lower_real,
    )


def make_lower_fake_lf(feature, threshold):
    def lower_fake(x):
        return Labels.FAKE if x[feature] < threshold else Labels.ABSTAIN

    return LabelingFunction(
        name=f"lf_{feature}_lower_fake",
        f=lower_fake,
    )


def create_content_lfs():
    thresholds_dict = get_thresholds_json()

    generated_lfs = []
    for feature in thresholds_dict.keys():
        value = thresholds_dict[feature]
        if "upper" in value.keys():
            if value["upper"]["side"] == "real":
                generated_lfs.append(
                    make_upper_real_lf(feature, value["upper"]["threshold"])
                )
            elif value["upper"]["side"] == "fake":
                generated_lfs.append(
                    make_upper_fake_lf(feature, value["upper"]["threshold"])
                )
        if "lower" in value.keys():
            if value["lower"]["side"] == "real":
                generated_lfs.append(
                    make_lower_real_lf(feature, value["lower"]["threshold"])
                )
            elif value["lower"]["side"] == "fake":
                generated_lfs.append(
                    make_lower_fake_lf(feature, value["lower"]["threshold"])
                )

    return generated_lfs


if __name__ == "__main__":
    lfs = create_content_lfs()
    print(lfs)

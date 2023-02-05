import enum
from dataclasses import dataclass

DATA_FOLDER_PREFIX = "data"
NELA_SOURCE_LABELS = "raw/nela/source-labels.csv"


@dataclass
class Dataset:
    raw_path: str
    sampled_path: str
    injected_path: str
    weak_labeled_path: str


class Datasets:
    NELA_COVID_2020 = Dataset(
        raw_path="raw/nela/nela-covid-2020.db",
        sampled_path="raw/nela/nela-covid-2020-sampled.csv",
        injected_path="feature_injected/nela/nela-covid-2020-sampled.csv",
        weak_labeled_path="with_weak_labels/nela/nela-covid-2020-sampled.csv",
    )
    NELA_COVID_2021 = Dataset(
        raw_path="raw/nela/nela-covid-2021.db",
        sampled_path="raw/nela/nela-covid-2021-sampled.csv",
        injected_path="feature_injected/nela/nela-covid-2021-sampled.csv",
        weak_labeled_path="with_weak_labels/nela/nela-covid-2021-sampled.csv",
    )
    NELA_ELECTIONS_2020 = Dataset(
        raw_path="raw/nela/nela-elections-2020.db",
        sampled_path="raw/nela/nela-elections-2020-sampled.csv",
        injected_path="feature_injected/nela/nela-elections-2020-sampled.csv",
        weak_labeled_path="with_weak_labels/nela/nela-elections-2020-sampled.csv",
    )
    NELA_CAPITOL_2021 = Dataset(
        raw_path="raw/nela/nela-capitol-2021.db",
        sampled_path="raw/nela/nela-capitol-2021-sampled.csv",
        injected_path="feature_injected/nela/nela-capitol-2021-sampled.csv",
        weak_labeled_path="with_weak_labels/nela/nela-capitol-2021-sampled.csv",
    )

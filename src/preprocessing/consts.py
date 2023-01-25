import enum


DATA_FOLDER_PREFIX = "data"


class Datasets(enum.Enum):
    NELA_COVID_2020_DATASET = "raw/nela/nela-covid-2020.db"
    NELA_COVID_2021_DATASET = "raw/nela/nela-covid-2021.db"
    NELA_ELECTIONS_2020_DATASET = "raw/nela/nela-elections-2020.db"
    NELA_CAPITOL_2021_DATASET = "raw/nela/nela-capitol-2021.db"
    NELA_COVID_2020_DATASET_SAMPLED = "raw/nela/nela-covid-2020-sampled.csv"
    NELA_COVID_2021_DATASET_SAMPLED = "raw/nela/nela-covid-2021-sampled.csv"
    NELA_ELECTIONS_2020_DATASET_SAMPLED = "raw/nela/nela-elections-2020-sampled.csv"
    NELA_CAPITOL_2021_DATASET_SAMPLED = "raw/nela/nela-capitol-2021-sampled.csv"
    NELA_SOURCE_LABELS = "raw/nela/source-labels.csv"

    # Feature injected datasets
    INJECTED_NELA_COVID_2020_DATASET_SAMPLED = (
        "feature_injected/nela/nela-covid-2020-sampled.csv"
    )
    INJECTED_NELA_COVID_2021_DATASET_SAMPLED = (
        "feature_injected/nela/nela-covid-2021-sampled.csv"
    )
    INJECTED_NELA_ELECTIONS_DATASET_2020_SAMPLED = (
        "feature_injected/nela/nela-elections-2020-sampled.csv"
    )
    INJECTED_NELA_CAPITOL_DATASET_2021_SAMPLED = (
        "feature_injected/nela/nela-capitol-2021-sampled.csv"
    )

    # Datasets with probabilistic labels
    LABELED_NELA_ELECTIONS_DATASET_2020_SAMPLED = (
        "with_weak_labels/nela/nela-elections-2020-sampled.csv"
    )
    LABELED_NELA_CAPITOL_DATASET_2021_SAMPLED = (
        "with_weak_labels/nela/nela-capitol-2021-sampled.csv"
    )

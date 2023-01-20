import enum


DATA_FOLDER_PREFIX = "data"


class Datasets(enum.Enum):
    NELA_COVID_DATASET = "raw/nela/nela-covid-2020.db"
    NELA_ELECTIONS_DATASET = "raw/nela/nela-elections-2020.db"
    NELA_COVID_DATASET_SAMPLED = "raw/nela/nela-covid-2020-sampled.csv"
    NELA_ELECTIONS_DATASET_SAMPLED = "raw/nela/nela-elections-2022-sampled.csv"
    NELA_SOURCE_LABELS = "raw/nela/source-labels.csv"

    # Feature injected datasets
    INJECTED_NELA_COVID_DATASET_SAMPLED = (
        "feature_injected/nela/nela-covid-2020-sampled.csv"
    )
    INJECTED_NELA_ELECTIONS_DATASET_SAMPLED = (
        "feature_injected/nela/nela-elections-2022-sampled.csv"
    )

    # Datasets with probabilistic labels
    LABELED_NELA_ELECTIONS_DATASET_SAMPLED = (
        "with_weak_labels/nela/nela-elections-2022-sampled.csv"
    )

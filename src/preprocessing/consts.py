import enum


RAW_DATA_FOLDER_PREFIX = "data/raw"


class RawDatasets(enum.Enum):
    COVID_DATASET = "nela-covid-2020.db"
    ELECTIONS_DATASET = "nela-elections-2020.db"
    COVID_DATASET_SAMPLED = "nela-covid-2020-sampled.csv"
    ELECTIONS_DATASET_SAMPLED = "nela-elections-2022-sampled.csv"
    SOURCE_LABELS = "source-labels.csv"

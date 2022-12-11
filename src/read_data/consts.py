import enum


RAW_DATA_FOLDER_PREFIX = "data/raw"


class RawDatasets(enum.Enum):
    COVID_DATASET = "nela-covid-2020.db"
    ELECTIONS_DATASET = "nela-elections-2020.db"

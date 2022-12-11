import os

from src.read_data.consts import RawDatasets, RAW_DATA_FOLDER_PREFIX
import argparse
import sqlite3
import pandas as pd

from src.utils.path_utils import get_project_root


def read_raw_data(dataset: RawDatasets = RawDatasets.ELECTIONS_DATASET):
    dataset_path = os.path.join(
        get_project_root(), RAW_DATA_FOLDER_PREFIX, dataset.value
    )
    print(f"Reading dataset from path: {dataset_path}")
    conn = sqlite3.connect(dataset_path)
    query = "SELECT * FROM newsdata"
    df = pd.read_sql_query(query, conn)

    print(df)


if __name__ == "__main__":
    read_raw_data(dataset=RawDatasets.ELECTIONS_DATASET)

import pandas as pd

from src.read_data.consts import RawDatasets
from src.read_data.read_raw_data import read_raw_data


def train_roberta_classifier(
    dataset: RawDatasets = RawDatasets.ELECTIONS_DATASET_SAMPLED,
):
    df: pd.DataFrame = read_raw_data(dataset)

    print(df.label.value_counts())
    for idx, row in df.iterrows():
        print(row)
        break


if __name__ == "__main__":
    train_roberta_classifier(dataset=RawDatasets.ELECTIONS_DATASET_SAMPLED)

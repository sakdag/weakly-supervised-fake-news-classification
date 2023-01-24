from typing import Optional

from src.preprocessing.consts import Datasets


def get_injected_dataset_name(dataset: Datasets) -> Optional[Datasets]:
    if dataset == Datasets.NELA_ELECTIONS_2020_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_ELECTIONS_DATASET_2020_SAMPLED
    elif dataset == Datasets.NELA_CAPITOL_2021_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_CAPITOL_DATASET_2021_SAMPLED
    elif dataset == Datasets.NELA_COVID_2020_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_COVID_2020_DATASET_SAMPLED
    elif dataset == Datasets.NELA_COVID_2021_DATASET_SAMPLED:
        injected_dataset = Datasets.INJECTED_NELA_COVID_2021_DATASET_SAMPLED
    elif dataset.value.startswith("feature_injected"):
        injected_dataset = dataset
    else:
        print(f"Cant inject to dataset: {dataset}")
        return None

    print(f"get_injected_dataset_name, input dataset: {dataset.value}, dataset name to inject: {injected_dataset.value}")

    return injected_dataset
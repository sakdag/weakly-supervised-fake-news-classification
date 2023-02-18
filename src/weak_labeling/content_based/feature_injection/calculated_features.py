from statistics import mean
from typing import List

import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

from src.weak_labeling.consts.content_features import ContentFeatures
from src.weak_labeling.consts.globals import NLP


def append_calculated_features(df: pd.DataFrame, features=List[str]) -> pd.DataFrame:
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    if not features:
        return df

    # Initialize empty columns to fill while iterating
    tqdm.pandas()

    res = df.progress_apply(
        append_calculated_features_helper,
        axis=1,
        result_type="expand",
        features_to_apply=features,
        sentence_transformer_model=model,
    )
    res.columns = features

    full = df.merge(res, left_index=True, right_index=True)
    print(full)

    return full


def append_calculated_features_helper(row, **kwargs):
    features = kwargs["features_to_apply"]
    model = kwargs["sentence_transformer_model"]

    spacy_content = NLP(row.content)

    title_embedding = model.encode(row.title)

    similarities = []
    for sentence in spacy_content.sents:
        current_sentence_embedding = model.encode(str(sentence))
        similarities.append(
            float(util.pytorch_cos_sim(title_embedding, current_sentence_embedding))
        )

    features_to_return = []
    if ContentFeatures.CONTENT_TITLE_BODY_SIMILARITY in features:
        features_to_return.append(mean(similarities))

    return tuple(features_to_return)

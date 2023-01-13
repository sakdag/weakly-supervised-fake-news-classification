import pandas as pd
import transformers
from keras_preprocessing import sequence
from sklearn.model_selection import train_test_split

from src.utils.preprocessing_utils import cleanup_df

ROBERTA_TOKENIZER = transformers.RobertaTokenizer.from_pretrained('roberta-base-openai-detector')

def xgboost_tokenizer(tokenizer_name, docs):
    features = []
    for doc in docs:
        tokens = tokenizer_name.tokenize(doc)
        ids = tokenizer_name.convert_tokens_to_ids(tokens)
        features.append(ids)
    return features

def preprocess_for_xgboost(df: pd.DataFrame):
    df_cleaned = cleanup_df(df)
    # df.sample(n=5000).reset_index(drop=True)

    X, y = df_cleaned['content'], df_cleaned['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    roberta_train_features = xgboost_tokenizer(ROBERTA_TOKENIZER, X_train)
    roberta_test_features = xgboost_tokenizer(ROBERTA_TOKENIZER, X_test)

    roberta_train = sequence.pad_sequences(roberta_train_features, maxlen=500)
    roberta_test = sequence.pad_sequences(roberta_test_features, maxlen=500)

    return roberta_train, roberta_test, y_test, y_train
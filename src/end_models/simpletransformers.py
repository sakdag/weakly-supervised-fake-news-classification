from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs

from src.preprocessing.consts import Datasets
from src.preprocessing.read_raw_data import read_weak_labeled_csv, balanced_undersample

if __name__ == "__main__":
    # df = read_weak_labeled_csv(
    #     Datasets.NELA_COVID_COMBINED.weak_labeled_path,
    #     train_df_name=Datasets.NELA_COVID_SUBSET.name
    #     top_n=10,
    # )
    df = read_weak_labeled_csv(
        Datasets.NELA_COVID_COMBINED.weak_labeled_path,
        train_df_name=Datasets.NELA_ELECTIONS_2020.name,
        top_n=5,
    )
    df = df[["title", "content", "actual_label", "snorkel_majority_vote_label"]].copy()
    df = df.drop(df[df.snorkel_majority_vote_label == -1].index)
    df = df.rename(
        columns={"snorkel_majority_vote_label": "weak_label", "content": "text"}
    )
    print(len(df))

    print(accuracy_score(df["actual_label"], df["weak_label"]))

    under_sampler = RandomUnderSampler(random_state=42)
    undersampled_df, undersampled_labels = under_sampler.fit_resample(
        df, df["weak_label"]
    )

    print(undersampled_df["weak_label"].value_counts())
    undersampled_df.head()

    undersampled_df = balanced_undersample(undersampled_df, 1000, "weak_label")
    undersampled_df["text"] = undersampled_df["title"] + ". " + undersampled_df["text"]

    undersampled_df.drop(["title"], axis=1, inplace=True)

    undersampled_df.head()

    len(undersampled_df)

    train_df, eval_df = train_test_split(
        undersampled_df,
        test_size=0.2,
        stratify=undersampled_df["weak_label"],
        random_state=42,
    )
    train_df, test_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df["weak_label"], random_state=42
    )

    print(len(train_df))
    print(len(eval_df))
    print(len(test_df))

    train_df_with_actual_labels = train_df.copy()
    train_df_with_actual_labels.rename(columns={"actual_label": "labels"}, inplace=True)
    train_df_with_actual_labels.drop(["weak_label"], axis=1, inplace=True)
    train_df_with_actual_labels.head()

    train_df_with_weak_labels = train_df.copy()
    train_df_with_weak_labels.rename(columns={"weak_label": "labels"}, inplace=True)
    train_df_with_weak_labels.drop(["actual_label"], axis=1, inplace=True)
    train_df_with_weak_labels.head()

    eval_df_with_actual_labels = eval_df.copy()
    eval_df_with_actual_labels.rename(columns={"actual_label": "labels"}, inplace=True)
    eval_df_with_actual_labels.drop(["weak_label"], axis=1, inplace=True)
    eval_df_with_actual_labels.head()

    eval_df_with_weak_labels = eval_df.copy()
    eval_df_with_weak_labels.rename(columns={"weak_label": "labels"}, inplace=True)
    eval_df_with_weak_labels.drop(["actual_label"], axis=1, inplace=True)
    eval_df_with_weak_labels.head()

    test_df_with_actual_labels = test_df.copy()
    test_df_with_actual_labels.rename(columns={"actual_label": "labels"}, inplace=True)
    test_df_with_actual_labels.drop(["weak_label"], axis=1, inplace=True)
    test_df_with_actual_labels.head()

    test_df_with_weak_labels = test_df.copy()
    test_df_with_weak_labels.rename(columns={"weak_label": "labels"}, inplace=True)
    test_df_with_weak_labels.drop(["actual_label"], axis=1, inplace=True)
    test_df_with_weak_labels.head()

    # import torch

    # is_cuda = torch.cuda.is_available()
    # if is_cuda:
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    # print(device)

    model_args = ClassificationArgs(
        num_train_epochs=3,
        overwrite_output_dir=True,
        manual_seed=42,
        use_multiprocessing=False,
        use_multiprocessing_for_evaluation=False,
    )
    model = ClassificationModel(
        model_type="roberta",
        model_name="roberta-base",
        use_cuda=True,
        num_labels=2,
        args=model_args,
    )

    # Train with weak labels
    # model.train_model(train_df_with_weak_labels, eval_df=eval_df_with_weak_labels)

    # Train with actual labels
    model.train_model(train_df_with_actual_labels, eval_df=eval_df_with_actual_labels)

    test_predictions, raw_outputs = model.predict(
        list(test_df_with_weak_labels["text"])
    )

    cr = classification_report(
        test_df_with_weak_labels["labels"], test_predictions, digits=5
    )
    print(cr)

    test_predictions, raw_outputs = model.predict(
        list(test_df_with_actual_labels["text"])
    )

    cr = classification_report(
        test_df_with_actual_labels["labels"], test_predictions, digits=5
    )
    print(cr)

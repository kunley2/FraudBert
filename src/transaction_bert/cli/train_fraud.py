"""Run the fraud-classifier notebook as a Python module."""

import argparse
import gc
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transaction_bert.data import (
    apply_quantization,
    load_transactions,
    split_data,
)
import transaction_bert.fraud as fraud
from transaction_bert.tokenization import (
    TransactionTokenizer,
)


def run(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fraud.device = device

    df = load_transactions(
        config["data_path"],
        nrows=config["max_rows"],
    )

    train_df, val_df, test_df = split_data(
        df,
        train_end=config["train_end"],
        validation_end=config["validation_end"],
    )

    del df
    gc.collect()

    dataset = [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]

    for name, part in dataset:
        print(
            name,
            part["is_fraud"].value_counts(
                normalize=True
            ),
        )

    foundation_dir = config[
        "foundation_model_dir"
    ]

    fraud.FOUNDATION_DIR = foundation_dir

    vocab_path = os.path.join(
        foundation_dir,
        "vocab.json",
    )

    preprocessing_path = os.path.join(
        foundation_dir,
        "preprocessing.json",
    )

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as file:
        token_to_id = json.load(file)

    with open(
        preprocessing_path,
        "r",
        encoding="utf-8",
    ) as file:
        preprocessing = json.load(file)

    print(
        "Vocabulary size:",
        len(token_to_id),
    )

    print(
        "Preprocessing keys:",
        preprocessing.keys(),
    )

    amount_boundaries = np.asarray(
        preprocessing["amount_boundaries"],
        dtype=np.float64,
    )

    delta_boundaries = np.asarray(
        preprocessing["delta_boundaries"],
        dtype=np.float64,
    )

    timestamp_boundaries = np.asarray(
        preprocessing["timestamp_boundaries"],
        dtype=np.float64,
    )

    for name, part in dataset:
        part["amount_bin"] = apply_quantization(
            part["amount_numeric"],
            amount_boundaries,
        )

        part["delta_bin"] = apply_quantization(
            part["previous_time"],
            delta_boundaries,
        )

        timestamp_seconds = (
            part["timestamp"].astype("int64")
            // 10**9
        )

        part["timestamp_bin"] = apply_quantization(
            timestamp_seconds,
            timestamp_boundaries,
        )

    tokenizer = TransactionTokenizer(
        token_to_id=token_to_id,
        include_timestamp=True,
    )

    keep_columns = [
        "user",
        "card",
        "timestamp",
        "hour",
        "day_of_week",
        "calendar_month",
        "day_of_month",
        "delta_bin",
        "amount_bin",
        "timestamp_bin",
        "use_chip",
        "merchant_name",
        "merchant_city",
        "merchant_state",
        "zip",
        "mcc",
        "errors",
        "is_fraud",
    ]

    all_df = pd.concat(
        [
            train_df[keep_columns],
            val_df[keep_columns],
            test_df[keep_columns],
        ],
        ignore_index=True,
        copy=False,
    )

    all_df = (
        all_df
        .sort_values(
            ["user", "timestamp"]
        )
        .reset_index(drop=True)
    )

    sample_row = next(
        train_df.itertuples(index=False)
    )

    sample_tokens = tokenizer.tokenize_transaction(
        sample_row
    )

    sample_ids = tokenizer.encode_transaction(
        sample_row
    )

    print(sample_tokens)
    print(sample_ids)
    print(tokenizer.decode(sample_ids))

    del train_df
    del val_df
    del test_df
    del dataset
    gc.collect()

    all_df["fraud_label"] = (
        all_df["is_fraud"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "YES": 1,
            "NO": 0,
        })
    )

    tokens_per_transaction = len(
        sample_ids
    )

    fraud.TOKENS_PER_TRANSACTION = (
        tokens_per_transaction
    )

    print(
        "Tokens per transaction:",
        tokens_per_transaction,
    )

    encoded_path = config["encoded_path"]

    encoded_transactions = np.memmap(
        encoded_path,
        dtype=np.uint16,
        mode="w+",
        shape=(
            len(all_df),
            tokens_per_transaction,
        ),
    )

    for index, row in enumerate(
        tqdm(
            all_df.itertuples(index=False),
            total=len(all_df),
            desc="Encoding transactions",
        )
    ):
        encoded_transactions[index] = np.asarray(
            tokenizer.encode_transaction(row),
            dtype=np.uint16,
        )

    encoded_transactions.flush()

    print(
        "Encoded transaction shape:",
        encoded_transactions.shape,
    )

    del encoded_transactions

    transactions_per_sequence = config[
        "transactions_per_sequence"
    ]

    history_required = (
        transactions_per_sequence - 1
    )

    users = all_df["user"].to_numpy()

    user_starts = np.flatnonzero(
        np.r_[True, users[1:] != users[:-1]]
    )

    user_ends = np.r_[
        user_starts[1:],
        len(all_df),
    ]

    target_parts = []

    for start, end in zip(
        user_starts,
        user_ends,
    ):
        first_target = (
            start + history_required
        )

        if first_target >= end:
            continue

        targets = np.arange(
            first_target,
            end,
            dtype=np.int32,
        )

        target_parts.append(targets)

    all_targets = np.concatenate(
        target_parts
    )

    del target_parts
    gc.collect()

    print(
        "Eligible fraud examples:",
        len(all_targets),
    )

    timestamps = all_df[
        "timestamp"
    ].to_numpy()

    labels = all_df[
        "fraud_label"
    ].to_numpy(dtype=np.uint8)

    fraud.labels = labels

    train_end = np.datetime64(
        config["train_end"]
    )

    validation_end = np.datetime64(
        config["validation_end"]
    )

    target_timestamps = timestamps[
        all_targets
    ]

    train_mask = (
        target_timestamps < train_end
    )

    validation_mask = (
        (target_timestamps >= train_end)
        & (target_timestamps < validation_end)
    )

    test_mask = (
        target_timestamps >= validation_end
    )

    train_targets_all = all_targets[
        train_mask
    ]

    validation_targets = all_targets[
        validation_mask
    ]

    test_targets = all_targets[
        test_mask
    ]

    print(
        "Training targets:",
        len(train_targets_all),
    )

    print(
        "Validation targets:",
        len(validation_targets),
    )

    print(
        "Test targets:",
        len(test_targets),
    )

    seed = config["seed"]
    negative_to_positive_ratio = config[
        "negative_to_positive_ratio"
    ]

    rng = np.random.default_rng(seed)

    train_y_all = labels[
        train_targets_all
    ]

    positive_targets = train_targets_all[
        train_y_all == 1
    ]

    negative_targets = train_targets_all[
        train_y_all == 0
    ]

    print(
        "Fraud:",
        len(positive_targets),
    )

    print(
        "Normal:",
        len(negative_targets),
    )

    number_negatives = min(
        len(negative_targets),
        len(positive_targets)
        * negative_to_positive_ratio,
    )

    sampled_negative_targets = rng.choice(
        negative_targets,
        size=number_negatives,
        replace=False,
    )

    train_targets = np.concatenate([
        positive_targets,
        sampled_negative_targets,
    ])

    rng.shuffle(
        train_targets
    )

    fraud.show_distribution(
        "TRAIN",
        train_targets,
    )

    fraud.show_distribution(
        "VALIDATION",
        validation_targets,
    )

    fraud.show_distribution(
        "TEST",
        test_targets,
    )

    train_dataset = fraud.FraudWindowDataset(
        targets=train_targets,
        labels=labels,
        encoded_path=encoded_path,
        number_rows=len(all_df),
        tokens_per_transaction=(
            tokens_per_transaction
        ),
        transactions_per_sequence=(
            transactions_per_sequence
        ),
        tokenizer=tokenizer,
    )

    validation_dataset = fraud.FraudWindowDataset(
        targets=validation_targets,
        labels=labels,
        encoded_path=encoded_path,
        number_rows=len(all_df),
        tokens_per_transaction=(
            tokens_per_transaction
        ),
        transactions_per_sequence=(
            transactions_per_sequence
        ),
        tokenizer=tokenizer,
    )

    test_dataset = fraud.FraudWindowDataset(
        targets=test_targets,
        labels=labels,
        encoded_path=encoded_path,
        number_rows=len(all_df),
        tokens_per_transaction=(
            tokens_per_transaction
        ),
        transactions_per_sequence=(
            transactions_per_sequence
        ),
        tokenizer=tokenizer,
    )

    batch_size = config["training"][
        "batch_size"
    ]

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    fraud.train_loader = train_loader
    fraud.validation_loader = validation_loader
    fraud.test_loader = test_loader
    fraud.TRANSACTIONS_PER_SEQUENCE = (
        transactions_per_sequence
    )

    train_labels_sampled = labels[
        train_targets
    ]

    positive_count = (
        train_labels_sampled.sum()
    )

    negative_count = (
        len(train_labels_sampled)
        - positive_count
    )

    pos_weight = (
        negative_count
        / positive_count
    )

    fraud.POS_WEIGHT = pos_weight

    print(
        "Positive weight:",
        pos_weight,
    )

    training = config["training"]
    fine_tuning = config["fine_tuning"]

    current_model = fraud.create_fraud_model(
        head_type="current_mlp",
    )

    current_model = fraud.train_model(
        current_model,
        epochs=training["epochs"],
        head_lr=training["head_lr"],
        encoder_lr=training["encoder_lr"],
        patience=training["patience"],
    )

    current_results = fraud.evaluate_model(
        current_model,
    )

    del current_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_model = fraud.create_fraud_model(
        head_type="mean_mlp",
    )

    mean_model = fraud.train_model(
        mean_model,
        epochs=training["epochs"],
        head_lr=training["head_lr"],
        encoder_lr=training["encoder_lr"],
        patience=training["patience"],
    )

    mean_results = fraud.evaluate_model(
        mean_model,
    )

    del mean_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lstm_model = fraud.create_fraud_model(
        head_type="lstm",
    )

    lstm_model = fraud.train_model(
        lstm_model,
        epochs=training["epochs"],
        head_lr=training["head_lr"],
        encoder_lr=training["encoder_lr"],
        patience=training["patience"],
    )

    lstm_results = fraud.evaluate_model(
        lstm_model,
    )

    os.makedirs(
        config["output_dir"],
        exist_ok=True,
    )

    torch.save(
        lstm_model.state_dict(),
        os.path.join(
            config["output_dir"],
            "frozen_bert_lstm_fraud.pt",
        ),
    )

    del lstm_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetune_model = fraud.create_fraud_model(
        head_type="lstm",
        fine_tune_last_layers=1,
    )

    finetune_model = fraud.train_model(
        finetune_model,
        epochs=fine_tuning["epochs"],
        head_lr=fine_tuning["head_lr"],
        encoder_lr=fine_tuning["encoder_lr"],
        patience=fine_tuning["patience"],
    )

    finetune_results = fraud.evaluate_model(
        finetune_model,
    )

    comparison = pd.DataFrame([
        {
            "Model": (
                "BERT Current Transaction + MLP"
            ),
            **current_results,
        },
        {
            "Model": (
                "BERT Whole-Sequence Mean + MLP"
            ),
            **mean_results,
        },
        {
            "Model": (
                "Frozen BERT + Transaction LSTM"
            ),
            **lstm_results,
        },
        {
            "Model": (
                "Last BERT Layer FT + LSTM"
            ),
            **finetune_results,
        },
    ])

    print(comparison)

    del finetune_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetune_model2 = fraud.create_fraud_model(
        head_type="lstm",
        fine_tune_last_layers=2,
    )

    finetune_model2 = fraud.train_model(
        finetune_model2,
        epochs=fine_tuning["epochs"],
        head_lr=fine_tuning["head_lr"],
        encoder_lr=fine_tuning["encoder_lr"],
        patience=fine_tuning["patience"],
    )

    finetune_results2 = fraud.evaluate_model(
        finetune_model2,
    )

    print(
        "Two BERT layers fine-tuned:",
        finetune_results2,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetune_model2 = fraud.train_model(
        finetune_model2,
        epochs=config["long_fine_tuning_epochs"],
        head_lr=fine_tuning["head_lr"],
        encoder_lr=fine_tuning["encoder_lr"],
        patience=fine_tuning["patience"],
    )

    finetune_results2 = fraud.evaluate_model(
        finetune_model2,
    )

    print(
        "Two BERT layers after longer training:",
        finetune_results2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fraud.json",
    )
    args = parser.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    run(config)


if __name__ == "__main__":
    main()

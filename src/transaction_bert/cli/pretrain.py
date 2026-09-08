"""Run the foundation-model notebook as a Python module."""

import argparse
import gc
import json
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
)

from transaction_bert.data import (
    apply_quantization,
    fit_quantile_boundaries,
    load_transactions,
    split_data,
)
import transaction_bert.foundation as foundation
import transaction_bert.tokenization as tokenization
from transaction_bert.tokenization import (
    SPECIAL_TOKENS,
    TransactionTokenizer,
    build_vocabulary,
    get_field,
)


def run(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    amount_boundaries = fit_quantile_boundaries(
        train_df["amount_numeric"],
        config["amount_bins"],
    )

    for name, part in dataset:
        part["amount_bin"] = apply_quantization(
            part["amount_numeric"],
            amount_boundaries,
        )

        part["timestamp_seconds"] = (
            part["timestamp"].astype("int64")
            // 10**9
        )

    timestamp_boundaries = fit_quantile_boundaries(
        train_df["timestamp_seconds"],
        config["timestamp_bins"],
    )

    for name, part in dataset:
        part["timestamp_bin"] = apply_quantization(
            part["timestamp_seconds"],
            timestamp_boundaries,
        )

    delta_boundaries = fit_quantile_boundaries(
        train_df["previous_time"],
        config["delta_bins"],
    )

    for name, part in dataset:
        part["delta_bin"] = apply_quantization(
            part["previous_time"],
            delta_boundaries,
        )

    temporary_token_to_id = {
        token: idx
        for idx, token
        in enumerate(SPECIAL_TOKENS)
    }

    sequence_tokenizer = TransactionTokenizer(
        token_to_id=temporary_token_to_id,
        include_timestamp=True,
    )

    tokenization.tokenizer = sequence_tokenizer

    sample_row = next(
        train_df.itertuples(index=False)
    )

    print(
        sequence_tokenizer.tokenize_transaction(
            sample_row
        )
    )

    train_sequences = tokenization.build_sequence(
        train_df,
        transaction_per_sequence=(
            config["transactions_per_sequence"]
        ),
        stride=config["stride"],
        max_sequence=config["max_train_sequences"],
    )

    del train_df
    gc.collect()

    val_sequences = tokenization.build_sequence(
        val_df,
        transaction_per_sequence=(
            config["transactions_per_sequence"]
        ),
        stride=config["stride"],
        max_sequence=config["max_validation_sequences"],
    )

    del val_df
    del test_df
    del dataset
    gc.collect()

    print(
        len(train_sequences),
        len(val_sequences),
    )

    token_to_id = build_vocabulary(
        train_sequences,
        minimum_frequency=(
            config["minimum_token_frequency"]
        ),
        field_limits=config["field_limits"],
    )

    id_to_token = {
        idx: token
        for token, idx
        in token_to_id.items()
    }

    tokenizer = TransactionTokenizer(
        token_to_id=token_to_id,
        include_timestamp=True,
    )

    foundation.tokenizer = tokenizer
    foundation.device = device

    print(
        "Vocabulary size:",
        len(token_to_id),
    )

    tokens_per_transaction = len(
        tokenizer.tokenize_transaction(sample_row)
    )

    max_length = (
        config["transactions_per_sequence"]
        * tokens_per_transaction
        + 2
    )

    print(
        "Tokens / transaction:",
        tokens_per_transaction,
    )

    print(
        "Max sequence length:",
        max_length,
    )

    train_dataset = foundation.FinancialTransactionDataset(
        train_sequences,
        max_length,
    )

    val_dataset = foundation.FinancialTransactionDataset(
        val_sequences,
        max_length,
    )

    field_to_ids = defaultdict(list)
    id_to_field = {}

    for idx, token in id_to_token.items():
        field = get_field(token)
        id_to_field[idx] = field

        if field is not None:
            field_to_ids[field].append(idx)

    fields = sorted(field_to_ids.keys())
    field_unknown_ids = set()

    for field in fields:
        token = f"[UNK_{field}]"
        field_unknown_ids.add(
            token_to_id[token]
        )

    for field in field_to_ids:
        field_to_ids[field] = [
            idx
            for idx in field_to_ids[field]
            if idx not in field_unknown_ids
        ]

    foundation.field_unknown_ids = field_unknown_ids
    foundation.id_to_field = id_to_field
    foundation.field_to_ids = field_to_ids

    collator = foundation.FinancialMLMCollator(
        mlm_probability=config["mlm_probability"],
    )

    foundation.collator = collator

    model_config = config["model"]

    bert_config = BertConfig(
        vocab_size=len(token_to_id),
        hidden_size=model_config["hidden_size"],
        num_hidden_layers=(
            model_config["num_hidden_layers"]
        ),
        num_attention_heads=(
            model_config["num_attention_heads"]
        ),
        intermediate_size=(
            model_config["intermediate_size"]
        ),
        hidden_dropout_prob=(
            model_config["hidden_dropout_probability"]
        ),
        attention_probs_dropout_prob=(
            model_config["attention_dropout_probability"]
        ),
        max_position_embeddings=(
            max_length + 8
        ),
        pad_token_id=token_to_id["[PAD]"],
    )

    model = BertForMaskedLM(
        bert_config
    )

    model.to(device)

    number_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"{number_parameters / 1e6:.2f}M parameters"
    )

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
    )

    batch = next(
        iter(loader)
    )

    for key, value in batch.items():
        print(
            key,
            value.shape,
        )

    batch = {
        key: value.to(device)
        for key, value
        in batch.items()
    }

    with torch.no_grad():
        output = model(
            **batch
        )

    print(
        "Initial loss:",
        output.loss.item(),
    )

    training = config["training"]

    training_args = TrainingArguments(
        output_dir=config["checkpoint_dir"],
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=(
            training["train_batch_size"]
        ),
        per_device_eval_batch_size=(
            training["eval_batch_size"]
        ),
        gradient_accumulation_steps=(
            training["gradient_accumulation_steps"]
        ),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_steps"],
        logging_steps=training["logging_steps"],
        eval_strategy="steps",
        eval_steps=training["eval_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=(
            training["dataloader_num_workers"]
        ),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)

    encoder_metrics = foundation.evaluate_masked_accuracy(
        model,
        val_dataset,
    )

    print(encoder_metrics)

    pretrained_dir = config["output_dir"]
    os.makedirs(
        pretrained_dir,
        exist_ok=True,
    )

    model.save_pretrained(
        pretrained_dir
    )

    tokenizer.save(
        os.path.join(
            pretrained_dir,
            "vocab.json",
        )
    )

    with open(
        os.path.join(
            pretrained_dir,
            "preprocessing.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "amount_boundaries": (
                    amount_boundaries.tolist()
                ),
                "delta_boundaries": (
                    delta_boundaries.tolist()
                ),
                "timestamp_boundaries": (
                    timestamp_boundaries.tolist()
                ),
                "transactions_per_sequence": (
                    config["transactions_per_sequence"]
                ),
                "tokens_per_transaction": (
                    tokens_per_transaction
                ),
                "max_length": max_length,
            },
            file,
        )

    print(
        "Saved foundation model to:",
        pretrained_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pretrain.json",
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

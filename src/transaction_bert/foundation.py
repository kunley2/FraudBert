"""Classes and functions used by the foundation-model notebook."""

import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset


tokenizer = None
field_unknown_ids = set()
id_to_field = {}
field_to_ids = {}
collator = None
device = None


class FinancialTransactionDataset(Dataset):
    def __init__(
        self,
        sequences,
        max_length,
    ):
        self.sequences = sequences
        self.max_length = max_length

        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(
            self.sequences
        )

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        ids = [
            tokenizer.encode_token(token)
            for token in sequence
        ]

        ids = (
            [self.cls_id]
            + ids[:self.max_length - 2]
            + [self.sep_id]
        )

        attention_mask = (
            [1] * len(ids)
        )

        padding = (
            self.max_length
            - len(ids)
        )

        ids += (
            [self.pad_id]
            * padding
        )

        attention_mask += (
            [0]
            * padding
        )

        return {
            "input_ids": torch.tensor(
                ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }


class FinancialMLMCollator:
    def __init__(
        self,
        mlm_probability=0.15,
    ):
        self.mlm_probability = mlm_probability
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.txn_id = tokenizer.txn_token_id
        self.never_mask = {
            self.pad_id,
            self.cls_id,
            self.sep_id,
            self.txn_id,
            *field_unknown_ids,
        }

    def __call__(self, examples):
        input_ids = torch.stack([
            x["input_ids"]
            for x in examples
        ])

        attention_mask = torch.stack([
            x["attention_mask"]
            for x in examples
        ])

        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability,
        )

        for token_id in self.never_mask:
            probability_matrix.masked_fill_(
                input_ids == token_id,
                0.0,
            )

        masked_indices = torch.bernoulli(
            probability_matrix
        ).bool()

        labels[
            ~masked_indices
        ] = -100

        replace_with_mask = (
            torch.rand(labels.shape) < 0.8
        ) & masked_indices

        input_ids[
            replace_with_mask
        ] = self.mask_id

        remaining = (
            masked_indices
            & ~replace_with_mask
        )

        replace_random = (
            torch.rand(labels.shape) < 0.5
        ) & remaining

        positions = (
            replace_random
            .nonzero(as_tuple=False)
        )

        for batch_idx, pos_idx in positions:
            original_id = int(
                labels[
                    batch_idx,
                    pos_idx,
                ]
            )

            field = id_to_field.get(
                original_id
            )

            candidates = field_to_ids.get(
                field,
                [],
            )

            if candidates:
                random_id = random.choice(
                    candidates
                )

                input_ids[
                    batch_idx,
                    pos_idx,
                ] = random_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def evaluate_masked_accuracy(
    model,
    dataset,
    batch_size=32,
):
    torch.manual_seed(1234)
    random.seed(1234)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model.eval()

    total_correct = 0
    total_count = 0

    total_loss = 0.0
    total_loss_tokens = 0

    field_correct = defaultdict(int)
    field_total = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value
                in batch.items()
            }

            outputs = model(
                **batch
            )

            labels = batch["labels"]

            predictions = (
                outputs.logits.argmax(dim=-1)
            )

            masked = (
                labels != -100
            )

            masked_count = (
                masked.sum().item()
            )

            if masked_count == 0:
                continue

            total_loss += (
                outputs.loss.item()
                * masked_count
            )

            total_loss_tokens += masked_count

            correct = (
                predictions[masked]
                == labels[masked]
            )

            total_correct += (
                correct.sum().item()
            )

            total_count += masked_count

            true_ids = (
                labels[masked]
                .detach()
                .cpu()
                .tolist()
            )

            predicted_ids = (
                predictions[masked]
                .detach()
                .cpu()
                .tolist()
            )

            for true_id, pred_id in zip(
                true_ids,
                predicted_ids,
            ):
                field = id_to_field.get(
                    true_id
                )

                if field is None:
                    continue

                field_total[field] += 1

                if true_id == pred_id:
                    field_correct[field] += 1

    overall_accuracy = (
        total_correct
        / max(total_count, 1)
    )

    average_loss = (
        total_loss
        / max(total_loss_tokens, 1)
    )

    per_field_accuracy = {}

    for field in sorted(field_total):
        per_field_accuracy[field] = (
            field_correct[field]
            / field_total[field]
        )

    return {
        "mlm_loss": average_loss,
        "masked_accuracy": overall_accuracy,
        "per_field_accuracy": per_field_accuracy,
    }

"""Classes and functions used by the fraud-classifier notebook."""

import gc

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import BertForMaskedLM


device = None
labels = None
train_loader = None
validation_loader = None
test_loader = None
FOUNDATION_DIR = None
TOKENS_PER_TRANSACTION = None
TRANSACTIONS_PER_SEQUENCE = None
POS_WEIGHT = None


def show_distribution(name, targets):
    y = labels[targets]
    print(f"\n{name}")
    print("Transactions:", len(y))
    print("Fraud:", int(y.sum()))
    print("Fraud rate:", f"{y.mean():.6%}")


class FraudWindowDataset(Dataset):
    def __init__(
        self,
        targets,
        labels,
        encoded_path,
        number_rows,
        tokens_per_transaction,
        transactions_per_sequence,
        tokenizer,
    ):
        self.targets = np.asarray(
            targets,
            dtype=np.int32,
        )

        self.labels = labels
        self.encoded_path = encoded_path
        self.number_rows = number_rows
        self.tokens_per_transaction = tokens_per_transaction
        self.transactions_per_sequence = transactions_per_sequence
        self.tokenizer = tokenizer
        self._mmap = None

    def _open_memmap(self):
        if self._mmap is None:
            self._mmap = np.memmap(
                self.encoded_path,
                dtype=np.uint16,
                mode="r",
                shape=(
                    self.number_rows,
                    self.tokens_per_transaction,
                ),
            )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self._open_memmap()

        target_index = int(
            self.targets[index]
        )

        start_index = (
            target_index
            - self.transactions_per_sequence
            + 1
        )

        transaction_ids = (
            self._mmap[
                start_index:target_index + 1
            ]
            .reshape(-1)
            .astype(np.int64)
        )

        input_ids = np.concatenate([
            np.asarray(
                [self.tokenizer.cls_token_id],
                dtype=np.int64,
            ),
            transaction_ids,
            np.asarray(
                [self.tokenizer.sep_token_id],
                dtype=np.int64,
            ),
        ])

        attention_mask = np.ones(
            len(input_ids),
            dtype=np.int64,
        )

        label = float(
            self.labels[target_index]
        )

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "labels": torch.tensor(
                label,
                dtype=torch.float32,
            ),
        }


class FinancialBERTFraudDetector(nn.Module):
    def __init__(
        self,
        encoder,
        hidden_size,
        head_type="lstm",
        lstm_hidden_size=128,
        dropout=0.2,
    ):
        super().__init__()

        self.encoder = encoder
        self.hidden_size = hidden_size
        self.head_type = head_type
        self.encoder_frozen = True

        self.freeze_encoder()

        if head_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )

            classifier_input = lstm_hidden_size

        elif head_type in {
            "current_mlp",
            "mean_mlp",
            "cls_mlp",
        }:
            classifier_input = hidden_size

        else:
            raise ValueError(
                f"Unknown head: {head_type}"
            )

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def freeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.encoder_frozen = True

    def unfreeze_last_n_layers(self, n=1):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        for layer in self.encoder.encoder.layer[-n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        self.encoder_frozen = False

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        if self.encoder_frozen:
            self.encoder.eval()

            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return outputs.last_hidden_state

    def transaction_pool(self, hidden):
        hidden = hidden[:, 1:-1, :]

        batch_size = hidden.shape[0]

        hidden = hidden.reshape(
            batch_size,
            TRANSACTIONS_PER_SEQUENCE,
            TOKENS_PER_TRANSACTION,
            self.hidden_size,
        )

        field_hidden = hidden[:, :, 1:, :]

        transaction_embeddings = field_hidden.mean(
            dim=2
        )

        return transaction_embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        hidden = self.encode(
            input_ids,
            attention_mask,
        )

        if self.head_type == "cls_mlp":
            representation = hidden[:, 0, :]

        else:
            transaction_embeddings = (
                self.transaction_pool(hidden)
            )

            if self.head_type == "current_mlp":
                representation = (
                    transaction_embeddings[:, -1, :]
                )

            elif self.head_type == "mean_mlp":
                representation = (
                    transaction_embeddings.mean(dim=1)
                )

            elif self.head_type == "lstm":
                sequence_output, _ = self.lstm(
                    transaction_embeddings
                )

                representation = (
                    sequence_output[:, -1, :]
                )

        logits = self.classifier(
            representation
        ).squeeze(-1)

        return logits


def create_fraud_model(
    head_type,
    fine_tune_last_layers=0,
):
    foundation_mlm = BertForMaskedLM.from_pretrained(
        FOUNDATION_DIR
    )

    model = FinancialBERTFraudDetector(
        encoder=foundation_mlm.bert,
        hidden_size=foundation_mlm.config.hidden_size,
        head_type=head_type,
        lstm_hidden_size=128,
        dropout=0.2,
    )

    if fine_tune_last_layers > 0:
        model.unfreeze_last_n_layers(
            fine_tune_last_layers
        )

    del foundation_mlm
    gc.collect()

    return model


@torch.no_grad()
def predict(model, loader):
    model.eval()

    all_probabilities = []
    all_labels = []

    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(
            input_ids,
            attention_mask,
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

        all_probabilities.append(
            probabilities
        )

        all_labels.append(
            batch["labels"].numpy()
        )

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall
        / np.maximum(
            precision + recall,
            1e-12,
        )
    )

    best = np.argmax(f1)

    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }


def train_model(
    model,
    epochs=6,
    head_lr=3e-4,
    encoder_lr=1e-5,
    patience=2,
):
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            POS_WEIGHT,
            dtype=torch.float32,
            device=device,
        )
    )

    head_parameters = []
    encoder_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("encoder."):
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    parameter_groups = [
        {
            "params": head_parameters,
            "lr": head_lr,
        }
    ]

    if encoder_parameters:
        parameter_groups.append({
            "params": encoder_parameters,
            "lr": encoder_lr,
        })

    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=1e-4,
    )

    best_ap = -1
    best_state = None
    no_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        examples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                1.0,
            )

            optimizer.step()

            size = input_ids.size(0)
            running_loss += loss.item() * size
            examples += size

            progress.set_postfix(
                loss=(running_loss / examples)
            )

        y_val, p_val = predict(
            model,
            validation_loader,
        )

        val_ap = average_precision_score(
            y_val,
            p_val,
        )

        val_auc = roc_auc_score(
            y_val,
            p_val,
        )

        threshold_info = find_best_threshold(
            y_val,
            p_val,
        )

        print(f"\nEpoch {epoch}")
        print(f"Validation AP: {val_ap:.6f}")
        print(f"Validation ROC-AUC: {val_auc:.6f}")
        print("Best validation:", threshold_info)

        if val_ap > best_ap:
            best_ap = val_ap

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            no_improvement = 0

        else:
            no_improvement += 1

            if no_improvement >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)

    return model


def evaluate_model(
    model,
):
    y_val, p_val = predict(
        model,
        validation_loader,
    )

    threshold_info = find_best_threshold(
        y_val,
        p_val,
    )

    threshold = threshold_info["threshold"]

    y_test, p_test = predict(
        model,
        test_loader,
    )

    predictions = (
        p_test >= threshold
    ).astype(np.uint8)

    ap = average_precision_score(
        y_test,
        p_test,
    )

    roc_auc = roc_auc_score(
        y_test,
        p_test,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    result = {
        "PR_AUC_AP": ap,
        "ROC_AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Threshold": threshold,
    }

    print(result)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=5,
            zero_division=0,
        )
    )

    return result

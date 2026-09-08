"""The transaction tokens and tokenizer used in both notebooks."""

import json
import re
from collections import Counter, defaultdict

import pandas as pd


SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[TXN]",
]


FIELD_LIMITS = {
    "MERCHANT": 20_000,
    "CITY": 5_000,
    "ZIP": 10_000,
}

tokenizer = None


def clean_data(values):
    if pd.isna(values):
        return "NONE"

    values = str(values).strip().upper()
    values = re.sub(r"\s+", "_", values)
    values = values.replace("=", "_")
    return values


def clean_zip(values):
    if pd.isna(values):
        return "NONE"

    try:
        number = float(values)
        if number.is_integer():
            return str(int(number))

    except (TypeError, ValueError):
        pass

    return clean_data(values)


def transaction_to_tokens(row):
    return [
        "[TXN]",
        f"CARD={clean_data(row.card)}",
        f"TIMESTAMP={row.timestamp_bin}",
        f"HOUR={row.hour}",
        f"DOW={row.day_of_week}",
        f"MONTH={row.calendar_month}",
        f"DOM={row.day_of_month}",
        f"DELTA={row.delta_bin}",
        f"AMOUNT={row.amount_bin}",
        f"CHANNEL={clean_data(row.use_chip)}",
        f"MERCHANT={clean_data(row.merchant_name)}",
        f"CITY={clean_data(row.merchant_city)}",
        f"STATE={clean_data(row.merchant_state)}",
        f"ZIP={clean_zip(row.zip)}",
        f"MCC={clean_data(row.mcc)}",
        f"ERROR={clean_data(row.errors)}",
    ]


def get_field(token):
    if token.startswith("["):
        return None

    if "=" not in token:
        return None

    return token.split("=", 1)[0]


class TransactionTokenizer:
    """The tokenizer class from the fraud notebook, shared with pretraining."""

    def __init__(
        self,
        token_to_id,
        include_timestamp=True,
    ):
        self.token_to_id = token_to_id

        self.id_to_token = {
            idx: token
            for token, idx
            in token_to_id.items()
        }

        self.include_timestamp = include_timestamp

    @staticmethod
    def clean(value):
        return clean_data(value)

    @staticmethod
    def clean_zip(value):
        return clean_zip(value)

    @staticmethod
    def get_field(token):
        return get_field(token)

    def tokenize_transaction(self, row):
        tokens = transaction_to_tokens(row)

        if not self.include_timestamp:
            del tokens[2]

        return tokens

    def encode_token(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]

        field = self.get_field(token)

        if field is not None:
            unknown_token = f"[UNK_{field}]"

            if unknown_token in self.token_to_id:
                return self.token_to_id[
                    unknown_token
                ]

        return self.token_to_id["[UNK]"]

    def encode_transaction(self, row):
        tokens = self.tokenize_transaction(row)

        return [
            self.encode_token(token)
            for token in tokens
        ]

    def decode(self, ids):
        return [
            self.id_to_token.get(
                int(idx),
                "[UNK]",
            )
            for idx in ids
        ]

    @property
    def cls_token_id(self):
        return self.token_to_id["[CLS]"]

    @property
    def sep_token_id(self):
        return self.token_to_id["[SEP]"]

    @property
    def pad_token_id(self):
        return self.token_to_id["[PAD]"]

    @property
    def unk_token_id(self):
        return self.token_to_id["[UNK]"]

    @property
    def txn_token_id(self):
        return self.token_to_id["[TXN]"]

    @property
    def mask_token_id(self):
        return self.token_to_id["[MASK]"]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.token_to_id, file)

    def __len__(self):
        return len(self.token_to_id)


def build_sequence(
    dataframe,
    transaction_per_sequence=16,
    stride=8,
    max_sequence=None,
):
    sequences = []
    dataframe = dataframe.sort_values(["user", "timestamp"])

    for user_id, user_df in dataframe.groupby("user", sort=False):
        transactions = [
            tokenizer.tokenize_transaction(row)
            for row in user_df.itertuples(index=False)
        ]

        if len(transactions) < transaction_per_sequence:
            continue

        for start in range(
            0,
            len(transactions) - transaction_per_sequence + 1,
            stride,
        ):
            window = transactions[
                start:start + transaction_per_sequence
            ]

            flattened = []

            for transaction in window:
                flattened.extend(transaction)

            sequences.append(flattened)

            if (
                max_sequence is not None
                and len(sequences) >= max_sequence
            ):
                return sequences

    return sequences


def build_vocabulary(
    train_sequences,
    minimum_frequency=5,
    field_limits=None,
):
    field_counters = defaultdict(Counter)

    for sequence in train_sequences:
        for token in sequence:
            field = get_field(token)

            if field is not None:
                field_counters[field][token] += 1

    if field_limits is None:
        field_limits = FIELD_LIMITS

    vocab_tokens = SPECIAL_TOKENS.copy()
    fields = sorted(field_counters.keys())

    for field in fields:
        vocab_tokens.append(
            f"[UNK_{field}]"
        )

    for field in fields:
        counter = field_counters[field]

        items = [
            (token, count)
            for token, count
            in counter.most_common()
            if count >= minimum_frequency
        ]

        limit = field_limits.get(field)

        if limit is not None:
            items = items[:limit]

        for token, count in items:
            if token not in vocab_tokens:
                vocab_tokens.append(token)

    return {
        token: idx
        for idx, token
        in enumerate(vocab_tokens)
    }

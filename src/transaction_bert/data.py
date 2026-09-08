"""Shared data preparation used by both notebooks."""

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "card_transaction.v1.csv": "user",
    "Card": "card",
    "Year": "year",
    "Month": "month",
    "Day": "day",
    "Time": "time",
    "Amount": "amount",
    "Use Chip": "use_chip",
    "Merchant Name": "merchant_name",
    "Merchant City": "merchant_city",
    "Merchant State": "merchant_state",
    "Zip": "zip",
    "MCC": "mcc",
    "Errors?": "errors",
    "Is Fraud?": "is_fraud",
}


def load_transactions(data_path, nrows=8_000_000):
    """Run the shared loading and feature cells from the notebooks."""
    df = pd.read_csv(
        data_path,
        compression="gzip",
        nrows=nrows,
    )

    df = df.rename(columns=COLUMN_MAP)

    date = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
        )
    )

    time = pd.to_timedelta(
        df["time"].astype(str) + ":00"
    )

    df["timestamp"] = date + time

    df["amount_numeric"] = (
        df["amount"]
        .astype(str)
        .str.replace("$", "")
        .str.replace(",", "")
    )

    df["amount_numeric"] = pd.to_numeric(
        df["amount_numeric"]
    )

    df = df.sort_values(
        ["user", "timestamp"]
    ).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["calendar_month"] = (
        df["timestamp"].dt.month
    )

    df["previous_time"] = (
        df.groupby("user")["timestamp"]
        .diff()
        .dt
        .total_seconds()
        .div(60)
    )

    df["previous_time"] = (
        df["previous_time"]
        .fillna(0)
        .clip(lower=0, upper=60 * 24 * 30)
    )

    return df


def split_data(
    df,
    train_end="2017-01-01",
    validation_end="2019-01-01",
):
    """Use the exact chronological split from both notebooks."""
    train_df = df[
        df["timestamp"] < pd.Timestamp(train_end)
    ].copy()

    val_df = df[
        (df["timestamp"] >= pd.Timestamp(train_end))
        & (df["timestamp"] < pd.Timestamp(validation_end))
    ].copy()

    test_df = df[
        df["timestamp"] >= pd.Timestamp(validation_end)
    ].copy()

    return train_df, val_df, test_df


def fit_quantile_boundaries(values, n_bins):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    boundaries = np.quantile(values, quantiles)
    boundaries = np.unique(boundaries)
    return boundaries


def apply_quantization(values, boundaries):
    values = np.asarray(values, dtype=np.float64)
    ind = np.digitize(values, boundaries, right=False)
    return ind.astype(dtype=np.int32)


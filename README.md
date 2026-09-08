# Transaction BERT for Fraud Detection

Transaction BERT is a representation-learning project for fraud detection in sequential
credit-card data. It first learns transaction representations with a BERT masked-language-model
objective, then uses those representations to classify fraudulent transactions from each user's
recent transaction history.

The project has two training stages:

1. Pretrain a BERT foundation model by predicting masked transaction fields.
2. Use the pretrained encoder with MLP and LSTM heads for fraud classification.

Ready-to-use foundation-model artifacts are available in `artifacts/foundation_model/`, so fraud
classification can be run directly without repeating pretraining.

## Research basis

The implementation is based on Padhi et al., **“Tabular Transformers for Modeling Multivariate
Time Series”** (ICASSP 2021, arXiv:2011.01843v2). The reference paper is stored at
`docs/references/2011.01843v2.pdf`.

The paper treats tabular time-series rows as language-like sequences. Continuous features are
quantized, every field has a finite vocabulary, consecutive transactions are arranged in temporal
windows, and a BERT masked-language-model objective learns representations for downstream fraud
detection.

The model flattens the fields from 16 consecutive transactions into one BERT input. The fraud
classifier pools the BERT field outputs back into transaction representations and applies either an
MLP or an LSTM classification head.

## Repository structure

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- configs/
|   |-- pretrain.json
|   `-- fraud.json
|-- data/
|   `-- raw/
|       `-- transactions.tgz
|-- artifacts/
|   |-- foundation_model/
|   |   |-- config.json
|   |   |-- model.safetensors
|   |   |-- preprocessing.json
|   |   `-- vocab.json
|   `-- notebook_exports/
|       |-- bert.zip
|       `-- bert_preprocess.zip
|-- docs/
|   `-- references/
|       `-- 2011.01843v2.pdf
|-- notebooks/
|   |-- 01_pretrain_foundation_model.ipynb
|   `-- 02_train_fraud_classifier.ipynb
`-- src/
    `-- transaction_bert/
        |-- data.py
        |-- tokenization.py
        |-- foundation.py
        |-- fraud.py
        `-- cli/
            |-- pretrain.py
            `-- train_fraud.py
```

## Code organization

Shared transaction processing is defined once:

- `data.py` contains transaction loading, feature creation, chronological splitting,
  `fit_quantile_boundaries`, and `apply_quantization`.
- `tokenization.py` contains `clean_data`, `clean_zip`, `transaction_to_tokens`, `get_field`,
  `TransactionTokenizer`, `build_sequence`, and vocabulary creation.

The foundation-model components are:

- `FinancialTransactionDataset`
- `FinancialMLMCollator`
- `evaluate_masked_accuracy`

The fraud-classification components are:

- `show_distribution`
- `FraudWindowDataset`
- `FinancialBERTFraudDetector`
- `create_fraud_model`
- `predict`
- `find_best_threshold`
- `train_model`
- `evaluate_model`

The two files under `cli/` execute these components in the required preprocessing and training
order.

## Foundation-model workflow

The foundation command performs the following steps:

1. Read the first 8,000,000 transactions.
2. Rename the original columns.
3. Create the timestamp and numeric amount.
4. Sort transactions by user and timestamp.
5. Create hour, day-of-week, day-of-month, month, and time-delta features.
6. Split before 2017 for training, 2017–2018 for validation, and 2019 onward for testing.
7. Fit 32 amount bins, 64 timestamp bins, and 32 time-delta bins on the training data.
8. Convert each transaction into 16 ordered field tokens.
9. Build 16-transaction windows with stride 8.
10. Build the field vocabulary from the training windows.
11. Apply the same 15% field-aware BERT masking.
12. Train the four-layer `BertForMaskedLM` for 12 epochs.
13. Evaluate masked-token accuracy and save the model, vocabulary, and preprocessing boundaries.

The fraud label is not included in the foundation-model tokens.

## Fraud-classifier workflow

The fraud command performs the following steps:

1. Repeat the same shared transaction preparation.
2. Load `vocab.json` and `preprocessing.json` from the pretrained foundation model.
3. Apply the saved amount, timestamp, and time-delta boundaries.
4. Encode every transaction to `encoded_transactions.uint16.mmap`.
5. Build fraud targets only where 15 earlier transactions exist for the same user.
6. Apply the same chronological train, validation, and test dates.
7. Keep all fraud examples and sample normal transactions at a 10:1 ratio.
8. Use `BCEWithLogitsLoss` with the same positive-class weight.
9. Select the fraud threshold from validation F1 and report results on the test period.

The classifier experiments are run in this order:

1. Current transaction BERT representation with an MLP.
2. Mean transaction representation with an MLP.
3. Frozen BERT with a transaction LSTM.
4. LSTM with the last BERT layer fine-tuned.
5. LSTM with the last two BERT layers fine-tuned.
6. Continue the two-layer model for a longer 12-epoch run.

The frozen BERT-LSTM state is saved as
`artifacts/fraud_classifier/frozen_bert_lstm_fraud.pt`.

## Installation

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

## Run the code

The ready-to-use foundation model makes the first command optional.

To rebuild the foundation model:

```powershell
python -m transaction_bert.cli.pretrain --config configs/pretrain.json
```

To run the complete fraud-classification workflow with the pretrained model:

```powershell
python -m transaction_bert.cli.train_fraud --config configs/fraud.json
```

Both configurations contain the reference experiment values. Run the commands from the repository
root so the data and artifact paths resolve correctly.

## Reference experiment results

The foundation-model experiment reported:

- vocabulary size: 22,223;
- model parameters: 3.71 million;
- validation MLM loss: 0.63094; and
- masked-token accuracy: 0.81378.

The fraud-classification experiments reported:

| Model | Test AP | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Current transaction + MLP | 0.05134 | 0.96560 | 0.11275 | 0.09729 | 0.10445 |
| Whole-sequence mean + MLP | 0.00540 | 0.76459 | 0.02281 | 0.02871 | 0.02542 |
| Frozen BERT + transaction LSTM | 0.08751 | 0.96176 | 0.13355 | 0.25837 | 0.17609 |
| Last BERT layer + LSTM | 0.11612 | 0.98440 | 0.16077 | 0.31898 | 0.21379 |
| Last two BERT layers + LSTM | 0.21015 | 0.98480 | 0.28852 | 0.32855 | 0.30723 |

These are the recorded reference results, not results from a new training run.

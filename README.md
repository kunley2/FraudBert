# Transaction BERT for Fraud Detection

## Overview

This project uses BERT representation learning to detect fraud in sequential credit-card
transactions. It learns relationships between transaction fields and patterns across a user's
recent transaction history.

The training pipeline has two stages:

1. **Foundation-model pretraining:** transaction fields are converted to tokens and a compact BERT
   model is trained with masked-language modeling.
2. **Fraud classification:** the pretrained encoder produces transaction representations that are
   passed to MLP and LSTM classification heads.

Ready-to-use foundation-model weights, vocabulary, and preprocessing boundaries are included in
`artifacts/foundation_model/`. The foundation model is also available on Hugging Face:
[kunley2/FinBERT](https://huggingface.co/kunley2/FinBERT).

## Method

Transactions are sorted chronologically for each user. Continuous features—amount, timestamp, and
time since the previous transaction—are quantized using boundaries fitted on the training period.
Each transaction becomes 16 ordered tokens containing card, time, amount, channel, merchant,
location, MCC, and error information.

Sixteen consecutive transactions form one BERT sequence. The foundation model masks 15% of the
field tokens and learns to predict them. For fraud detection, the BERT outputs are pooled into one
representation per transaction and evaluated with:

- current-transaction MLP;
- whole-sequence mean MLP;
- frozen BERT with an LSTM;
- LSTM with the final one or two BERT layers fine-tuned.

Training uses data before 2017, validation data from 2017–2018, and test data from 2019 onward.

## Project structure

```text
configs/                    Training configurations
data/raw/                   Transaction dataset
artifacts/foundation_model/ Pretrained BERT model, vocabulary, and quantizers
docs/references/            Research paper
notebooks/                  Foundation and fraud experiment notebooks
src/transaction_bert/
  data.py                   Loading, features, splits, and quantization
  tokenization.py           Transaction tokens, vocabulary, and tokenizer
  foundation.py             MLM dataset, collator, and evaluation
  fraud.py                  Fraud datasets, model heads, training, and evaluation
  cli/pretrain.py           Foundation-model workflow
  cli/train_fraud.py        Fraud-classification workflow
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

The foundation model is already included, so pretraining is optional.

```powershell
# Optional: rebuild the foundation model
python -m transaction_bert.cli.pretrain --config configs/pretrain.json

# Train and evaluate all fraud classifiers
python -m transaction_bert.cli.train_fraud --config configs/fraud.json
```

Run both commands from the repository root. The full configuration processes eight million rows
and is intended for a CUDA-capable machine.

## Research basis

The approach is based on Padhi et al., **“Tabular Transformers for Modeling Multivariate Time
Series”** (ICASSP 2021, arXiv:2011.01843v2). The paper is available at
`docs/references/2011.01843v2.pdf`.

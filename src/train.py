"""
Script di training / retraining del modello di sentiment analysis.

Effettua il fine-tuning del modello pre-addestrato
`cardiffnlp/twitter-roberta-base-sentiment-latest` sul dataset pubblico
`tweet_eval` (subset "sentiment"), utilizzabile sia per il training iniziale
sia per il retraining periodico su nuovi dati etichettati.

Uso:
    python -m src.train --epochs 1 --output_dir ./retrained_model
    python -m src.train --data_path ./data/new_labeled_data.csv  # retraining su nuovi dati
"""

from __future__ import annotations

import argparse
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]


def load_training_data(data_path: str | None):
    """Carica il dataset pubblico tweet_eval, oppure un CSV custom con nuovi
    dati etichettati (colonne: text,label) per il retraining incrementale."""
    if data_path:
        import pandas as pd

        df = pd.read_csv(data_path)
        return Dataset.from_pandas(df)
    ds = load_dataset("tweet_eval", "sentiment")
    return ds["train"], ds["validation"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None,
                         help="CSV opzionale (text,label) per retraining su nuovi dati")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./retrained_model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS)
    )

    train_ds, eval_ds = load_training_data(args.data_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Metriche finali di retraining:", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Modello ri-addestrato salvato in: {args.output_dir}")


if __name__ == "__main__":
    main()

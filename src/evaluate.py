"""
Valutazione del modello pre-addestrato sul dataset pubblico `tweet_eval`
(subset "sentiment"). Calcola accuracy, F1-macro e confusion matrix.

Uso:
    python -m src.evaluate --n_samples 500
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.sentiment_model import SentimentAnalyzer

# Il modello cardiffnlp usa questo ordine di etichette
MODEL_LABELS = ["negative", "neutral", "positive"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument(
        "--n_samples", type=int, default=300,
        help="Numero di esempi da valutare (per velocità in demo)",
    )
    args = parser.parse_args()

    # Il dataset "tweet_eval" senza namespace non è più caricabile: HF ha
    # rimosso il loading script dal repo legacy, il dataset ora vive sotto
    # il namespace cardiffnlp (vedi https://huggingface.co/datasets/cardiffnlp/tweet_eval).
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")[args.split]
    if args.n_samples:
        ds = ds.select(range(min(args.n_samples, len(ds))))

    analyzer = SentimentAnalyzer()
    results = analyzer.predict(list(ds["text"]))

    y_true = list(ds["label"])  # 0=negative,1=neutral,2=positive nel dataset tweet_eval
    y_pred = [MODEL_LABELS.index(r.label) for r in results]

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy su {len(ds)} esempi ({args.split}): {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=MODEL_LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

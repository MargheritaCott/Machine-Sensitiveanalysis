"""
Wrapper per il modello pre-addestrato di sentiment analysis.

Modello: cardiffnlp/twitter-roberta-base-sentiment-latest
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass
class SentimentResult:
    text: str
    label: str  # "negative" | "neutral" | "positive"
    score: float


class SentimentAnalyzer:
    """Carica il modello pre-addestrato ed espone un metodo di inferenza."""

    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def predict(self, texts: List[str]) -> List[SentimentResult]:
        if isinstance(texts, str):
            texts = [texts]
        raw = self.pipe(texts, truncation=True)
        return [
            SentimentResult(text=t, label=r["label"].lower(), score=float(r["score"]))
            for t, r in zip(texts, raw)
        ]


def main():
    if len(sys.argv) < 2:
        print('Uso: python -m src.sentiment_model "testo da analizzare"')
        sys.exit(1)
    text = " ".join(sys.argv[1:])
    analyzer = SentimentAnalyzer()
    for res in analyzer.predict([text]):
        print(f"Testo: {res.text}")
        print(f"Sentiment: {res.label} (score={res.score:.4f})")


if __name__ == "__main__":
    main()

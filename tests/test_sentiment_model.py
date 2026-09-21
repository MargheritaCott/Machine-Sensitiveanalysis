"""
Test unitari e di integrazione usati dalla pipeline CI/CD.

NB: i testi di esempio sono in inglese perché
cardiffnlp/twitter-roberta-base-sentiment-latest è addestrato esclusivamente
su tweet in inglese. Con testo in italiano il modello collassa quasi sempre
su "neutral" indipendentemente dal contenuto, il che farebbe fallire questi
test per un motivo linguistico e non per un problema del codice.
"""

import pytest

from src.sentiment_model import SentimentAnalyzer, SentimentResult

VALID_LABELS = {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_predict_returns_valid_schema(analyzer):
    results = analyzer.predict(["I love this product!"])
    assert len(results) == 1
    assert isinstance(results[0], SentimentResult)
    assert results[0].label in VALID_LABELS
    assert 0.0 <= results[0].score <= 1.0


def test_predict_batch(analyzer):
    texts = [
        "I'm very happy with the service",
        "Terrible experience, I do not recommend it",
        "It's fine, nothing special",
    ]
    results = analyzer.predict(texts)
    assert len(results) == len(texts)
    for r in results:
        assert r.label in VALID_LABELS


def test_positive_example_classified_correctly(analyzer):
    result = analyzer.predict(["Excellent service, super friendly staff, highly recommended!"])[0]
    assert result.label == "positive"


def test_negative_example_classified_correctly(analyzer):
    result = analyzer.predict(["Terrible service, the product arrived broken, I do not recommend it to anyone"])[0]
    assert result.label == "negative"


def test_empty_string_handled(analyzer):
    results = analyzer.predict([" "])
    assert len(results) == 1

"""
Test unitari e di integrazione usati dalla pipeline CI/CD.
"""

import pytest

from src.sentiment_model import SentimentAnalyzer, SentimentResult

VALID_LABELS = {"negative", "neutral", "positive"}


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_predict_returns_valid_schema(analyzer):
    results = analyzer.predict(["Adoro questo prodotto!"])
    assert len(results) == 1
    assert isinstance(results[0], SentimentResult)
    assert results[0].label in VALID_LABELS
    assert 0.0 <= results[0].score <= 1.0


def test_predict_batch(analyzer):
    texts = [
        "Sono molto soddisfatto del servizio",
        "Esperienza pessima, non consiglio",
        "Ok, niente di particolare",
    ]
    results = analyzer.predict(texts)
    assert len(results) == len(texts)
    for r in results:
        assert r.label in VALID_LABELS


def test_positive_example_classified_correctly(analyzer):
    result = analyzer.predict(["Servizio eccellente, staff gentilissimo, consigliatissimo!"])[0]
    assert result.label == "positive"


def test_negative_example_classified_correctly(analyzer):
    result = analyzer.predict(["Pessimo servizio, prodotto rotto, non lo consiglio a nessuno"])[0]
    assert result.label == "negative"


def test_empty_string_handled(analyzer):
    results = analyzer.predict([" "])
    assert len(results) == 1

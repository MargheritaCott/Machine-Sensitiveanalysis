"""
Sistema di monitoraggio continuo della reputazione e delle performance del
modello.

Responsabilità:
1. Elaborare un flusso di post social (batch periodici) e calcolare la
   distribuzione di sentiment (positivo/neutro/negativo) nel tempo.
2. Salvare le metriche in un log strutturato (CSV) per dashboard/audit.
3. Rilevare "drift" nella distribuzione dei sentiment rispetto a una
   baseline, o un calo di accuracy su un set di validazione periodico.
4. Se il drift supera una soglia, generare un trigger che avvia il
   retraining (src/train.py).

Uso:
    python -m src.monitor
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np

from src.sentiment_model import SentimentAnalyzer

LOG_PATH = Path("./monitoring_log.csv")
DRIFT_THRESHOLD = 0.15  # scostamento massimo accettabile nella distribuzione


@dataclass
class MonitoringSnapshot:
    timestamp: str
    n_posts: int
    pct_negative: float
    pct_neutral: float
    pct_positive: float
    drift_score: float
    retrain_triggered: bool


def compute_distribution(labels: List[str]) -> dict:
    n = len(labels)
    return {
        "negative": labels.count("negative") / n,
        "neutral": labels.count("neutral") / n,
        "positive": labels.count("positive") / n,
    }


def compute_drift(baseline: dict, current: dict) -> float:
    """Total variation distance tra due distribuzioni di sentiment."""
    return 0.5 * sum(abs(baseline[k] - current[k]) for k in baseline)


def log_snapshot(snapshot: MonitoringSnapshot):
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshot.__dict__.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(snapshot.__dict__)


def trigger_retraining():
    print(">>> Drift rilevato oltre soglia: avvio retraining automatico...")
    # In produzione: subprocess.run(["python", "-m", "src.train", "--data_path", "data/new_labeled_data.csv"])
    print(">>> (demo) retraining simulato completato.")


def run_monitoring_cycle(batches: List[List[str]], baseline_distribution: dict | None = None):
    """Esegue il monitoraggio su una sequenza di batch di post social,
    simulando l'arrivo di dati nel tempo."""
    analyzer = SentimentAnalyzer()
    baseline = baseline_distribution
    now = datetime.utcnow()

    for i, batch in enumerate(batches):
        results = analyzer.predict(batch)
        labels = [r.label for r in results]
        dist = compute_distribution(labels)

        if baseline is None:
            baseline = dist  # il primo batch diventa la baseline
            drift = 0.0
        else:
            drift = compute_drift(baseline, dist)

        retrain = drift > DRIFT_THRESHOLD
        snapshot = MonitoringSnapshot(
            timestamp=(now + timedelta(hours=i)).isoformat(),
            n_posts=len(batch),
            pct_negative=round(dist["negative"], 3),
            pct_neutral=round(dist["neutral"], 3),
            pct_positive=round(dist["positive"], 3),
            drift_score=round(drift, 3),
            retrain_triggered=retrain,
        )
        log_snapshot(snapshot)
        print(snapshot)

        if retrain:
            trigger_retraining()
            baseline = dist  # dopo il retraining, la nuova distribuzione diventa la baseline


if __name__ == "__main__":
    # Esempio dimostrativo: 4 batch di post social simulati nel tempo,
    # con un evento negativo improvviso nel batch 3 per mostrare il drift detection.
    demo_batches = [
        ["Adoro questo prodotto!", "Ottimo servizio clienti", "Consegna puntuale come sempre",
         "Niente di che ma va bene", "Prodotto nella media"],
        ["Il supporto è stato gentile", "Buona qualità costruttiva", "App un po' lenta ma utile",
         "Prezzo giusto", "Consiglierei ad un amico"],
        ["Servizio pessimo, ho aspettato ore", "Prodotto rotto all'arrivo", "Mai più un acquisto qui",
         "Assistenza clienti inesistente", "Delusione totale, sconsigliato"],
        ["Servizio pessimo di nuovo", "Ancora problemi con la spedizione", "App si blocca sempre",
         "Rimborso mai arrivato", "Esperienza negativa"],
    ]
    run_monitoring_cycle(demo_batches)
    print(f"\nLog di monitoraggio salvato in: {LOG_PATH.resolve()}")

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


def trigger_retraining(data_path: str = "data/new_labeled_data.csv", dry_run: bool = False):
    """Avvia il retraining del modello (src/train.py) sui nuovi dati etichettati.

    dry_run=True (default nella demo di questo modulo se non diversamente
    specificato dal chiamante) stampa il comando senza eseguirlo, utile per
    non lanciare un training reale da una semplice demo locale; in un
    ambiente di produzione/CI va invocato con dry_run=False.
    """
    print(">>> Drift rilevato oltre soglia: avvio retraining automatico...")
    cmd = ["python", "-m", "src.train", "--data_path", data_path, "--epochs", "1"]
    if dry_run:
        print(f">>> (dry run) comando che verrebbe eseguito: {' '.join(cmd)}")
        return
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(">>> Retraining fallito:")
        print(result.stderr)
    else:
        print(">>> Retraining completato con successo.")
        print(result.stdout)


def run_monitoring_cycle(
    batches: List[List[str]],
    baseline_distribution: dict | None = None,
    retrain_dry_run: bool = True,
):
    """Esegue il monitoraggio su una sequenza di batch di post social,
    simulando l'arrivo di dati nel tempo.

    retrain_dry_run=True (default) stampa il comando di retraining senza
    eseguirlo realmente: utile in demo locali dove data/new_labeled_data.csv
    non esiste ancora. In produzione/CI va passato retrain_dry_run=False.
    """
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
            trigger_retraining(dry_run=retrain_dry_run)
            baseline = dist  # dopo il retraining, la nuova distribuzione diventa la baseline


if __name__ == "__main__":
    # Esempio dimostrativo: 4 batch di post social simulati nel tempo,
    # con un evento negativo improvviso nel batch 3 per mostrare il drift
    # detection. NOTA: il modello cardiffnlp/twitter-roberta-base-sentiment-
    # latest è addestrato su tweet in INGLESE; testi in italiano collassano
    # quasi sempre su "neutral" e la demo non mostrerebbe alcun drift. I post
    # sono quindi in inglese per riflettere realisticamente il comportamento
    # del modello (in produzione, su testi in italiano, andrebbe usato un
    # modello multilingue, es. cardiffnlp/twitter-xlm-roberta-base-sentiment).
    demo_batches = [
        ["I love this product!", "Great customer service", "Delivery was on time as always",
         "Nothing special but it's fine", "Average product, does the job"],
        ["Support staff was very kind", "Good build quality", "App is a bit slow but useful",
         "Fair price for what you get", "Would recommend it to a friend"],
        ["Terrible service, waited for hours", "Product arrived broken", "Never buying from here again",
         "Customer support is nonexistent", "Total disappointment, do not recommend"],
        ["Terrible service again", "Still having shipping problems", "App keeps crashing constantly",
         "Refund never arrived", "Overall a really negative experience"],
    ]
    # dry_run=True: nella demo locale non è presente data/new_labeled_data.csv,
    # quindi il retraining viene solo mostrato, non eseguito realmente.
    # In CI/produzione, con dati etichettati disponibili, impostare False.
    run_monitoring_cycle(demo_batches, retrain_dry_run=True)
    print(f"\nLog di monitoraggio salvato in: {LOG_PATH.resolve()}")

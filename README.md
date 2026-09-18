# MachineInnovators Inc. — Sentiment Analysis & Social Media Reputation Monitoring

Sistema MLOps end-to-end per l'analisi automatica del sentiment sui social media,
il monitoraggio continuo della reputazione aziendale e il retraining automatico
del modello.

## 1. Obiettivo del progetto

MachineInnovators Inc. vuole automatizzare il monitoraggio della propria
reputazione online tramite un modello di sentiment analysis che classifica i
testi social in **positivo / neutro / negativo**, con:

- pipeline CI/CD per test e deploy automatico,
- monitoraggio continuo delle performance del modello e del sentiment rilevato,
- retraining automatico del modello quando le performance degradano o arrivano
  nuovi dati etichettati.

## 2. Architettura del progetto

```

├── src/
│   ├── sentiment_model.py   # Wrapper del modello pre-addestrato (inferenza)
│   ├── train.py             # Script di training / retraining
│   ├── evaluate.py          # Calcolo metriche (accuracy, F1, confusion matrix)
│   └── monitor.py           # Monitoraggio continuo + drift detection
├── tests/
│   └── test_sentiment_model.py
├── .github/workflows/
│   └── ci-cd.yml            # Pipeline CI/CD (test, build, deploy su HF Hub)
├── data/
│   └── README.md            # Note sul dataset pubblico utilizzato
├── requirements.txt
└── README.md
```

## 3. Fase 1 — Modello di Sentiment Analysis

Si utilizza il modello pre-addestrato
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
basato su RoBERTa e fine-tuned su ~124M tweet, che classifica un testo in
`negative`, `neutral`, `positive`.

**Motivazione della scelta:** è addestrato specificamente su testo da social
media (linguaggio informale, hashtag, emoji, abbreviazioni), il che lo rende
più adatto rispetto a un modello generico addestrato su recensioni o news.

**Dataset pubblico:** per validazione e per il retraining periodico si usa il
dataset [`tweet_eval` (subset `sentiment`)](https://huggingface.co/datasets/tweet_eval),
che contiene tweet etichettati con le stesse 3 classi, garantendo coerenza
con il task del modello.

## 4. Fase 2 — Pipeline CI/CD

La pipeline (`.github/workflows/ci-cd.yml`) si attiva ad ogni push/PR sul
branch `main` ed esegue:

1. **Lint & unit test** (`pytest`) sul codice di inferenza e sulle metriche.
2. **Integration test**: esecuzione del modello su un piccolo batch di frasi
   di esempio, verifica che l'output rispetti lo schema atteso (etichetta +
   score).
3. **Build**: verifica che le dipendenze si installino correttamente.
4. **Deploy (facoltativo, su tag di release)**: push del modello/app su
   Hugging Face Hub / Spaces tramite `huggingface_hub`, usando un secret
   `HF_TOKEN` configurato nel repository GitHub.

## 5. Fase 3 — Deploy e Monitoraggio continuo

- **Deploy (facoltativo):** l'app (`src/app.py`, interfaccia Gradio) e il
  modello possono essere pubblicati come Hugging Face Space, per esporre
  un endpoint pubblico di inferenza.
- **Monitoraggio (`src/monitor.py`):** simula/ingerisce un flusso di post
  social, calcola la distribuzione di sentiment nel tempo, salva metriche in
  un log strutturato (CSV/JSON) e implementa una semplice **drift detection**
  basata sullo scostamento della distribuzione delle predizioni (o
  sull'accuracy su un set di validazione periodico) rispetto a una baseline.
- **Retraining automatico:** quando il drift/la degradazione supera una
  soglia, `monitor.py` genera un trigger che avvia `train.py` per
  ri-addestrare (fine-tuning) il modello sui nuovi dati etichettati,
  chiudendo il ciclo MLOps.

## 6. Risultati

Il notebook di consegna (`https://colab.research.google.com/drive/1gSPyNzd0Rhfa28qoCNsBv509RhVbddYn?usp=sharing`)
riporta:
- esempi di inferenza su testi tipici da social media,
- metriche di valutazione del modello pre-addestrato sul dataset `tweet_eval`
  (accuracy, F1-score, confusion matrix),
- una dimostrazione del sistema di monitoraggio con rilevazione di drift
  simulata e conseguente trigger di retraining,
- il link alla repository GitHub con il codice sorgente completo.

## 7. Come eseguire il progetto

```bash
git clone https://github.com/MargheritaCott/Machine-Sensitiveanalysis
cd repo
pip install -r requirements.txt

# Inferenza rapida
python -m src.sentiment_model "Il nuovo prodotto è fantastico!"

# Valutazione sul dataset pubblico
python -m src.evaluate

# Simulazione monitoraggio + retraining
python -m src.monitor
```



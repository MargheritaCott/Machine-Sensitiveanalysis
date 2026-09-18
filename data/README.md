# Dataset

Il progetto utilizza il dataset pubblico **tweet_eval** (subset `sentiment`),
disponibile su Hugging Face Datasets:
https://huggingface.co/datasets/tweet_eval

Contiene tweet in inglese etichettati con 3 classi coerenti con l'output del
modello `cardiffnlp/twitter-roberta-base-sentiment-latest`:

| Label id | Significato |
|----------|-------------|
| 0        | negative    |
| 1        | neutral     |
| 2        | positive    |

Viene caricato direttamente via `datasets.load_dataset("tweet_eval", "sentiment")`,
quindi non è necessario scaricare manualmente alcun file: questa cartella resta
come riferimento e come posto dove salvare eventuali nuovi dati etichettati
raccolti dai social media aziendali per il retraining incrementale
(es. `new_labeled_data.csv` con colonne `text,label`).

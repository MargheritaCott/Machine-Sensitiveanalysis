"""
Uso locale:
    python -m src.app
"""

import gradio as gr

from src.sentiment_model import SentimentAnalyzer

analyzer = SentimentAnalyzer()


def classify(text: str):
    if not text or not text.strip():
        return {}
    result = analyzer.predict([text])[0]
    return {result.label: result.score}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Testo da un post/commento social", lines=3),
    outputs=gr.Label(label="Sentiment"),
    title="MachineInnovators — Social Media Sentiment Monitor",
    description=(
        "Analizza il sentiment (positivo / neutro / negativo) di un testo "
        "proveniente dai social media, usando cardiffnlp/twitter-roberta-base-sentiment-latest."
    ),
    examples=[
        "Adoro il nuovo prodotto, funziona benissimo!",
        "Il servizio clienti non ha risolto il mio problema.",
        "Consegna nella media, niente di speciale.",
    ],
)

if __name__ == "__main__":
    demo.launch()

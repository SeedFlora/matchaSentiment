from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from matcha_sentiment.config import ARTIFACT_DIR, ID2LABEL, MODEL_DIR
from matcha_sentiment.text import normalize_text


ROOT = Path(__file__).resolve().parent
FIG_DIR = ARTIFACT_DIR / "figures"
TRANSFORMER_MODEL_PATH = Path(os.getenv("MODEL_DIR", MODEL_DIR / "best_transformer"))
CLASSICAL_MODEL_PATH = Path(os.getenv("CLASSICAL_MODEL_PATH", MODEL_DIR / "classical" / "best_model.joblib"))
MODEL_ID = os.getenv("MODEL_ID", "")


class Predictor:
    def __init__(self):
        self.kind = "none"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.classical = None
        self.load()

    def load(self) -> None:
        model_source = None
        if (TRANSFORMER_MODEL_PATH / "config.json").exists():
            model_source = str(TRANSFORMER_MODEL_PATH)
        elif MODEL_ID:
            model_source = MODEL_ID

        if model_source:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
            self.model.to(self.device)
            self.model.eval()
            self.kind = "transformer"
            return

        if CLASSICAL_MODEL_PATH.exists():
            self.classical = joblib.load(CLASSICAL_MODEL_PATH)
            self.kind = "classical"

    def predict(self, text: str) -> tuple[dict[str, float], str]:
        text = normalize_text(text)
        if not text:
            return {"Negatif": 0.0, "Positif": 0.0}, "Masukkan teks review."

        if self.kind == "transformer":
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=160,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            scores = {ID2LABEL[idx]: float(probs[idx]) for idx in range(len(probs))}
            label = ID2LABEL[int(np.argmax(probs))]
            return scores, f"{label} - {self.kind} on {self.device.type}"

        if self.kind == "classical":
            pred = int(self.classical.predict([text])[0])
            if hasattr(self.classical, "predict_proba"):
                proba = self.classical.predict_proba([text])
                if proba is not None:
                    probs = proba[0]
                    return {ID2LABEL[idx]: float(probs[idx]) for idx in range(len(probs))}, f"{ID2LABEL[pred]} - classical"
            if hasattr(self.classical, "decision_function"):
                score = self.classical.decision_function([text])
                if score is not None:
                    p_pos = 1.0 / (1.0 + np.exp(-float(np.ravel(score)[0])))
                    return {"Negatif": 1.0 - p_pos, "Positif": p_pos}, f"{ID2LABEL[pred]} - classical"
            return {ID2LABEL[pred]: 1.0}, f"{ID2LABEL[pred]} - classical"

        return {"Negatif": 0.0, "Positif": 0.0}, "Model belum tersedia."


predictor = Predictor()


def predict_review(text: str):
    return predictor.predict(text)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def image_value(name: str):
    path = FIG_DIR / name
    return str(path) if path.exists() else None


summary = read_json(ROOT / "data" / "processed" / "summary.json")
classical_results = read_csv(ARTIFACT_DIR / "classical" / "results.csv")
transformer_results = read_csv(ARTIFACT_DIR / "transformers" / "results.csv")
top_words = read_csv(ARTIFACT_DIR / "classical" / "top_words_tfidf.csv")
keyword_counts = read_csv(ARTIFACT_DIR / "classical" / "keyword_counts.csv")


css = """
.metric-card textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.gradio-container { max-width: 1180px !important; }
"""


with gr.Blocks(title="Matcha Sentiment", css=css) as demo:
    gr.Markdown("# Matcha Sentiment")

    with gr.Tab("Prediksi"):
        with gr.Row():
            review = gr.Textbox(
                label="Review",
                lines=7,
                value="Matchanya enak, tempatnya nyaman, tapi harganya agak mahal.",
            )
            with gr.Column():
                output_label = gr.Label(label="Sentimen")
                output_text = gr.Textbox(label="Model", interactive=False)
                submit = gr.Button("Analisis", variant="primary")
        submit.click(predict_review, inputs=review, outputs=[output_label, output_text])
        gr.Examples(
            examples=[
                ["Matchanya enak dan pelayanannya ramah."],
                ["Harganya terlalu mahal dan rasanya biasa saja."],
                ["Tempat nyaman, tetapi antrean lama dan staf kurang ramah."],
            ],
            inputs=review,
        )

    with gr.Tab("Metrik"):
        with gr.Row():
            gr.JSON(value=summary, label="Dataset")
        with gr.Row():
            gr.Dataframe(value=classical_results, label="TF-IDF dan Word2Vec 10-fold", interactive=False)
        with gr.Row():
            gr.Dataframe(value=transformer_results, label="Transformer", interactive=False)

    with gr.Tab("Visual"):
        with gr.Row():
            gr.Image(value=image_value("transformer_best_training_loss.png"), label="Training loss")
            gr.Image(value=image_value("transformer_best_confusion_matrix.png"), label="Confusion matrix transformer")
        with gr.Row():
            gr.Image(value=image_value("transformer_best_roc_auc.png"), label="ROC AUC transformer")
            gr.Image(value=image_value("classical_best_confusion_matrix.png"), label="Confusion matrix klasik")
        with gr.Row():
            gr.Image(value=image_value("classical_best_roc_auc.png"), label="ROC AUC klasik")
            gr.Image(value=image_value("top_words_tfidf.png"), label="Top words")
        with gr.Row():
            gr.Image(value=image_value("wordcloud_positif.png"), label="Word cloud positif")
            gr.Image(value=image_value("wordcloud_negatif.png"), label="Word cloud negatif")

    with gr.Tab("Kata Kunci"):
        gr.Dataframe(value=top_words, label="Top words TF-IDF", interactive=False)
        gr.Dataframe(value=keyword_counts, label="Keyword penting", interactive=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))

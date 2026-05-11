from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .text import tokenize, tokenized_documents


def tfidf_tokenizer(text: str) -> list[str]:
    return tokenize(text, remove_stopwords=False, min_len=2)


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.toarray() if hasattr(x, "toarray") else x


def vectorize_with_w2v(model, texts: list[str], vector_size: int) -> np.ndarray:
    docs = tokenized_documents(texts, remove_stopwords=False)
    matrix = np.zeros((len(docs), vector_size), dtype=np.float32)
    for row_idx, tokens in enumerate(docs):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            matrix[row_idx] = np.mean(vectors, axis=0)
    return matrix


@dataclass
class W2VBundle:
    word2vec: object
    classifier: object
    vector_size: int

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict(vectorize_with_w2v(self.word2vec, texts, self.vector_size))

    def predict_proba(self, texts: list[str]):
        features = vectorize_with_w2v(self.word2vec, texts, self.vector_size)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(features)
        return None

    def decision_function(self, texts: list[str]):
        features = vectorize_with_w2v(self.word2vec, texts, self.vector_size)
        if hasattr(self.classifier, "decision_function"):
            return self.classifier.decision_function(features)
        return None

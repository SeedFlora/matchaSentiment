import re
import unicodedata
from typing import Iterable

from .config import STOPWORDS


TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+(?:[-'][a-zA-ZÀ-ÿ]+)?")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(value: object) -> str:
    label = normalize_text(value).lower()
    if label == "positif":
        return "Positif"
    if label == "negatif":
        return "Negatif"
    if label == "netral":
        return "Netral"
    return normalize_text(value)


def tokenize(text: str, *, remove_stopwords: bool = False, min_len: int = 2) -> list[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(normalize_text(text))]
    tokens = [t for t in tokens if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def tokenized_documents(texts: Iterable[str], *, remove_stopwords: bool = False) -> list[list[str]]:
    return [tokenize(text, remove_stopwords=remove_stopwords) for text in texts]


def compact_for_key(text: str) -> str:
    return " ".join(tokenize(text, remove_stopwords=False, min_len=1))

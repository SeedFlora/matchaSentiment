from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from matcha_sentiment.classical import DenseTransformer, W2VBundle, tfidf_tokenizer, vectorize_with_w2v
from matcha_sentiment.config import ARTIFACT_DIR, DATA_PATH, ID2LABEL, MODEL_DIR, STOPWORDS
from matcha_sentiment.data import load_binary_dataset
from matcha_sentiment.metrics import binary_metrics, report_dict
from matcha_sentiment.plots import plot_confusion, plot_roc_curve, plot_top_words, write_wordcloud
from matcha_sentiment.text import tokenize, tokenized_documents


ROOT = Path(__file__).resolve().parents[1]

KEYWORD_SEEDS = [
    "enak",
    "lezat",
    "nikmat",
    "nyaman",
    "ramah",
    "bagus",
    "terbaik",
    "mantap",
    "autentik",
    "direkomendasikan",
    "murah",
    "mahal",
    "harga",
    "harganya",
    "buruk",
    "kecewa",
    "lama",
    "antrean",
    "menunggu",
    "kurang",
    "tidak",
    "biasa",
    "pahit",
    "manis",
]


def make_tfidf(max_features: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=tfidf_tokenizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def classifiers(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def positive_score(model, features):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return None


def import_word2vec():
    try:
        from gensim.models import Word2Vec
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        raise RuntimeError(
            "Word2Vec needs gensim and compatible scipy. Install requirements.txt or use the Docker image."
        ) from exc
    return Word2Vec


def train_word2vec(texts: list[str], *, vector_size: int, random_state: int):
    Word2Vec = import_word2vec()
    docs = tokenized_documents(texts, remove_stopwords=False)
    return Word2Vec(
        sentences=docs,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        seed=random_state,
        epochs=60,
    )

def make_tfidf_pipeline(estimator, model_name: str, max_features: int, min_df: int) -> Pipeline:
    steps = [("tfidf", make_tfidf(max_features=max_features, min_df=min_df))]
    if model_name == "gradient_boosting":
        steps.append(("dense", DenseTransformer()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def evaluate_feature_model(
    *,
    feature_name: str,
    model_name: str,
    estimator,
    texts: np.ndarray,
    y: np.ndarray,
    folds: int,
    random_state: int,
    max_features: int,
    min_df: int,
    w2v_size: int,
    out_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof_pred = np.zeros_like(y)
    oof_score = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(texts, y), start=1):
        train_texts = texts[train_idx].tolist()
        valid_texts = texts[valid_idx].tolist()
        y_train, y_valid = y[train_idx], y[valid_idx]
        clf = clone(estimator)

        if feature_name == "tfidf":
            model = make_tfidf_pipeline(clf, model_name, max_features, min_df)
            model.fit(train_texts, y_train)
            pred = model.predict(valid_texts)
            score = positive_score(model, valid_texts)
        elif feature_name == "word2vec":
            w2v = train_word2vec(train_texts, vector_size=w2v_size, random_state=random_state + fold)
            x_train = vectorize_with_w2v(w2v, train_texts, w2v_size)
            x_valid = vectorize_with_w2v(w2v, valid_texts, w2v_size)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_valid)
            score = positive_score(clf, x_valid)
        else:
            raise ValueError(f"Unsupported feature: {feature_name}")

        oof_pred[valid_idx] = pred
        if score is not None:
            oof_score[valid_idx] = score

        fold_metrics = binary_metrics(
            y_valid,
            pred,
            None if score is None else score,
        )
        fold_metrics.update(
            {
                "feature": feature_name,
                "model": model_name,
                "fold": fold,
                "n_valid": int(len(valid_idx)),
            }
        )
        fold_rows.append(fold_metrics)

    usable_score = None if np.isnan(oof_score).any() else oof_score
    aggregate = binary_metrics(y, oof_pred, usable_score)
    aggregate.update(
        {
            "feature": feature_name,
            "model": model_name,
            "folds": folds,
            "n": int(len(y)),
        }
    )

    pred_frame = pd.DataFrame(
        {
            "text": texts,
            "label": y,
            "label_name": [ID2LABEL[int(v)] for v in y],
            "prediction": oof_pred,
            "prediction_name": [ID2LABEL[int(v)] for v in oof_pred],
            "score": oof_score,
        }
    )
    pred_frame.to_csv(out_dir / f"oof_{feature_name}_{model_name}.csv", index=False)
    (out_dir / f"report_{feature_name}_{model_name}.json").write_text(
        json.dumps(report_dict(y, oof_pred), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate, pd.DataFrame(fold_rows)


def fit_final_model(
    *,
    feature_name: str,
    model_name: str,
    estimator,
    texts: list[str],
    y: np.ndarray,
    random_state: int,
    max_features: int,
    min_df: int,
    w2v_size: int,
):
    if feature_name == "tfidf":
        model = make_tfidf_pipeline(clone(estimator), model_name, max_features, min_df)
        model.fit(texts, y)
        return model

    w2v = train_word2vec(texts, vector_size=w2v_size, random_state=random_state)
    features = vectorize_with_w2v(w2v, texts, w2v_size)
    clf = clone(estimator)
    clf.fit(features, y)
    return W2VBundle(word2vec=w2v, classifier=clf, vector_size=w2v_size)


def extract_tfidf_top_words(texts: list[str], y: np.ndarray, max_features: int, min_df: int, out_dir: Path) -> pd.DataFrame:
    vectorizer = make_tfidf(max_features=max_features, min_df=min_df)
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x, y)
    terms = np.array(vectorizer.get_feature_names_out())
    weights = model.coef_[0]

    def meaningful(term: str) -> bool:
        parts = term.split()
        return any(part not in STOPWORDS and len(part) >= 3 for part in parts)

    candidates = pd.DataFrame({"term": terms, "weight": weights})
    candidates = candidates[candidates["term"].map(meaningful)].copy()
    pos = candidates.sort_values("weight", ascending=False).head(60)
    pos["label_name"] = "Positif"
    neg = candidates.sort_values("weight", ascending=True).head(60)
    neg["label_name"] = "Negatif"
    top_words = pd.concat([pos, neg], ignore_index=True)
    top_words.to_csv(out_dir / "top_words_tfidf.csv", index=False)
    return top_words


def write_keyword_counts(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    token_sets = df["text"].map(lambda text: set(tokenize(text, remove_stopwords=False, min_len=2)))
    rows: list[dict] = []
    pos_total = int(df["label"].eq(1).sum())
    neg_total = int(df["label"].eq(0).sum())
    for term in KEYWORD_SEEDS:
        pos_count = int(((df["label"].eq(1)) & token_sets.map(lambda tokens: term in tokens)).sum())
        neg_count = int(((df["label"].eq(0)) & token_sets.map(lambda tokens: term in tokens)).sum())
        if pos_count == 0 and neg_count == 0:
            continue
        pos_rate = pos_count / pos_total if pos_total else 0.0
        neg_rate = neg_count / neg_total if neg_total else 0.0
        dominant = "Positif" if pos_rate >= neg_rate else "Negatif"
        rows.append(
            {
                "term": term,
                "positif_docs": pos_count,
                "negatif_docs": neg_count,
                "positif_rate": pos_rate,
                "negatif_rate": neg_rate,
                "dominant_label": dominant,
                "lift": (pos_rate + 1e-9) / (neg_rate + 1e-9),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["dominant_label", "lift", "positif_docs", "negatif_docs"],
        ascending=[False, False, False, False],
    )
    result.to_csv(out_dir / "keyword_counts.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF and Word2Vec classical baselines with 10-fold CV.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Prepared binary CSV.")
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR / "classical"), help="Output artifact directory.")
    parser.add_argument("--fig-dir", default=str(ARTIFACT_DIR / "figures"), help="Shared figure directory.")
    parser.add_argument("--folds", type=int, default=10, help="Number of stratified CV folds.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--w2v-size", type=int, default=150)
    parser.add_argument(
        "--features",
        nargs="+",
        default=["tfidf", "word2vec"],
        choices=["tfidf", "word2vec"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    model_dir = MODEL_DIR / "classical"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_binary_dataset(args.data)
    texts = df["text"].to_numpy()
    text_list = df["text"].tolist()
    y = df["label"].to_numpy(dtype=int)

    estimators = classifiers(args.random_state)
    results: list[dict] = []
    fold_frames: list[pd.DataFrame] = []

    for feature in args.features:
        for model_name, estimator in estimators.items():
            if feature == "tfidf":
                model_estimator = estimator if model_name != "gradient_boosting" else estimator
                if model_name == "multinomial_nb":
                    model_estimator = MultinomialNB()
            else:
                model_estimator = estimator

            print(f"Training {feature} + {model_name} with {args.folds}-fold CV")
            aggregate, folds = evaluate_feature_model(
                feature_name=feature,
                model_name=model_name,
                estimator=model_estimator,
                texts=texts,
                y=y,
                folds=args.folds,
                random_state=args.random_state,
                max_features=args.max_features,
                min_df=args.min_df,
                w2v_size=args.w2v_size,
                out_dir=out_dir,
            )
            results.append(aggregate)
            fold_frames.append(folds)

    results_df = pd.DataFrame(results).sort_values(["f1", "roc_auc", "accuracy"], ascending=False)
    results_df.to_csv(out_dir / "results.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(out_dir / "fold_metrics.csv", index=False)

    best = results_df.iloc[0].to_dict()
    best_feature = best["feature"]
    best_model_name = best["model"]
    best_estimator = estimators[best_model_name]
    final_model = fit_final_model(
        feature_name=best_feature,
        model_name=best_model_name,
        estimator=best_estimator,
        texts=text_list,
        y=y,
        random_state=args.random_state,
        max_features=args.max_features,
        min_df=args.min_df,
        w2v_size=args.w2v_size,
    )
    joblib.dump(final_model, model_dir / "best_model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_oof = pd.read_csv(out_dir / f"oof_{best_feature}_{best_model_name}.csv")
    score = None if best_oof["score"].isna().any() else best_oof["score"].to_numpy()
    plot_confusion(
        best_oof["label"].to_numpy(),
        best_oof["prediction"].to_numpy(),
        fig_dir / "classical_best_confusion_matrix.png",
        title=f"Classical best: {best_feature} + {best_model_name}",
    )
    plot_roc_curve(
        best_oof["label"].to_numpy(),
        score,
        fig_dir / "classical_best_roc_auc.png",
        title=f"Classical ROC AUC: {best_feature} + {best_model_name}",
    )

    top_words = extract_tfidf_top_words(text_list, y, args.max_features, args.min_df, out_dir)
    write_keyword_counts(df, out_dir)
    plot_top_words(top_words, fig_dir / "top_words_tfidf.png", title="Top words TF-IDF")
    write_wordcloud(
        df.loc[df["label"].eq(1), "text"].tolist(),
        fig_dir / "wordcloud_positif.png",
        title="Word cloud Positif",
    )
    write_wordcloud(
        df.loc[df["label"].eq(0), "text"].tolist(),
        fig_dir / "wordcloud_negatif.png",
        title="Word cloud Negatif",
    )
    print("Best classical model:")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

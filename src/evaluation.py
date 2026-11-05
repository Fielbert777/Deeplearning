"""Metrik evaluasi dan loop cross-validation yang transparan."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from . import config


def recall_at_top_k(y_true, score, k: float = config.TOP_K_FRACTION) -> float:
    """Proporsi seluruh nasabah 'yes' yang tertangkap jika hanya k% nasabah dengan skor
    tertinggi yang ditelepon. Metrik bisnis: 'dengan menelepon 20% daftar, berapa persen
    calon nasabah yang kita dapat?' (acak = k)."""
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(k * len(y_true)))
    # mergesort = stabil, sehingga nilai skor yang sama diurutkan konsisten
    order = np.argsort(-np.asarray(score), kind="mergesort")[:n_top]
    return y_true[order].sum() / y_true.sum()


def compute_metrics(y_true, score, y_pred) -> dict:
    return {
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall_top20": recall_at_top_k(y_true, score),
    }


def run_cv(estimator, X: pd.DataFrame, y: np.ndarray, cv, name: str = "") -> tuple[pd.DataFrame, np.ndarray]:
    """Cross-validation manual: clone -> fit di fold latih -> evaluasi di fold validasi.

    Seluruh preprocessing ada di dalam `estimator` (Pipeline), jadi statistik imputasi,
    scaling, dan encoding hanya dihitung dari fold latih (tidak bocor ke fold validasi).

    Mengembalikan metrik per fold dan prediksi out-of-fold (OOF) untuk kurva/analisis.
    Untuk splitter temporal, baris yang tidak pernah jadi validasi bernilai NaN di OOF.
    """
    rows, oof = [], np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y[tr])
        fit_s = time.perf_counter() - t0
        score = est.predict_proba(X.iloc[va])[:, 1]
        pred = est.predict(X.iloc[va])
        oof[va] = score
        rows.append({"model": name, "fold": fold, "n_train": len(tr), "n_val": len(va),
                     "pos_rate_val": y[va].mean(), "fit_s": fit_s,
                     **compute_metrics(y[va], score, pred)})
    return pd.DataFrame(rows), oof


METRIC_LABELS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1@0.5", "recall_top20": "Recall@Top20%"}


def summarize(folds: pd.DataFrame, group_cols=("model",)) -> pd.DataFrame:
    """Rata-rata dan simpangan baku metrik antar fold."""
    metrics = list(METRIC_LABELS)
    agg = folds.groupby(list(group_cols), sort=False)[metrics + ["fit_s"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def format_table(summary: pd.DataFrame, extra_cols=()) -> pd.DataFrame:
    """Tabel siap tampil: 'mean ± std' untuk tiap metrik."""
    out = summary[list(extra_cols) + ["model"]].copy()
    for m, label in METRIC_LABELS.items():
        out[label] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    out["Waktu fit (s)"] = summary["fit_s_mean"].round(1)
    return out

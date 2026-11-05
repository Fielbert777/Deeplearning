"""Definisi baseline dan model kandidat."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .features import build_preprocessor


class CampaignHistoryRule(BaseEstimator, ClassifierMixin):
    """Baseline aturan bisnis (tidak belajar dari data).

    Logika yang biasa dipakai tim telemarketing tanpa model:
      1. telepon dulu nasabah yang SUKSES pada kampanye sebelumnya,
      2. lalu nasabah yang pernah dihubungi tetapi gagal,
      3. terakhir sisanya; nasabah dengan nomor seluler didahulukan di tiap kelompok.
    Skor hanya dipakai untuk mengurutkan. Prediksi kelas = 1 jika poutcome == 'success'.
    """

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def _score(self, X):
        history = X["poutcome"].map({"success": 2.0, "failure": 1.0}).fillna(0.0).to_numpy()
        cellular = (X["contact"] == "cellular").to_numpy().astype(float)
        return (history + 0.5 * cellular) / 2.5  # skala ke [0, 1]

    def predict_proba(self, X):
        s = self._score(X)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (X["poutcome"] == "success").to_numpy().astype(int)


def make_pipeline(model, feature_set="fe", include_leaky=False) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(feature_set, include_leaky=include_leaky)),
        ("model", model),
    ])


def logistic_regression():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)


def decision_tree():
    return DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced",
                                  random_state=config.SEED)


def random_forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=config.SEED)


def hist_gradient_boosting():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          class_weight="balanced", early_stopping=False,
                                          random_state=config.SEED)


def mlp():
    return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=256,
                         learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                         n_iter_no_change=10, random_state=config.SEED)


def baselines() -> dict:
    """Baseline: tanpa informasi (Dummy), aturan bisnis, dan model ML paling sederhana."""
    return {
        "B0 Dummy (prior)": DummyClassifier(strategy="prior"),
        "B1 Aturan riwayat kampanye": CampaignHistoryRule(),
        "B2 LogReg fitur dasar": make_pipeline(logistic_regression(), "base"),
    }


def candidates(feature_set="fe") -> dict:
    """Model kandidat awal (semua tanpa `duration`)."""
    return {
        "Logistic Regression": make_pipeline(logistic_regression(), feature_set),
        "Decision Tree": make_pipeline(decision_tree(), feature_set),
        "Random Forest": make_pipeline(random_forest(), feature_set),
        "HistGradientBoosting": make_pipeline(hist_gradient_boosting(), feature_set),
        "MLP (64-32)": make_pipeline(mlp(), feature_set),
    }

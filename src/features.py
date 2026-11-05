"""Preprocessing dan feature engineering.

Prinsip pencegahan leakage:
- Transformasi STATELESS (tidak belajar dari data, mis. log1p, binning umur, 999 -> NaN)
  boleh dilakukan di mana saja karena tidak memakai statistik data.
- Transformasi STATEFUL (imputasi median, standardisasi, one-hot + pengelompokan kategori
  langka) dibungkus di dalam `Pipeline` sehingga di-`fit` HANYA pada fold latih
  di setiap iterasi cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config

AGE_BINS = [0, 24, 34, 44, 54, 64, 120]
AGE_LABELS = ["<=24", "25-34", "35-44", "45-54", "55-64", "65+"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering awal (stateless, aman dari leakage).

    1. pdays == 999 berarti "belum pernah dihubungi pada kampanye sebelumnya", bukan 999 hari.
       Diubah menjadi NaN -> nanti diimputasi + diberi indikator `missingindicator_pdays`
       (= nasabah belum pernah dihubungi) di dalam pipeline.
    2. campaign sangat menceng ke kanan (maks 56) -> log1p mengurangi pengaruh nilai ekstrem
       untuk model linear.
    3. age_group: hubungan umur dengan target berbentuk U (muda & lansia lebih tinggi),
       yang tidak bisa ditangkap satu koefisien linear -> dibuat kelompok umur.
    """
    X = X.copy()
    X["pdays"] = X["pdays"].where(X["pdays"] != 999, np.nan)
    X["campaign_log"] = np.log1p(X["campaign"])
    X["age_group"] = pd.cut(X["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    return X.drop(columns=["campaign"])


def _one_hot():
    # min_frequency: kategori dengan < 20 baris di fold latih digabung jadi "infrequent"
    # (mis. default='yes' hanya 3 baris). handle_unknown='infrequent_if_exist' membuat
    # kategori yang tidak terlihat saat fit tidak menimbulkan error.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)


def build_preprocessor(feature_set: str = "fe", include_leaky: bool = False) -> Pipeline:
    """Buat pipeline preprocessing.

    feature_set:
        "base" -> fitur asli apa adanya (pdays=999 tetap angka, campaign mentah).
        "fe"   -> dengan feature engineering awal (lihat `add_engineered_features`).
    include_leaky: tambahkan `duration` (HANYA untuk demonstrasi efek leakage).
    """
    cat_cols = list(config.CATEGORICAL_COLS)
    num_cols = list(config.NUMERIC_COLS)
    steps = []
    if feature_set == "fe":
        steps.append(("feat_eng", FunctionTransformer(add_engineered_features)))
        num_cols = [c if c != "campaign" else "campaign_log" for c in num_cols]
        cat_cols = cat_cols + ["age_group"]
    elif feature_set != "base":
        raise ValueError(f"feature_set tidak dikenal: {feature_set}")
    if include_leaky:
        num_cols = num_cols + config.LEAKY_COLS

    numeric = Pipeline([
        # add_indicator=True membuat kolom biner untuk baris yang awalnya NaN (pdays=999).
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    columns = ColumnTransformer(
        [("num", numeric, num_cols), ("cat", _one_hot(), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps.append(("columns", columns))
    return Pipeline(steps)


def feature_names(preprocessor: Pipeline) -> list[str]:
    """Nama kolom keluaran preprocessor yang sudah di-fit."""
    return list(preprocessor.named_steps["columns"].get_feature_names_out())

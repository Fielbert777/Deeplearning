"""Memuat, memeriksa kualitas, membersihkan, dan membagi dataset Bank Marketing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=config.DATA_RAW) -> pd.DataFrame:
    """Baca CSV asli UCI (separator ';'), rapikan nama kolom, ubah target ke 0/1.

    Urutan baris dipertahankan karena dataset asli terurut kronologis
    (Mei 2008 - November 2010); urutan ini disimpan di kolom `time_idx`.
    """
    df = pd.read_csv(path, sep=";")
    df.columns = [c.replace(".", "_") for c in df.columns]
    df[config.TARGET] = (df[config.TARGET] == "yes").astype(int)
    return add_time_columns(df)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rekonstruksi tahun kontak dari urutan baris.

    Dataset tidak punya kolom tahun, tetapi baris terurut berdasarkan tanggal.
    Setiap kali nomor bulan turun (mis. dec -> mar), berarti tahun berganti.
    Kolom ini hanya dipakai untuk analisis dan validasi temporal, BUKAN sebagai fitur.
    """
    df = df.copy()
    month_num = df["month"].map({m: i + 1 for i, m in enumerate(config.MONTH_ORDER)}).to_numpy()
    year_increment = np.r_[0, (np.diff(month_num) < 0).astype(int)]
    df["time_idx"] = np.arange(len(df))
    df["year"] = 2008 + np.cumsum(year_increment)
    df["period"] = df["year"].astype(str) + "-" + pd.Series(month_num, index=df.index).map("{:02d}".format)
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan kualitas per kolom: tipe, jumlah nilai unik, nilai hilang, dan nilai 'unknown'."""
    cols = [c for c in df.columns if c not in config.TIME_COLS]
    rows = []
    for c in cols:
        s = df[c]
        is_text = not pd.api.types.is_numeric_dtype(s)
        n_unknown = int((s == "unknown").sum()) if is_text else 0
        rows.append({
            "kolom": c,
            "tipe": "kategorikal" if is_text else "numerik",
            "n_unik": int(s.nunique()),
            "n_null": int(s.isna().sum()),
            "n_unknown": n_unknown,
            "pct_unknown": round(100 * n_unknown / len(df), 2),
            "contoh_nilai": ", ".join(map(str, s.dropna().unique()[:4])),
        })
    return pd.DataFrame(rows)


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Hapus baris duplikat persis (berdasarkan kolom asli, bukan kolom waktu buatan).

    Duplikat dibuang SEBELUM split supaya baris yang sama tidak muncul sekaligus
    di data latih dan data uji (salah satu bentuk kebocoran data).
    """
    original_cols = [c for c in df.columns if c not in config.TIME_COLS]
    mask = df.duplicated(subset=original_cols, keep="first")
    return df.loc[~mask].reset_index(drop=True), int(mask.sum())


def split_dev_test(df: pd.DataFrame, test_size=config.TEST_SIZE, seed=config.SEED):
    """Bagi menjadi data development (untuk EDA, CV, seleksi model) dan test set terkunci.

    Split stratified menjaga proporsi kelas positif (~11%) sama di kedua bagian.
    Test set TIDAK disentuh selama Milestone 1; baru dipakai sekali untuk evaluasi final (UAS).
    """
    dev, test = train_test_split(
        df, test_size=test_size, stratify=df[config.TARGET], random_state=seed
    )
    # Urutkan kembali secara kronologis agar validasi temporal bisa dilakukan.
    dev = dev.sort_values("time_idx").reset_index(drop=True)
    test = test.sort_values("time_idx").reset_index(drop=True)
    return dev, test


def save_split_ids(dev: pd.DataFrame, test: pd.DataFrame, out_dir=config.SPLIT_DIR) -> None:
    """Simpan `time_idx` baris dev/test agar split bisa diaudit dan direproduksi persis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dev[["time_idx"]].to_csv(out_dir / "dev_time_idx.csv", index=False)
    test[["time_idx"]].to_csv(out_dir / "test_time_idx.csv", index=False)


def get_X_y(df: pd.DataFrame, include_leaky: bool = False):
    """Pisahkan fitur dan target. `duration` dibuang kecuali untuk eksperimen demonstrasi leakage."""
    drop = [config.TARGET] + config.TIME_COLS
    if not include_leaky:
        drop += config.LEAKY_COLS
    X = df.drop(columns=drop)
    y = df[config.TARGET].to_numpy()
    return X, y

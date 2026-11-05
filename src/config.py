"""Konfigurasi global proyek: path, seed, dan daftar kolom."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "bank-additional-full.csv"
SPLIT_DIR = ROOT / "data" / "splits"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
TOP_K_FRACTION = 0.20  # proporsi nasabah yang "ditelepon" untuk metrik Recall@Top20%

TARGET = "y"

# Kolom yang TIDAK BOLEH dipakai sebagai fitur model prediktif.
# - duration : lama panggilan baru diketahui SETELAH panggilan selesai (target leakage).
# - time_idx, year, period : penanda waktu hasil rekonstruksi, hanya untuk analisis/validasi.
LEAKY_COLS = ["duration"]
TIME_COLS = ["time_idx", "year", "period"]

NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
MACRO_COLS = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri"]

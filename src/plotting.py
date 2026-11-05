"""Gaya grafik yang konsisten untuk notebook dan laporan.

Palet kategorikal diambil dalam urutan tetap (sudah divalidasi untuk buta warna);
warna mengikuti entitas, jadi satu model selalu memakai warna yang sama di semua grafik.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
# Warna kelas target: tidak berlangganan vs berlangganan.
CLASS_COLORS = {0: BLUE, 1: ORANGE}
CLASS_LABELS = {0: "Tidak (no)", 1: "Ya (yes)"}
# Warna per model (tetap di semua grafik).
MODEL_COLORS = {
    "Logistic Regression": BLUE,
    "HistGradientBoosting": ORANGE,
    "B1 Aturan riwayat kampanye": AQUA,
    "Random Forest": YELLOW,
    "MLP (64-32)": VIOLET,
    "Decision Tree": MAGENTA,
}
# Diverging biru <-> merah dengan titik tengah abu-abu netral (untuk korelasi).
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ee9493", "#b8302f"]
)


def setup():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "text.color": INK,
    })


def savefig(fig, name: str):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / f"{name}.png")


def hbar(ax, labels, values, color=BLUE, fmt="{:.1%}", height=0.6):
    """Bar horizontal tipis dengan label nilai di ujung bar."""
    ys = range(len(labels))
    ax.barh(ys, values, color=color, height=height, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for y, v in zip(ys, values):
        ax.text(v + vmax * 0.01, y, fmt.format(v), va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, vmax * 1.18)

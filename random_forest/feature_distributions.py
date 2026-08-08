import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

data = np.load(DATA_PATH)
x, y = data["x"], data["y"]

fraud_mask = y == 1
normal_mask = y == 0

x_fraud = x[fraud_mask]
x_normal = x[normal_mask]

print(f"Fraud: {x_fraud.shape[0]:,} redova, Normal: {x_normal.shape[0]:,} redova")

# feature importance redosled iz Random Forest-a (najvaznije prvo), da grafovi budu u tom redu
FEATURE_IMPORTANCE_ORDER = [2, 6, 3, 8, 12, 11, 13, 7, 4, 1, 9, 15, 5, 14, 16, 0, 10]

fig, axes = plt.subplots(5, 4, figsize=(20, 22))
axes = axes.flatten()

for plot_idx, col in enumerate(FEATURE_IMPORTANCE_ORDER):
    ax = axes[plot_idx]

    fraud_col = x_fraud[:, col]
    normal_col = x_normal[:, col]

    pct_missing_fraud = (fraud_col == -1).mean() * 100
    pct_missing_normal = (normal_col == -1).mean() * 100

    # zajednicki binovi preko opsega OBE klase (ukljucujuci -1)
    combined_min = min(fraud_col.min(), normal_col.min())
    combined_max = max(fraud_col.max(), normal_col.max())
    bins = np.linspace(combined_min, combined_max, 40)

    ax.hist(normal_col, bins=bins, density=True, alpha=0.5, label="normal", color="tab:blue")
    ax.hist(fraud_col, bins=bins, density=True, alpha=0.5, label="fraud", color="tab:red")

    ax.set_title(
        f"kol_{col}  (RF importance rank #{plot_idx+1})\n"
        f"missing: normal={pct_missing_normal:.1f}%  fraud={pct_missing_fraud:.1f}%",
        fontsize=9,
    )
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

# ugasi neiskoriscene subplot-ove (17 kolona u 5x4=20 slotova)
for i in range(len(FEATURE_IMPORTANCE_ORDER), len(axes)):
    axes[i].axis("off")

fig.suptitle(
    "Raspodela vrednosti po koloni: fraud vs normal (ceo graf, density-normalizovano, log y-osa)\n"
    "Sortirano po Random Forest feature importance rangu (najvaznije prvo)",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_feature_distributions_fraud_vs_normal.png", dpi=100, bbox_inches="tight")
print(f"\nSlika sacuvana: {OUT_DIR / 'plot_feature_distributions_fraud_vs_normal.png'}")

# ===========================================================
# Dodatna tabela: prosek/medijana po klasi za svaku kolonu (samo ne-missing vrednosti)
# ===========================================================
print("\n=== Prosek/medijana po koloni (samo ne-missing, tj. != -1) ===")
print(f"{'kol':6s} {'fraud_mean':12s} {'normal_mean':12s} {'fraud_med':12s} {'normal_med':12s}")
for col in range(17):
    fc = x_fraud[:, col]
    nc = x_normal[:, col]
    fc_valid = fc[fc != -1]
    nc_valid = nc[nc != -1]
    fm = fc_valid.mean() if len(fc_valid) else float("nan")
    nm = nc_valid.mean() if len(nc_valid) else float("nan")
    fmed = np.median(fc_valid) if len(fc_valid) else float("nan")
    nmed = np.median(nc_valid) if len(nc_valid) else float("nan")
    print(f"kol_{col:<3d}{fm:<12.4f}{nm:<12.4f}{fmed:<12.4f}{nmed:<12.4f}")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # da vidimo common_data.py iz roditeljskog foldera

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from common_data import load_split, SEED

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [f"kol_{i}" for i in range(17)]


# rucna implementacija PR/ROC/AUC (bez sklearn.metrics) -- da se vidi kako kriva zapravo nastaje iz praga
def precision_recall_curve_manual(y_true, y_score):
    order = np.argsort(-y_score)  # sortiranje po opadajucem skoru = redom kako bi prag rastao od 0 ka 1
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)  # kumulativna suma na sortiranom nizu daje TP/FP na svakom mogucem pragu odjednom
    fps = np.cumsum(1 - y_sorted)
    n_pos = y_true.sum()
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return precision, recall, y_score[order]


def roc_curve_manual(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return fps / n_neg, tps / n_pos


def auc_trapz(x_vals, y_vals):
    order = np.argsort(x_vals)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # trapz je preimenovan u trapezoid u novijim numpy verzijama
    return trapz_fn(y_vals[order], x_vals[order])


if __name__ == "__main__":
    # Random Forest NE treba standardizaciju (invarijantan na skalu feature-a)
    x_train, y_train, x_val, y_val = load_split()  # isti 60/40 split kao NN eksperimenti
    print(f"Train: {len(x_train)} ({(y_train == 1).sum()} fraud), Val: {len(x_val)} ({(y_val == 1).sum()} fraud)")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,  # sprecava da pojedinacna stabla fituju sum/pojedinacne primere (regularizacija)
        n_jobs=-1,
        random_state=SEED,
    )
    clf.fit(x_train, y_train)

    probs = clf.predict_proba(x_val)[:, 1]  # verovatnoca klase 1 (fraud)
    preds = (probs > 0.5).astype(int)
    y_val_int = y_val.astype(int)

    accuracy = (preds == y_val_int).mean()
    tp = int(((preds == 1) & (y_val_int == 1)).sum())
    fp = int(((preds == 1) & (y_val_int == 0)).sum())
    fn = int(((preds == 0) & (y_val_int == 1)).sum())
    tn = int(((preds == 0) & (y_val_int == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n=== Random Forest rezultati (val skup, prag=0.5) ===")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ===========================================================
    # Plot 1: Feature importance
    # ===========================================================
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    plt.bar(range(17), importances[order], color="tab:blue")
    plt.xticks(range(17), [FEATURE_NAMES[i] for i in order], rotation=45)
    plt.ylabel("Feature importance (Gini)")
    plt.title("Random Forest -- vaznost svake od 17 kolona za razdvajanje fraud/normal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_feature_importance.png")
    print(f"\nSlika: {OUT_DIR / 'plot_feature_importance.png'}")
    print("Top 5 najvaznijih kolona:", [FEATURE_NAMES[i] for i in order[:5]])

    # ===========================================================
    # Plot 2: Konfuziona matrica (rucno crtanje, bez sklearn display alata)
    # ===========================================================
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["predvidjeno normal", "predvidjeno fraud"])
    plt.yticks([0, 1], ["stvarno normal", "stvarno fraud"])
    plt.title(f"Konfuziona matrica (prag=0.5)\nAccuracy={accuracy*100:.2f}%  F1={f1:.4f}")
    plt.colorbar(label="broj primera")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_confusion_matrix.png")
    print(f"Slika: {OUT_DIR / 'plot_confusion_matrix.png'}")

    # ===========================================================
    # Plot 3: ROC kriva
    # ===========================================================
    fpr, tpr = roc_curve_manual(y_val_int, probs)
    roc_auc = auc_trapz(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="tab:orange", label=f"Random Forest (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="slucajan model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (recall)")
    plt.title("ROC kriva -- Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_roc_curve.png")
    print(f"Slika: {OUT_DIR / 'plot_roc_curve.png'}  (AUC={roc_auc:.4f})")

    # ===========================================================
    # Plot 4: Precision-Recall kriva
    # ===========================================================
    prec_curve, rec_curve, _ = precision_recall_curve_manual(y_val_int, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prec_curve, color="tab:purple")
    plt.scatter([recall], [precision], color="green", zorder=5, label=f"prag=0.5 (P={precision:.3f}, R={recall:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall kriva -- Random Forest (balansiran 60/40 val skup)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_precision_recall_curve.png")
    print(f"Slika: {OUT_DIR / 'plot_precision_recall_curve.png'}")

    # ===========================================================
    # Plot 5: RF vs NN poredjenje (rucno uneti rezultati iz ranijih NN eksperimenata,
    # isti 60/40 balansiran val skup)
    # ===========================================================
    comparison = {  # rezultati ranijih NN eksperimenata uneti rucno, da bi se videli na istom grafiku
        "Logisticka regresija\n(18 param)": {"val_loss": 0.585, "accuracy": 0.6685, "f1": None},
        "MLP 5 slojeva\n(34.2k param)": {"val_loss": 0.578, "accuracy": None, "f1": None},
        "Random Forest\n(200 stabala)": {"val_loss": None, "accuracy": accuracy, "f1": f1},
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(comparison.keys())
    accs = [comparison[n]["accuracy"] * 100 if comparison[n]["accuracy"] else 0 for n in names]
    bars = ax.bar(names, accs, color=["tab:blue", "tab:gray", "tab:green"])
    for bar, acc, n in zip(bars, accs, names):
        label = f"{acc:.1f}%" if comparison[n]["accuracy"] else "N/A\n(nije direktno merena)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, label, ha="center", fontsize=9)
    ax.set_ylabel("Accuracy (%) na val skupu")
    ax.set_title("Random Forest vs Neuronske mreze -- Accuracy poredjenje\n(isti 60/40 balansiran val skup)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_rf_vs_nn_comparison.png")
    print(f"Slika: {OUT_DIR / 'plot_rf_vs_nn_comparison.png'}")

    print("\nSvi plotovi sacuvani u:", OUT_DIR)

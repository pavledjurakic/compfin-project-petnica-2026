"""
Random Forest baseline for fraud classification.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
)

SEED = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
OUT_DIR = Path(__file__).resolve().parent / "plots"
FEATURE_NAMES = [f"col_{i}" for i in range(17)]


def load_balanced_split(val_fraction=0.2, fraud_ratio=0.4, seed=SEED, data_path=DATA_PATH):
    np.random.seed(seed)
    data = np.load(data_path)
    x, y = data["x"], data["y"]

    fraud_idx = np.where(y == 1)[0]
    normal_idx_all = np.where(y == 0)[0]
    n_normal = int(len(fraud_idx) * (1 - fraud_ratio) / fraud_ratio)
    normal_idx = np.random.choice(normal_idx_all, size=n_normal, replace=False)

    subset_idx = np.concatenate([fraud_idx, normal_idx])
    np.random.shuffle(subset_idx)
    x_subset, y_subset = x[subset_idx], y[subset_idx]

    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in np.unique(y_subset):
        cls_idx = np.where(y_subset == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(len(cls_idx) * val_fraction)
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return x_subset[train_idx], y_subset[train_idx], x_subset[val_idx], y_subset[val_idx]


def train_random_forest(x_train, y_train, seed=SEED):
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    clf.fit(x_train, y_train)
    return clf


def main():
    OUT_DIR.mkdir(exist_ok=True)

    x_train, y_train, x_val, y_val = load_balanced_split()
    print(f"train: {len(x_train)} ({int((y_train == 1).sum())} fraud), val: {len(x_val)}")

    clf = train_random_forest(x_train, y_train)

    probs = clf.predict_proba(x_val)[:, 1]
    preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

    print(f"\naccuracy={accuracy*100:.2f}%  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(9, 5))
    plt.bar(range(17), importances[order])
    plt.xticks(range(17), [FEATURE_NAMES[i] for i in order], rotation=45)
    plt.ylabel("feature importance (Gini)")
    plt.title("Random Forest -- which columns separate fraud from normal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png")
    print(f"top 5 columns: {[FEATURE_NAMES[i] for i in order[:5]]}")

    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["predicted normal", "predicted fraud"])
    plt.yticks([0, 1], ["actual normal", "actual fraud"])
    plt.title(f"Confusion matrix (threshold=0.5)\naccuracy={accuracy*100:.2f}%  F1={f1:.4f}")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = roc_auc_score(y_val, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png")

    prec_curve, rec_curve, _ = precision_recall_curve(y_val, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(rec_curve, prec_curve)
    plt.scatter([recall], [precision], color="green", zorder=5, label=f"threshold=0.5 (P={precision:.3f}, R={recall:.3f})")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "precision_recall_curve.png")

    print(f"\nplots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

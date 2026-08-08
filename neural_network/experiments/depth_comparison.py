from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SEED = 42
torch.manual_seed(SEED)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
OUT_DIR = Path(__file__).resolve().parent / "plots"


def load_split(val_fraction=0.2, fraud_ratio=0.4, seed=SEED, data_path=DATA_PATH):
    np.random.seed(seed)
    data = np.load(data_path)
    x, y = data["x"], data["y"]

    fraud_idx = np.where(y == 1)[0]
    normal_idx_all = np.where(y == 0)[0]
    n_normal = int(len(fraud_idx) * (1 - fraud_ratio) / fraud_ratio)
    normal_idx = np.random.choice(normal_idx_all, size=n_normal, replace=False)

    subset_idx = np.concatenate([fraud_idx, normal_idx])
    np.random.shuffle(subset_idx)
    x_subset, y_subset = x[subset_idx], y[subset_idx].astype(np.float32)

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


def standardize(x_train, x_val):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std


class BlockMLP(nn.Module):
    """Blok Linear -> BatchNorm1d -> ReLU -> Dropout, ponovljen N puta (N=len(hidden_dims)),
    zavrseno cistim Linear slojem (sirovi logit, bez norm/aktivacije)."""

    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_accuracy_f1(model, x, y, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        y_t = torch.tensor(y, dtype=torch.float32)
        accuracy = (preds == y_t).float().mean().item()
        tp = ((preds == 1) & (y_t == 1)).sum().item()
        fp = ((preds == 1) & (y_t == 0)).sum().item()
        fn = ((preds == 0) & (y_t == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, f1


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=64, learning_rate=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    train_losses, val_losses, val_accs, val_f1s = [], [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            n_seen += len(xb)
        train_losses.append(running_loss / n_seen)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * len(xb)
        val_losses.append(running_val_loss / len(x_val))

        acc, f1 = compute_accuracy_f1(model, x_val, y_val)
        val_accs.append(acc)
        val_f1s.append(f1)
        print(f"    Epoch {epoch+1:3d}/{epochs}  train_loss={train_losses[-1]:.4f}  "
              f"val_loss={val_losses[-1]:.4f}  val_acc={acc*100:.2f}%  val_f1={f1:.4f}")

    return train_losses, val_losses, val_accs, val_f1s


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = load_split()  # 60/40 default
    x_train, x_val = standardize(x_train, x_val)

    n_train = len(x_train)
    n_val = len(x_val)
    n_train_fraud = int((y_train == 1).sum())
    n_val_fraud = int((y_val == 1).sum())

    print(f"Train: {n_train} ({n_train_fraud} fraud, {n_train_fraud/n_train*100:.1f}%)")
    print(f"Val:   {n_val} ({n_val_fraud} fraud, {n_val_fraud/n_val*100:.1f}%)")

    configs = [
        ("1 blok",  (32,)),
        ("2 bloka", (48, 24)),
        ("3 bloka", (64, 32, 16)),
        ("5 blokova", (128, 128, 64, 64, 32)),
    ]

    results = {}
    for name, hidden_dims in configs:
        print(f"\n=== {name}: hidden_dims={hidden_dims} ===")
        model = BlockMLP(17, hidden_dims)
        n_params = sum(p.numel() for p in model.parameters())
        ratio = n_train / n_params
        print(f"    Parametri: {n_params}   (odnos primeri/parametri = {ratio:.2f}x)")

        train_losses, val_losses, val_accs, val_f1s = train_model(model, x_train, y_train, x_val, y_val)
        results[name] = dict(
            hidden_dims=hidden_dims, n_params=n_params, ratio=ratio,
            train_losses=train_losses, val_losses=val_losses, val_accs=val_accs, val_f1s=val_f1s,
        )
        print(f"    -> FINALNO: val_loss={val_losses[-1]:.4f}  val_acc={val_accs[-1]*100:.2f}%  val_f1={val_f1s[-1]:.4f}")

    # ===========================================================
    # JEDAN veliki plot: tekstualni objasnjavajuci panel + 4x3 grid (loss/acc/F1 po dubini)
    # ===========================================================
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(6, 3, height_ratios=[1.3, 1, 1, 1, 1, 0.55], hspace=0.55, wspace=0.3)

    # --- Panel 0: objasnjenje strukture + statistika dataset-a (tekst) ---
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")

    explanation = (
        "STRUKTURA MREZE: blok  [ Linear -> BatchNorm1d -> ReLU -> Dropout(0.2) ]  se ponavlja N puta (N = broj 'blokova' ispod),\n"
        "sirina slojeva opada sa dubinom (npr. 5 blokova = 128,128,64,64,32), a na kraju ide JEDAN cist Linear sloj (sirov logit, bez norm/aktivacije).\n\n"
        f"DATASET (balansiran 60/40 subset, data/dgraphfin.npz):\n"
        f"  Trening skup:    {n_train:,} primera   ({n_train_fraud:,} fraud / {n_train - n_train_fraud:,} normal  ->  {n_train_fraud/n_train*100:.1f}% / {(n_train-n_train_fraud)/n_train*100:.1f}%)\n"
        f"  Validacioni skup: {n_val:,} primera   ({n_val_fraud:,} fraud / {n_val - n_val_fraud:,} normal  ->  {n_val_fraud/n_val*100:.1f}% / {(n_val-n_val_fraud)/n_val*100:.1f}%)\n\n"
        "10x PRAVILO: preporuceno je da trening primeri budu bar 10x broj parametara modela (primeri/parametri >= 10) da bi se izbegao rizik od overfitting-a.\n"
        + "  ".join([
            f"[{name}: {r['n_params']:,} param, odnos={r['ratio']:.1f}x {'OK' if r['ratio'] >= 10 else 'RIZICNO (<10x)'}]"
            for name, r in results.items()
        ])
    )
    ax_text.text(0.01, 0.95, explanation, transform=ax_text.transAxes, fontsize=11,
                 verticalalignment="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

    # --- Paneli 1-4: po red za svaku dubinu (loss, accuracy, F1) ---
    for row, (name, r) in enumerate(results.items(), start=1):
        ax_loss = fig.add_subplot(gs[row, 0])
        ax_loss.plot(r["train_losses"], label="train loss", color="tab:blue")
        ax_loss.plot(r["val_losses"], label="val loss", color="tab:orange")
        ax_loss.set_title(f"{name} ({r['hidden_dims']}) -- BCE Loss", fontsize=10)
        ax_loss.set_xlabel("Epoha")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=8)

        ax_acc = fig.add_subplot(gs[row, 1])
        ax_acc.plot(r["val_accs"], color="tab:green")
        ax_acc.set_title(f"{name} -- Val Accuracy", fontsize=10)
        ax_acc.set_xlabel("Epoha")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0.4, 0.8)

        ax_f1 = fig.add_subplot(gs[row, 2])
        ax_f1.plot(r["val_f1s"], color="tab:red")
        ax_f1.set_title(f"{name} -- Val F1", fontsize=10)
        ax_f1.set_xlabel("Epoha")
        ax_f1.set_ylabel("F1")
        ax_f1.set_ylim(0, 0.8)

        # parametri/odnos ispisani u uglu loss plota
        ax_loss.text(
            0.02, 0.05,
            f"params={r['n_params']:,}\nodnos={r['ratio']:.1f}x",
            transform=ax_loss.transAxes, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # --- Panel 5: sumarna tabela na dnu (tekst) ---
    ax_table = fig.add_subplot(gs[5, :])
    ax_table.axis("off")
    table_text = f"{'Dubina':12s}{'Parametri':12s}{'Odnos p/p':12s}{'Final val_loss':16s}{'Final val_acc':16s}{'Final val_F1':12s}\n"
    table_text += "-" * 80 + "\n"
    for name, r in results.items():
        table_text += (f"{name:12s}{r['n_params']:<12,}{r['ratio']:<12.2f}"
                        f"{r['val_losses'][-1]:<16.4f}{r['val_accs'][-1]*100:<15.2f}%{r['val_f1s'][-1]:<12.4f}\n")
    ax_table.text(0.01, 0.9, table_text, transform=ax_table.transAxes, fontsize=10,
                  verticalalignment="top", family="monospace",
                  bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

    fig.suptitle("Uticaj dubine mreze (broj Linear-BatchNorm-ReLU-Dropout blokova) na performanse\n"
                 "lr=0.01, batch_size=64, epochs=20, balansiran 60/40 dataset", fontsize=14, y=0.995)

    plt.savefig(OUT_DIR / "findings_depth_comparison.png", dpi=110, bbox_inches="tight")
    print(f"\nSlika sacuvana: {OUT_DIR / 'findings_depth_comparison.png'}")

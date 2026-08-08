import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

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


class BudgetMLP(nn.Module):
    """Manja mreza, da stane u preporucenu velicinu parametara - 10x ratio
    """

    def __init__(self, input_dim, hidden_dims=(48, 24), dropout=0.2):
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


class OriginalMLP(nn.Module):
    """Originalna (preveliko-parametrizovana) mreza iz lab.py, za poredjenje."""

    def __init__(self, input_dim, hidden_dims=(128, 128, 64, 64, 32), dropout=0.2):
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
    """Accuracy i F1 (klasa 1 = fraud = pozitivna) na datom skupu, prag=0.5
    """
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

    return accuracy, f1, precision, recall


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=64, learning_rate=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, drop_last=True,  # drop_last -- izbegava batch=1 BatchNorm gresku
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False,
    )

    train_losses, val_losses = [], []
    val_accuracies, val_f1s = [], []

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

        acc, f1, prec, rec = compute_accuracy_f1(model, x_val, y_val)
        val_accuracies.append(acc)
        val_f1s.append(f1)

        print(f"  Epoch {epoch+1:3d}/{epochs}  train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}  "
              f"val_acc={acc*100:.2f}%  val_f1={f1:.4f}")

    return model, train_losses, val_losses, val_accuracies, val_f1s


if __name__ == "__main__":
    x_train, y_train, x_val, y_val = load_split()  # 60/40 default
    x_train, x_val = standardize(x_train, x_val)
    print(f"Train: {len(x_train)} ({(y_train == 1).sum()} fraud), Val: {len(x_val)} ({(y_val == 1).sum()} fraud)")

    learning_rates = [0.1, 0.05, 0.01]
    batch_size = 64

    model_configs = [
        ("BudgetMLP", "~2.2k parametara, unutar preporucenog budzeta", BudgetMLP),
        ("OriginalMLP", "34.2k parametara, preveliko za trening skup", OriginalMLP),
    ]

    results = {}
    for model_name, model_desc, ModelClass in model_configs:
        for lr in learning_rates:
            key = (model_name, lr)
            print(f"\n=== {model_name} ({model_desc}) | lr={lr} | batch_size={batch_size} ===")
            model = ModelClass(x_train.shape[1])
            n_params = sum(p.numel() for p in model.parameters())
            model, train_losses, val_losses, val_accs, val_f1s = train_model(
                model, x_train, y_train, x_val, y_val,
                epochs=20, batch_size=batch_size, learning_rate=lr,
            )
            results[key] = dict(
                train_losses=train_losses, val_losses=val_losses,
                val_accs=val_accs, val_f1s=val_f1s, n_params=n_params, desc=model_desc,
            )
            print(f"  -> FINALNO: val_loss={val_losses[-1]:.4f}  val_acc={val_accs[-1]*100:.2f}%  "
                  f"val_f1={val_f1s[-1]:.4f}  (params={n_params})")


    for (model_name, lr), r in results.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        axes[0].plot(r["train_losses"], label="train loss", color="tab:blue")
        axes[0].plot(r["val_losses"], label="val loss", color="tab:orange")
        axes[0].set_title("BCE Loss (trening vs validacija)")
        axes[0].set_xlabel("Epoha")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(r["val_accs"], color="tab:green")
        axes[1].set_title("Validaciona Accuracy")
        axes[1].set_xlabel("Epoha")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)

        axes[2].plot(r["val_f1s"], color="tab:red")
        axes[2].set_title("Validacioni F1 (fraud klasa)")
        axes[2].set_xlabel("Epoha")
        axes[2].set_ylabel("F1 score")
        axes[2].set_ylim(0, 1)

        fig.suptitle(
            f"{model_name} ({r['desc']}) | lr={lr} | batch_size={batch_size} | params={r['n_params']}\n"
            f"Finalno: val_loss={r['val_losses'][-1]:.4f}, accuracy={r['val_accs'][-1]*100:.2f}%, F1={r['val_f1s'][-1]:.4f}",
            fontsize=11,
        )
        plt.tight_layout()
        fname = f"plot_case_{model_name}_lr{str(lr).replace('.', 'p')}.png"
        plt.savefig(OUT_DIR / fname)
        print(f"Slika sacuvana: {fname}")

    # Sumarni plot: 2 reda (Budget/Original) x 3 kolone (lr) -- samo loss
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    for row, (model_name, model_desc, ModelClass) in enumerate(model_configs):
        for col, lr in enumerate(learning_rates):
            r = results[(model_name, lr)]
            ax = axes[row, col]
            ax.plot(r["train_losses"], label="train loss", color="tab:blue")
            ax.plot(r["val_losses"], label="val loss", color="tab:orange")
            ax.set_title(
                f"{model_name} ({model_desc})\nlr={lr}, batch={batch_size}, params={r['n_params']}\n"
                f"val_loss={r['val_losses'][-1]:.4f}, acc={r['val_accs'][-1]*100:.1f}%, F1={r['val_f1s'][-1]:.3f}",
                fontsize=8,
            )
            ax.set_xlabel("Epoha")
            if col == 0:
                ax.set_ylabel("BCE loss")
            ax.legend(fontsize=7)

    fig.suptitle("Sumarni pregled: Budzetirani (2.2k) vs Originalni (34.2k) model, razliciti lr, batch_size=64", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_budget_vs_original_lr_sweep_SUMARNO.png")
    print(f"\nSumarna slika sacuvana: plot_budget_vs_original_lr_sweep_SUMARNO.png")

    print("\n=== FINALNA TABELA SVIH REZULTATA ===")
    print(f"{'Model':15s} {'lr':8s} {'params':10s} {'val_loss':10s} {'val_acc':10s} {'val_f1':10s}")
    for model_name, model_desc, ModelClass in model_configs:
        for lr in learning_rates:
            r = results[(model_name, lr)]
            print(f"{model_name:15s} {lr:<8} {r['n_params']:<10} {r['val_losses'][-1]:<10.4f} "
                  f"{r['val_accs'][-1]*100:<9.2f}% {r['val_f1s'][-1]:<10.4f}")

"""Fraud classification baseline: logistic regression vs a small (1-3-5 layer) MLP.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

SEED = 42 # seed for rng

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
OUT_DIR = Path(__file__).resolve().parent


def load_balanced_split(val_fraction=0.2, fraud_ratio=0.4, seed=SEED, data_path=DATA_PATH):
    """All fraud nodes plus a random sample of normal nodes sized to hit fraud_ratio, then a train/val split."""
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

    rng = np.random.default_rng(seed) # random number generator
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


class LogisticRegressionNet(nn.Module):
    """Simplest baseline: a linear layer w/o hidden layers."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)  # logits, no sigmoid -- BCEWithLogitsLoss handles that


class MLPNet(nn.Module):
    """Configurable MLP -- Linear -> BatchNorm1d -> ReLU -> Dropout block, repeated
    per hidden_dims. Last layer is a plain Linear (raw logit, no norm/activation)."""

    def __init__(self, input_dim, hidden_dims=(48, 24, 12), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=256, learning_rate=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(x_train)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * len(xb)
        val_loss = running_val_loss / len(x_val)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return model, train_losses, val_losses


def main():
    x_train, y_train, x_val, y_val = load_balanced_split()
    x_train, x_val = standardize(x_train, x_val)
    print(f"Train: {len(x_train)} ({(y_train == 1).sum()} fraud), Val: {len(x_val)} ({(y_val == 1).sum()} fraud)")

    print("\n--- Logisticka regresija ---")
    logreg = LogisticRegressionNet(x_train.shape[1])
    logreg, logreg_train_losses, logreg_val_losses = train_model(logreg, x_train, y_train, x_val, y_val)

    print("\n--- MLP (3 sloja: 48,24,12) ---")
    mlp = MLPNet(x_train.shape[1])
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"Broj parametara: {n_params}")
    mlp, mlp_train_losses, mlp_val_losses = train_model(mlp, x_train, y_train, x_val, y_val)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    axes[0].plot(logreg_train_losses, label="train loss", color="tab:blue")
    axes[0].plot(logreg_val_losses, label="val loss", color="tab:orange")
    axes[0].set_title("Logisticka regresija")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()

    axes[1].plot(mlp_train_losses, label="train loss", color="tab:blue")
    axes[1].plot(mlp_val_losses, label="val loss", color="tab:orange")
    axes[1].set_title("MLP")
    axes[1].set_xlabel("Epoha")
    axes[1].legend()

    fig.suptitle("Trening i validacioni loss po epohi")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_training_loss.png")
    print(f"\nSlika sacuvana: {OUT_DIR / 'plot_training_loss.png'}")


if __name__ == "__main__":
    main()

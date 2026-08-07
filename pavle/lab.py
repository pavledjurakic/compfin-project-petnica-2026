import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

from common_data import load_split, standardize, SEED

torch.manual_seed(SEED)
OUT_DIR = Path(__file__).resolve().parent


# ===========================================================
# 1. Podaci -- balansiran subset (40% fraud / 60% normal) + train/val split
#    (implementacija u common_data.py, deljena sa svim model_*.py eksperimentima)
# ===========================================================
x_train, y_train, x_val, y_val = load_split()
x_train, x_val = standardize(x_train, x_val)

print(f"Train: {len(x_train)} ({(y_train == 1).sum()} fraud), Val: {len(x_val)} ({(y_val == 1).sum()} fraud)")


# ===========================================================
# 2. Modeli -- jedan vektor (x) -> jedan logit
# ===========================================================
class LogisticRegressionNet(nn.Module):
    """Najprostiji baseline: linearni sloj, bez skrivenih slojeva."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)  # logits (bez sigmoida -- to radi BCEWithLogitsLoss)


class MLPNet(nn.Module):
    """Konfigurabilna MLP -- blok Linear -> BatchNorm1d -> ReLU -> Dropout, ponovljen
    po hidden_dims. Poslednji sloj je cist Linear (sirovi logit, bez norm/aktivacije)."""

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
        layers.append(nn.Linear(prev_dim, 1))  # izlazni logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits


# ===========================================================
# 3. Trening petlja -- belezi train i val loss po epohi
# ===========================================================
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
        # --- trening korak ---
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

        # --- validacija (bez gradijenta) ---
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


if __name__ == "__main__":
    input_dim = x_train.shape[1]

    print("\n--- Logisticka regresija ---")
    logreg = LogisticRegressionNet(input_dim)
    logreg, logreg_train_losses, logreg_val_losses = train_model(logreg, x_train, y_train, x_val, y_val)

    print("\n--- MLP (5 slojeva: 128,128,64,64,32) ---")
    mlp = MLPNet(input_dim)
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"Broj parametara: {n_params}")
    mlp, mlp_train_losses, mlp_val_losses = train_model(mlp, x_train, y_train, x_val, y_val)

    # ===========================================================
    # 4. Plot: train vs val loss po epohi, 2 subplota (logreg | MLP)
    # ===========================================================
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

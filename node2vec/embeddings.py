"""node2vec embedding vs raw features for fraud classification, on subgraph_k30_N1.txt.

Only 80 fraud nodes are available in this subgraph (mentor-assigned dataset), so we
use small embedding dimensions and k-fold cross-validation instead of a single
train/val split, and try concatenating the embedding with raw features too.

Run with: python embeddings.py
"""

from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
SUBGRAPH_PATH = REPO_ROOT / "data" / "subgraph_k30_N1.txt"
OUT_DIR = Path(__file__).resolve().parent / "plots"

EMBED_DIMS = [8, 16, 32]
N_FOLDS = 5


def load_subgraph_edges(subgraph_path):
    edges = []
    with open(subgraph_path) as f:
        for line in f:
            src, dst, _ = line.split()
            edges.append((int(src), int(dst)))
    return edges


def build_graph(edges):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def compute_embeddings(graph, dimensions):
    node2vec = Node2Vec(graph, dimensions=dimensions, p=1, q=0.125, workers=5, seed=SEED, quiet=True)
    model = node2vec.fit()
    return {node: model.wv[str(node)] for node in graph.nodes}


class MLPNet(nn.Module):
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


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32, learning_rate=0.01, verbose=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loss, val_acc, val_f1 = None, None, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(x_val, dtype=torch.float32)).squeeze(1)
            val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.float32)).item()
            preds = (torch.sigmoid(val_logits).numpy() > 0.5).astype(int)
        val_acc = accuracy_score(y_val, preds)
        val_f1 = f1_score(y_val, preds, zero_division=0)

        if verbose:
            print(f"epoch {epoch+1:3d}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  val_f1={val_f1:.4f}")

    return val_loss, val_acc, val_f1


def standardize(x_train, x_val):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std


def balance_classes(node_ids, labels, fraud_ratio=0.4, seed=SEED):
    rng = np.random.default_rng(seed)
    fraud_ids = node_ids[labels == 1]
    normal_ids = node_ids[labels == 0]
    n_normal = min(int(len(fraud_ids) * (1 - fraud_ratio) / fraud_ratio), len(normal_ids))
    normal_ids = rng.choice(normal_ids, size=n_normal, replace=False)
    balanced_ids = np.concatenate([fraud_ids, normal_ids])
    balanced_labels = np.concatenate([np.ones(len(fraud_ids)), np.zeros(n_normal)]).astype(np.float32)
    return balanced_ids, balanced_labels


def cross_validate(x, y, n_folds=N_FOLDS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    losses, accs, f1s = [], [], []
    for train_idx, val_idx in skf.split(x, y):
        x_train, x_val = standardize(x[train_idx], x[val_idx])
        model = MLPNet(x_train.shape[1])
        loss, acc, f1 = train_model(model, x_train, y[train_idx], x_val, y[val_idx])
        losses.append(loss)
        accs.append(acc)
        f1s.append(f1)
    return np.mean(losses), np.mean(accs), np.mean(f1s)


def main(subgraph_path, data_path, out_dir):
    out_dir.mkdir(exist_ok=True)

    edges = load_subgraph_edges(subgraph_path)
    graph = build_graph(edges)
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    data = np.load(data_path)
    x_full, y_full = data["x"], data["y"]

    node_ids = np.array(list(graph.nodes))
    labels = y_full[node_ids]
    keep = (labels != 2) & (labels != 3)
    node_ids, labels = node_ids[keep], labels[keep].astype(np.float32)

    node_ids, labels = balance_classes(node_ids, labels)
    print(f"balanced: {len(node_ids)} ({int((labels == 1).sum())} fraud)")

    x_raw = x_full[node_ids]

    print(f"\n--- raw features (17-dim) baseline, {N_FOLDS}-fold CV ---")
    raw_loss, raw_acc, raw_f1 = cross_validate(x_raw, labels)
    print(f"raw: val_loss={raw_loss:.4f}  val_acc={raw_acc*100:.2f}%  val_f1={raw_f1:.4f}")

    results = {"raw": (raw_loss, raw_acc, raw_f1)}
    for dim in EMBED_DIMS:
        print(f"\n--- node2vec embedding, dim={dim}, {N_FOLDS}-fold CV ---")
        embeddings = compute_embeddings(graph, dim)
        x_emb = np.array([embeddings[n] for n in node_ids])

        emb_loss, emb_acc, emb_f1 = cross_validate(x_emb, labels)
        print(f"embedding only: val_loss={emb_loss:.4f}  val_acc={emb_acc*100:.2f}%  val_f1={emb_f1:.4f}")
        results[f"embedding dim={dim}"] = (emb_loss, emb_acc, emb_f1)

        x_concat = np.concatenate([x_emb, x_raw], axis=1)
        concat_loss, concat_acc, concat_f1 = cross_validate(x_concat, labels)
        print(f"embedding+raw: val_loss={concat_loss:.4f}  val_acc={concat_acc*100:.2f}%  val_f1={concat_f1:.4f}")
        results[f"embedding+raw dim={dim}"] = (concat_loss, concat_acc, concat_f1)

    names = list(results.keys())
    losses = [results[n][0] for n in names]
    f1s = [results[n][2] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(names, losses)
    axes[0].set_xlabel(f"mean val_loss ({N_FOLDS}-fold CV)")
    axes[0].set_title("Loss by input type")

    axes[1].barh(names, f1s)
    axes[1].set_xlabel(f"mean val_f1 ({N_FOLDS}-fold CV)")
    axes[1].set_title("F1 by input type")

    plt.tight_layout()
    plt.savefig(out_dir / "embedding_vs_raw.png")
    print(f"\nsaved: {out_dir / 'embedding_vs_raw.png'}")

    print(f"\n{'input':25s} {'val_loss':10s} {'val_f1':10s}")
    for name in names:
        print(f"{name:25s} {results[name][0]:<10.4f} {results[name][2]:<10.4f}")


if __name__ == "__main__":
    main(SUBGRAPH_PATH, DATA_PATH, OUT_DIR)

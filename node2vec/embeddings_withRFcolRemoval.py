"""
node2vec embedding vs raw features for fraud classification on subgraph_k30_N1.txt.
turned out bad for use, since it destorys the dataset essentially.
"""

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "neural_network"))
from lab import MLPNet, standardize

DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
SUBGRAPH_PATH = REPO_ROOT / "data" / "subgraph_k30_N1.txt"
OUT_DIR = Path(__file__).resolve().parent / "plots"

N_FOLDS = 5
BEST_DIM = 8
WALK_CONFIGS = [(10, 5), (10, 20), (40, 5), (40, 20)]  # (walk_length, num_walks)
TOP_RF_COLS = [2, 6, 3, 8, 12]  # highest Random Forest feature importance


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


def compute_embeddings(graph, dimensions, walk_length=80, num_walks=10):
    node2vec = Node2Vec(
        graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
        p=1, q=0.125, workers=5, seed=SEED, quiet=True,
    )
    model = node2vec.fit()
    return {node: model.wv[str(node)] for node in graph.nodes}


def engineered_raw_features(x_raw, top_cols=TOP_RF_COLS):
    """Top-5 RF-important columns + a missing-value count (fraud runs ~30% higher
    missing rate than normal across almost every column, so the count itself is signal)."""
    missing_count = (x_raw == -1).sum(axis=1, keepdims=True)
    return np.concatenate([x_raw[:, top_cols], missing_count], axis=1)


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32, learning_rate=0.01, verbose=False):
    """Returns per-epoch history (train_loss, val_loss, val_acc, val_f1 lists), same shape as lab.py."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
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
        with torch.no_grad():
            val_logits = model(torch.tensor(x_val, dtype=torch.float32)).squeeze(1)
            val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.float32)).item()
            preds = (torch.sigmoid(val_logits).numpy() > 0.5).astype(int)
        val_losses.append(val_loss)
        val_accs.append(accuracy_score(y_val, preds))
        val_f1s.append(f1_score(y_val, preds, zero_division=0))

        if verbose:
            print(f"epoch {epoch+1:3d}/{epochs}  val_loss={val_losses[-1]:.4f}  "
                  f"val_acc={val_accs[-1]*100:.2f}%  val_f1={val_f1s[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        final_val_probs = torch.sigmoid(model(torch.tensor(x_val, dtype=torch.float32)).squeeze(1)).numpy()

    return train_losses, val_losses, val_accs, val_f1s, final_val_probs


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
    """
    Also returns out-of-fold probabilities (each example scored only by a model that never
    trained on it) so a single, leakage-free ROC curve can be drawn over the whole dataset."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_losses, val_losses, val_accs, val_f1s = [], [], [], []
    oof_probs = np.zeros(len(y))
    for train_idx, val_idx in skf.split(x, y):
        x_train, x_val = standardize(x[train_idx], x[val_idx])
        model = MLPNet(x_train.shape[1])
        tl, vl, va, vf, probs = train_model(model, x_train, y[train_idx], x_val, y[val_idx])
        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)
        val_f1s.append(vf)
        oof_probs[val_idx] = probs
    return {
        "train_loss": np.mean(train_losses, axis=0),
        "val_loss": np.mean(val_losses, axis=0),
        "val_acc": np.mean(val_accs, axis=0),
        "val_f1": np.mean(val_f1s, axis=0),
        "oof_probs": oof_probs,
    }


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
    x_eng = engineered_raw_features(x_raw)

    print(f"\n--- raw features (17-dim) baseline, {N_FOLDS}-fold CV ---")
    raw_hist = cross_validate(x_raw, labels)
    print(f"raw: val_loss={raw_hist['val_loss'][-1]:.4f}  val_acc={raw_hist['val_acc'][-1]*100:.2f}%  val_f1={raw_hist['val_f1'][-1]:.4f}")

    print(f"\n--- engineered features (top-5 RF cols + missing count, 6-dim), {N_FOLDS}-fold CV ---")
    eng_hist = cross_validate(x_eng, labels)
    print(f"engineered: val_loss={eng_hist['val_loss'][-1]:.4f}  val_acc={eng_hist['val_acc'][-1]*100:.2f}%  val_f1={eng_hist['val_f1'][-1]:.4f}")

    results = {"raw (17-dim)": raw_hist, "engineered (6-dim)": eng_hist}
    for walk_length, num_walks in WALK_CONFIGS:
        tag = f"wl={walk_length},nw={num_walks}"
        print(f"\n--- node2vec dim={BEST_DIM}, {tag}, {N_FOLDS}-fold CV ---")
        embeddings = compute_embeddings(graph, BEST_DIM, walk_length, num_walks)
        x_emb = np.array([embeddings[n] for n in node_ids])

        emb_hist = cross_validate(x_emb, labels)
        print(f"embedding only: val_loss={emb_hist['val_loss'][-1]:.4f}  val_acc={emb_hist['val_acc'][-1]*100:.2f}%  val_f1={emb_hist['val_f1'][-1]:.4f}")
        results[f"embedding {tag}"] = emb_hist

        x_concat = np.concatenate([x_emb, x_eng], axis=1)
        concat_hist = cross_validate(x_concat, labels)
        print(f"embedding+engineered: val_loss={concat_hist['val_loss'][-1]:.4f}  val_acc={concat_hist['val_acc'][-1]*100:.2f}%  val_f1={concat_hist['val_f1'][-1]:.4f}")
        results[f"embedding+engineered {tag}"] = concat_hist

    names = list(results.keys())
    losses = [results[n]["val_loss"][-1] for n in names]
    f1s = [results[n]["val_f1"][-1] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(names, losses)
    axes[0].set_xlabel(f"final mean val_loss ({N_FOLDS}-fold CV)")
    axes[0].set_title("Loss by input type")

    axes[1].barh(names, f1s)
    axes[1].set_xlabel(f"final mean val_f1 ({N_FOLDS}-fold CV)")
    axes[1].set_title("F1 by input type")

    plt.tight_layout()
    plt.savefig(out_dir / "embedding_vs_raw.png")
    print(f"\nsaved: {out_dir / 'embedding_vs_raw.png'}")

    # per-epoch curves + ROC for the 3 most relevant configs: raw, engineered, best embedding
    embedding_names = [n for n in names if n.startswith("embedding ") or n.startswith("embedding+engineered ")]
    best_embedding_name = min(embedding_names, key=lambda n: results[n]["val_loss"][-1])
    curve_names = ["raw (17-dim)", "engineered (6-dim)", best_embedding_name]
    colors = {name: f"C{i}" for i, name in enumerate(curve_names)}  # same color for a config across all 3 panels

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name in curve_names:
        hist = results[name]
        color = colors[name]
        axes[0].plot(hist["train_loss"], linestyle="--", color=color, label=f"{name} (train)")
        axes[0].plot(hist["val_loss"], color=color, label=f"{name} (val)")
        axes[1].plot(hist["val_f1"], color=color, label=name)

        fpr, tpr, _ = roc_curve(labels, hist["oof_probs"])
        auc = roc_auc_score(labels, hist["oof_probs"])
        axes[2].plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title(f"Loss per epoch (mean over {N_FOLDS} folds)")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val F1")
    axes[1].set_title(f"F1 per epoch (mean over {N_FOLDS} folds)")
    axes[1].legend(fontsize=7)

    axes[2].plot([0, 1], [0, 1], color="gray", linestyle=":", label="random")
    axes[2].set_xlabel("false positive rate")
    axes[2].set_ylabel("true positive rate")
    axes[2].set_title("ROC (out-of-fold predictions)")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "loss_f1_curves.png")
    print(f"saved: {out_dir / 'loss_f1_curves.png'}")

    print(f"\n{'input':25s} {'val_loss':10s} {'val_f1':10s}")
    for name in names:
        print(f"{name:25s} {results[name]['val_loss'][-1]:<10.4f} {results[name]['val_f1'][-1]:<10.4f}")


if __name__ == "__main__":
    main(SUBGRAPH_PATH, DATA_PATH, OUT_DIR)

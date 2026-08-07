import numpy as np
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent  # CompFin - projekat/ (gde su dgraphfin.npz i subgraph_k30_N1.txt)


def load_small_subgraph_node_ids():
    """Cvorovi (indeksi) koji se pojavljuju u subgraph_k30_N1.txt (30-hop okolina
    1 fraud cvora). Koristi se da se eksperimenti ograniče na mali graf umesto punog."""
    node_ids = set()
    with open(DATA_DIR / "subgraph_k30_N1.txt") as f:
        for line in f:
            src, dst, ts = line.split()
            node_ids.add(int(src))
            node_ids.add(int(dst))
    return np.array(sorted(node_ids))


def load_split(val_frac=0.2, ratio_fraud_pct=40, seed=SEED, node_pool=None, return_indices=False):
    """Ucitava dgraphfin.npz, pravi balansiran subset (ratio_fraud_pct% fraud)
    i stratifikovan train/val split. Identicno onome iz lab.py da rezultati
    budu uporedivi izmedju modela.

    node_pool: ako je zadat (npr. iz load_small_subgraph_node_ids()), fraud/normal
    kandidati se biraju SAMO medju tim indeksima -- za eksperimente na malom podgrafu."""
    np.random.seed(seed)

    data = np.load(r"C:\Users\BG Computers\OneDrive\Radna površina\CompFin - projekat\dgraphfin.npz")
    x, y = data["x"], data["y"]

    if node_pool is not None:
        pool_mask = np.zeros(len(y), dtype=bool)
        pool_mask[node_pool] = True
        fraud_idx = np.where((y == 1) & pool_mask)[0]
        normal_idx_all = np.where((y == 0) & pool_mask)[0]
    else:
        fraud_idx = np.where(y == 1)[0]
        normal_idx_all = np.where(y == 0)[0]

    n_fraud = len(fraud_idx)
    n_normal_wanted = int(n_fraud * ((100 - ratio_fraud_pct) / ratio_fraud_pct))
    n_normal_wanted = min(n_normal_wanted, len(normal_idx_all))  # ne trazi vise nego sto ima u poolu
    normal_idx = np.random.choice(normal_idx_all, size=n_normal_wanted, replace=False)

    subset_idx = np.concatenate([fraud_idx, normal_idx])
    np.random.shuffle(subset_idx)

    x_subset = x[subset_idx]
    y_subset = y[subset_idx].astype(np.float32)

    rng = np.random.default_rng(seed)
    train_idx_parts, val_idx_parts = [], []
    for cls in np.unique(y_subset):
        cls_idx = np.where(y_subset == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(len(cls_idx) * val_frac)
        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if return_indices:
        # originalni (npz) indeksi cvorova koji su otisli u train/val -- korisno da se
        # izgradi "stvarna raspodela" eval skup bez preklapanja sa trening podacima
        return (
            x_subset[train_idx], y_subset[train_idx], x_subset[val_idx], y_subset[val_idx],
            subset_idx[train_idx], subset_idx[val_idx],
        )
    return x_subset[train_idx], y_subset[train_idx], x_subset[val_idx], y_subset[val_idx]


def standardize(x_train, x_val, return_stats=False):
    """Z-score standardizacija, statistike racunate SAMO na train skupu."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    if return_stats:
        return (x_train - mean) / std, (x_val - mean) / std, mean, std
    return (x_train - mean) / std, (x_val - mean) / std

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common_data import SEED
from model_budget_lr_sweep import BudgetMLP, train_model

np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).resolve().parent / "plots"
DATA_PATH = r"C:\Users\BG Computers\OneDrive\Radna površina\CompFin - projekat\dgraphfin.npz"

P, Q = 1.0, 0.125            # node2vec parametri (tvoj empirijski izbor)
NUM_WALKS = 5
WALK_LENGTH = 15
WINDOW = 3
EMBED_DIMS = [32, 64, 128]   # "igranje" sa dimenzionalnoscu
NEG_SAMPLES = 5
SKIPGRAM_EPOCHS = 3
SKIPGRAM_BATCH = 2048

# ===========================================================
# 1. Podgraf: svi fraud cvorovi + njihovi 1-hop susedi
# ===========================================================
print("=== 1. Gradim podgraf (fraud + 1-hop susedi) ===")
data = np.load(DATA_PATH)
y_full = data["y"]
edge_index = data["edge_index"]

fraud_idx_full = np.where(y_full == 1)[0]
fraud_set_full = set(fraud_idx_full.tolist())

und_src = np.concatenate([edge_index[:, 0], edge_index[:, 1]])
und_dst = np.concatenate([edge_index[:, 1], edge_index[:, 0]])

fraud_mask = np.isin(und_src, fraud_idx_full)
neighbors_of_fraud = set(und_dst[fraud_mask].tolist())
subgraph_nodes_orig = np.array(sorted(fraud_set_full | neighbors_of_fraud))
n_sub = len(subgraph_nodes_orig)  # ~43.6k umesto 3.7M -- na celom grafu bi cist Python node2vec bio prespor
print(f"Podgraf: {n_sub} cvorova")

# original id -> lokalni indeks (0..n_sub-1)
orig_to_local = {orig: i for i, orig in enumerate(subgraph_nodes_orig.tolist())}
subgraph_node_set = set(subgraph_nodes_orig.tolist())

# ivice unutar podgrafa, mapirane na lokalne indekse
edge_mask = np.isin(und_src, subgraph_nodes_orig) & np.isin(und_dst, subgraph_nodes_orig)
local_src = np.array([orig_to_local[n] for n in und_src[edge_mask]], dtype=np.int32)
local_dst = np.array([orig_to_local[n] for n in und_dst[edge_mask]], dtype=np.int32)
print(f"Ivica unutar podgrafa (usmerene, undirected duplirano): {len(local_src)}")

# adjacency liste (lista numpy nizova + lista setova za brzu 2nd-order proveru)
adj_list = [[] for _ in range(n_sub)]
for s, d in zip(local_src.tolist(), local_dst.tolist()):
    adj_list[s].append(d)
adj_list = [np.array(sorted(set(neigh)), dtype=np.int32) for neigh in adj_list]
adj_set = [set(neigh.tolist()) for neigh in adj_list]

n_isolated = sum(1 for a in adj_list if len(a) == 0)
print(f"Izolovanih cvorova (bez ivica unutar podgrafa): {n_isolated}")

# ===========================================================
# 2. Pristrasan 2nd-order random walk (node2vec), p=1, q=0.125
# ===========================================================
print(f"\n=== 2. Generisem random walk-ove (p={P}, q={Q}, num_walks={NUM_WALKS}, walk_length={WALK_LENGTH}) ===")


def node2vec_walk(start, walk_length, p, q):
    walk = [start]
    for _ in range(walk_length - 1):
        cur = walk[-1]
        cur_neighbors = adj_list[cur]
        if len(cur_neighbors) == 0:
            break
        if len(walk) == 1:
            nxt = cur_neighbors[np.random.randint(len(cur_neighbors))]
        else:
            prev = walk[-2]
            prev_neighbors = adj_set[prev]
            weights = np.empty(len(cur_neighbors), dtype=np.float64)
            for i, x in enumerate(cur_neighbors):
                if x == prev:
                    weights[i] = 1.0 / p  # vracanje na prethodni cvor (p=1 => neutralno)
                elif x in prev_neighbors:
                    weights[i] = 1.0  # sused i trenutnog i prethodnog cvora -- ostajanje "lokalno"
                else:
                    weights[i] = 1.0 / q  # dalji cvor -- malo q=0.125 povecava tezinu, walk vise "beži" napolje (DFS-like)
            probs = weights / weights.sum()
            nxt = cur_neighbors[np.random.choice(len(cur_neighbors), p=probs)]
        walk.append(int(nxt))
    return walk


t0 = time.time()
walks = []
node_order = np.arange(n_sub)
for walk_iter in range(NUM_WALKS):
    np.random.shuffle(node_order)
    for node in node_order:
        if len(adj_list[node]) == 0:
            continue
        walks.append(node2vec_walk(int(node), WALK_LENGTH, P, Q))
print(f"Generisano {len(walks)} walk-ova za {time.time()-t0:.1f}s")

# ===========================================================
# 3. Skip-gram trening parovi (center, context) sa prozorom WINDOW
# ===========================================================
print(f"\n=== 3. Gradim skip-gram parove (window={WINDOW}) ===")
centers, contexts = [], []
for walk in walks:
    L = len(walk)
    for i, center in enumerate(walk):
        lo, hi = max(0, i - WINDOW), min(L, i + WINDOW + 1)
        for j in range(lo, hi):
            if j != i:
                centers.append(center)
                contexts.append(walk[j])
centers = np.array(centers, dtype=np.int64)
contexts = np.array(contexts, dtype=np.int64)
print(f"Ukupno (center, context) parova: {len(centers):,}")

centers_t = torch.tensor(centers)
contexts_t = torch.tensor(contexts)


# ===========================================================
# 4. Skip-gram sa negative sampling -- cist PyTorch
# ===========================================================
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)  # embedding koji na kraju koristimo (cvor kao "centar")
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # pomocni embedding za kontekst, odbacuje se posle treninga
        nn.init.uniform_(self.in_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.out_embed.weight)  # standardni word2vec trik -- nule na startu stabilizuju rani trening

    def forward(self, center, context, negatives):
        v_c = self.in_embed(center)                         # (B, D)
        v_o = self.out_embed(context)                        # (B, D)
        pos_score = (v_c * v_o).sum(dim=1)                    # (B,)
        pos_loss = F.logsigmoid(pos_score)

        v_neg = self.out_embed(negatives)                     # (B, K, D)
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


def train_skipgram(embed_dim, vocab_size, centers_t, contexts_t, epochs=SKIPGRAM_EPOCHS, batch_size=SKIPGRAM_BATCH):
    model = SkipGramNS(vocab_size, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n = len(centers_t)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            c = centers_t[idx]
            ctx = contexts_t[idx]
            neg = torch.randint(0, vocab_size, (len(idx), NEG_SAMPLES))

            optimizer.zero_grad()
            loss = model(c, ctx, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"    [dim={embed_dim}] skip-gram epoch {epoch+1}/{epochs}  loss={total_loss/n_batches:.4f}")

    return model.in_embed.weight.detach().numpy()


# ===========================================================
# 5. Za svaku dimenziju: treniraj embedding, pa klasifikator (BudgetMLP, 2 sloja)
# ===========================================================
fraud_local = np.array([orig_to_local[n] for n in fraud_idx_full if n in orig_to_local])
normal_local_all = np.array([i for i, orig in enumerate(subgraph_nodes_orig) if y_full[orig] == 0])
print(f"\nU podgrafu: {len(fraud_local)} fraud, {len(normal_local_all)} normal cvorova dostupno za klasifikaciju")

n_fraud = len(fraud_local)
n_normal_wanted = min(int(n_fraud * 1.5), len(normal_local_all))  # 60/40 odnos
rng = np.random.default_rng(SEED)
normal_local = rng.choice(normal_local_all, size=n_normal_wanted, replace=False)

subset_local = np.concatenate([fraud_local, normal_local])
labels = np.array([1.0 if l in set(fraud_local.tolist()) else 0.0 for l in subset_local], dtype=np.float32)

# stratifikovan train/val split (80/20)
def stratified_split(idx, lab, val_frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in [0.0, 1.0]:
        cls_idx = idx[lab == cls]
        rng.shuffle(cls_idx)
        n_val = int(len(cls_idx) * val_frac)
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


train_local, val_local = stratified_split(subset_local, labels)
train_labels = np.array([1.0 if l in set(fraud_local.tolist()) else 0.0 for l in train_local], dtype=np.float32)
val_labels = np.array([1.0 if l in set(fraud_local.tolist()) else 0.0 for l in val_local], dtype=np.float32)

print(f"Klasifikacioni train: {len(train_local)} ({int(train_labels.sum())} fraud), "
      f"val: {len(val_local)} ({int(val_labels.sum())} fraud)")

results = {}
for dim in EMBED_DIMS:
    print(f"\n=== TRENING SKIP-GRAM EMBEDDING-A, dim={dim} ===")
    embeddings = train_skipgram(dim, n_sub, centers_t, contexts_t)

    x_train_emb = embeddings[train_local]
    x_val_emb = embeddings[val_local]

    print(f"  Klasifikator (BudgetMLP, 2 sloja) na {dim}-dim embedding-u:")
    model = BudgetMLP(dim, hidden_dims=(48, 24))
    n_params = sum(p.numel() for p in model.parameters())
    _, train_losses, val_losses, val_accs, val_f1s = train_model(
        model, x_train_emb, train_labels, x_val_emb, val_labels,
        epochs=20, batch_size=64, learning_rate=0.01,
    )
    results[dim] = dict(n_params=n_params, train_losses=train_losses, val_losses=val_losses,
                         val_accs=val_accs, val_f1s=val_f1s)
    print(f"  -> FINALNO dim={dim}: val_loss={val_losses[-1]:.4f}  val_acc={val_accs[-1]*100:.2f}%  val_f1={val_f1s[-1]:.4f}")

# ===========================================================
# 6. Baseline za poredjenje: isti cvorovi, ali sirovih 17 feature-a (bez embeddinga)
# ===========================================================
print("\n=== BASELINE: isti cvorovi, sirovih 17 feature-a (bez node2vec) ===")
x_full = data["x"]
orig_train = subgraph_nodes_orig[train_local]
orig_val = subgraph_nodes_orig[val_local]

x_train_raw = x_full[orig_train]
x_val_raw = x_full[orig_val]
mean = x_train_raw.mean(axis=0, keepdims=True)
std = x_train_raw.std(axis=0, keepdims=True)
std[std == 0] = 1.0
x_train_raw = (x_train_raw - mean) / std
x_val_raw = (x_val_raw - mean) / std

model_raw = BudgetMLP(17, hidden_dims=(48, 24))
_, train_losses_raw, val_losses_raw, val_accs_raw, val_f1s_raw = train_model(
    model_raw, x_train_raw, train_labels, x_val_raw, val_labels,
    epochs=20, batch_size=64, learning_rate=0.01,
)
print(f"BASELINE (17 raw feature-a): val_loss={val_losses_raw[-1]:.4f}  "
      f"val_acc={val_accs_raw[-1]*100:.2f}%  val_f1={val_f1s_raw[-1]:.4f}")

# ===========================================================
# 7. Jedan veliki plot: objasnjenje + poredjenje dimenzija + baseline
# ===========================================================
all_configs = [(f"node2vec dim={d}", results[d]) for d in EMBED_DIMS]
all_configs.append(("BASELINE (17 raw feat.)", dict(
    n_params=sum(p.numel() for p in model_raw.parameters()),
    train_losses=train_losses_raw, val_losses=val_losses_raw,
    val_accs=val_accs_raw, val_f1s=val_f1s_raw,
)))

fig = plt.figure(figsize=(18, 18))
gs = GridSpec(5, 3, height_ratios=[1.1, 1, 1, 1, 1], hspace=0.5, wspace=0.3)

ax_text = fig.add_subplot(gs[0, :])
ax_text.axis("off")
explanation = (
    f"node2vec: podgraf = fraud cvorovi + 1-hop susedi ({n_sub:,} cvorova, {len(centers):,} skip-gram parova)\n"
    f"Parametri walk-a: p={P}, q={Q}, num_walks={NUM_WALKS}, walk_length={WALK_LENGTH}, prozor={WINDOW}\n"
    f"Klasifikator: isti 2-slojni BudgetMLP (Linear->BatchNorm->ReLU->Dropout x2) kao ranije, na balansiranom 60/40 skupu\n"
    f"Klasifikacioni train: {len(train_local):,} ({int(train_labels.sum()):,} fraud), val: {len(val_local):,} ({int(val_labels.sum()):,} fraud)"
)
ax_text.text(0.01, 0.9, explanation, transform=ax_text.transAxes, fontsize=11, verticalalignment="top",
             family="monospace", bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

for row, (name, r) in enumerate(all_configs, start=1):
    ax_loss = fig.add_subplot(gs[row, 0])
    ax_loss.plot(r["train_losses"], label="train loss", color="tab:blue")
    ax_loss.plot(r["val_losses"], label="val loss", color="tab:orange")
    ax_loss.set_title(f"{name} -- Loss (params={r['n_params']:,})", fontsize=10)
    ax_loss.set_xlabel("Epoha")
    ax_loss.legend(fontsize=8)

    ax_acc = fig.add_subplot(gs[row, 1])
    ax_acc.plot(r["val_accs"], color="tab:green")
    ax_acc.set_title(f"{name} -- Val Accuracy", fontsize=10)
    ax_acc.set_xlabel("Epoha")
    ax_acc.set_ylim(0.4, 0.9)

    ax_f1 = fig.add_subplot(gs[row, 2])
    ax_f1.plot(r["val_f1s"], color="tab:red")
    ax_f1.set_title(f"{name} -- Val F1", fontsize=10)
    ax_f1.set_xlabel("Epoha")
    ax_f1.set_ylim(0, 0.9)

fig.suptitle("node2vec embedding (32/64/128 dim) vs sirovi feature-i -- ista klasifikaciona arhitektura", fontsize=14)
plt.savefig(OUT_DIR / "plot_node2vec_dim_comparison.png", dpi=110, bbox_inches="tight")
print(f"\nSlika sacuvana: {OUT_DIR / 'plot_node2vec_dim_comparison.png'}")

print("\n=== FINALNA TABELA ===")
for name, r in all_configs:
    print(f"{name:25s} val_loss={r['val_losses'][-1]:.4f}  val_acc={r['val_accs'][-1]*100:.2f}%  val_f1={r['val_f1s'][-1]:.4f}")

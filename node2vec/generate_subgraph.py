"""
    Produces a subgraph in the file subgraph_k{k}_N{N}.txt in the data folder.
"""

from pathlib import Path

import numpy as np
import networkx as nx

SEED = 42
K = 2
N = 5000

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "dgraphfin.npz"
OUT_PATH = REPO_ROOT / "data" / f"subgraph_k{K}_N{N}.txt"


def generate_k_hop_subgraph(g, ug, y, fraud_nodes, k, n, seed=SEED):
    rng = np.random.default_rng(seed)
    random_fraud_nodes = rng.choice(fraud_nodes, size=n, replace=False)

    k_hop_neighbors = set()
    for node in random_fraud_nodes:
        print("seed=", node)
        path_lengths = nx.single_source_shortest_path_length(ug, node, cutoff=k)
        k_hop_neighbors.update(path_lengths.keys())

    subgraph = g.subgraph(k_hop_neighbors)
    print("Total nodes in subgraph:", subgraph.number_of_nodes())
    print("Total edges in subgraph:", subgraph.number_of_edges())
    print("Normal nodes in subgraph:", len([n for n in subgraph.nodes if y[n] == 0]))
    print("Fraud nodes in subgraph:", len([n for n in subgraph.nodes if y[n] == 1]))
    print("Background nodes in subgraph:", len([n for n in subgraph.nodes if y[n] in (2, 3)]))

    events = []
    for src, dest in subgraph.edges:
        timestamp = g[src][dest]["timestamp"]
        events.append((src, dest, timestamp))
    events.sort(key=lambda e: e[2])

    with open(OUT_PATH, "w") as f:
        for src, dest, timestamp in events:
            f.write(f"{src} {dest} {timestamp}\n")
    print(f"saved: {OUT_PATH}")


def main():
    data = np.load(DATA_PATH)
    x, y = data["x"], data["y"]
    edge_index = data["edge_index"]
    edge_timestamp = data["edge_timestamp"]
    fraud_nodes = np.where(y == 1)[0]

    print("building full directed graph...")
    g = nx.DiGraph()
    for i in range(edge_index.shape[0]):
        g.add_edge(int(edge_index[i, 0]), int(edge_index[i, 1]), timestamp=int(edge_timestamp[i]))
    ug = g.to_undirected()
    print(f"full graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    generate_k_hop_subgraph(g, ug, y, fraud_nodes, K, N)


if __name__ == "__main__":
    main()

# node2vec eksperiment

Graf-embedding umesto sirovih 17 feature-a, da vidimo da li topologija sama nosi
bolji signal za fraud klasifikaciju. Koristi gotovu `node2vec` biblioteku
i `networkx` za graf.

## Fajlovi

- `embeddings.py` — gradi graf iz `subgraph_k30_N1.txt`, računa node2vec embedding
  u nekoliko dimenzija, pa poredi klasifikaciju na embeddingu, sirovim feature-ima,
  i njihovoj konkatenaciji.
- `embeddings_withRFcolRemoval.py` — gradi graf iz `subgraph_k30_N1.txt`, računa node2vec embedding
  i uklanja 5 najkarakteristicnijih feature-a (2, 6, 3, 8, 12).
- `generate_subgraph.py` — gradi podgraf poput `subgraph_k30_N1.txt`


## Metodologija

1. **Graf**: `subgraph_k30_N1.txt` (22,078 čvorova) - mali test-podgraf
   (30-hop okolina 1 fraud čvora). Sadrži samo 80 fraud čvorova - mala, seed-pristrasna
   populacija (videti napomenu ispod), ne menjamo dataset.
2. **node2vec**: `Node2Vec(graph, dimensions=8, walk_length=wl, num_walks=nw, p=1, q=0.125)`,
   sweep `wl ∈ {10,40}, nw ∈ {5,20}` — `dimensions=8` je najbolja od ranije probanih
   {8,16,32} (10x pravilo: parametara treba da bude bar 10x manje od broja trening
   primera). `p=1, q=0.125` po empirijskom izboru za ovakve grafove.
3. **Labele**: iz `dgraphfin.npz`, za čvorove iz podgrafa — fraud (1) i normal (0),
   background (2/3) isključen, balansirano na 40% fraud.
4. **Evaluacija**: 5-fold cross-validation umesto jednog train/val split-a — sa
   samo 80 fraud čvorova, jedan split ostavlja ~16 u val-u što je previše šumovito.
   ROC/AUC se računa preko out-of-fold predikcija (svaki čvor ocenjen samo modelom
   koji nije treniran na njemu), ne preko jednog fold-a.
5. **`engineered` feature-i**: top-5 kolona po Random Forest feature importance-u
   (2, 6, 3, 8, 12) + broj `-1` (missing) vrednosti po čvoru = 6-dim vektor, direktno
   iz `x` (bez učenja/embeddinga). Testirano samostalno i konkatenisano sa node2vec
   embeddingom.

## `plots/` sadržaj

- `embedding_vs_raw.png` — mean val_loss i val_F1 (5-fold CV) za sve varijante ulaza.
- `loss_f1_curves.png` — loss/F1 po epohi + ROC krive (out-of-fold) za `raw`,
  `engineered`, i najbolju node2vec varijantu.

## Dodatak: isti pristup na većem grafu (`embeddings_large_graph.py`)

Isti kod, ali graf = **svi 15,509 fraud čvorova + njihovi 1-hop susedi** (~43.6k
čvorova, iz punog grafa preko `dgraphfin.npz`, ne iz mentorskog `subgraph_k30_N1.txt`)
umesto malog, jednim semenom pristrasnog podgrafa. Balansiran skup: 15,423 primera
(6,169 fraud, tačno 40%) — 77× više podataka nego na malom podgrafu.

**Rezultat je potpuno drugačiji — node2vec sad dramatično pobeđuje:**

| ulaz | val_loss | val_F1 | AUC |
|---|---|---|---|
| raw (17-dim) | 0.581 | 0.605 | 0.731 |
| engineered (6-dim) | 0.585 | 0.612 | 0.719 |
| embedding (samo, dim=8) | 0.40-0.42 | 0.61 | — |
| **embedding+engineered (najbolji: wl=10,nw=20)** | **0.333** | **0.819** | **0.926** |


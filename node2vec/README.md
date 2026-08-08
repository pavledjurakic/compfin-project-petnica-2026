# node2vec eksperiment

Graf-embedding umesto sirovih 17 feature-a, da vidimo da li topologija sama nosi
bolji signal za fraud klasifikaciju. Koristi gotovu `node2vec` pip biblioteku
(gensim Word2Vec ispod haube) i `networkx` za graf — nema custom implementacije.

## Fajlovi

- `embeddings.py` — gradi graf iz `subgraph_k30_N1.txt`, računa node2vec embedding
  u nekoliko dimenzija, pa poredi klasifikaciju na embeddingu, sirovim feature-ima,
  i njihovoj konkatenaciji.

## Metodologija

1. **Graf**: `subgraph_k30_N1.txt` (22,078 čvorova) — mentorom zadat mali test-podgraf
   (30-hop okolina 1 fraud čvora). Sadrži samo 80 fraud čvorova — mala, seed-pristrasna
   populacija (videti napomenu ispod), ne menjamo dataset.
2. **node2vec**: `Node2Vec(graph, dimensions=d, p=1, q=0.125, workers=1)` za
   `d ∈ {8, 16, 32}` — male dimenzije jer je trening skup mali (10x pravilo:
   parametara treba da bude bar 10x manje od broja trening primera).
   `p=1, q=0.125` po empirijskom izboru za ovakve grafove.
3. **Labele**: iz `dgraphfin.npz`, za čvorove iz podgrafa — fraud (1) i normal (0),
   background (2/3) isključen, balansirano na 40% fraud.
4. **Evaluacija**: 5-fold cross-validation umesto jednog train/val split-a — sa
   samo 80 fraud čvorova, jedan split ostavlja ~16 u val-u što je previše šumovito.
5. **Tri varijante ulaza** po dimenziji: samo embedding, samo sirovi feature-i
   (baseline), i njihova konkatenacija.

## Nalaz

Ni jedna node2vec varijanta (nijedna dimenzija, ni konkatenacija) ne prestiže
raw-feature baseline na ovom podgrafu — loss raste sa dimenzijom, F1 opada.
Konzistentno sa slabom homofilijom izmerenom na punom grafu i sa time da je ovaj
konkretan podgraf premali/redak da node2vec izvuče koristan signal iz njega.

## `plots/` sadržaj

- `embedding_vs_raw.png` — mean val_loss i val_F1 (5-fold CV) za sve varijante ulaza.

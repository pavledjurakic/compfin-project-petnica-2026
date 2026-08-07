# pavle/ — glavni radni folder (aktuelna verzija)

Ovo je **konsolidovan, aktivan** deo projekta — sve što je trenutno relevantno za
fraud-detekciju na DGraphFin dataset-u. Sve ostalo van ovog foldera
(u `compfin-project-petnica-2026/` korenu — stare `explore_*.py`, `model_*.py`
skripte, `test.py`, itd.) je **arhiva/istorija eksperimentisanja**, nije
neophodna za razumevanje trenutnog stanja i ne treba je dirati niti se na nju
oslanjati.

## Podaci (izvan ovog foldera, ne diraj)

- `../../dgraphfin.npz` (tj. `CompFin - projekat/dgraphfin.npz`) — pun graf,
  3,700,550 čvorova, 17-dim feature-i, klase 0=normal/1=fraud/2,3=background.
  Ovo je **jedini** podatak koji sve skripte ovde stvarno koriste.
- `../../subgraph_k30_N1.txt` — mali test-podgraf (22,078 čvorova). Postoji
  helper (`common_data.load_small_subgraph_node_ids()`) koji ga zna pročitati,
  ali **trenutno ga nijedna skripta u ovom folderu ne poziva** — pun graf se
  uvek koristi direktno.

## Struktura

```
pavle/
├── common_data.py              -- DELJENA logika ucitavanja/split-a, uvoze je SVE ostale skripte
├── lab.py                      -- GLAVNI template: LogisticRegressionNet + MLPNet (3 bloka, 2.5k param)
├── model_budget_lr_sweep.py    -- MLPVariant + poredjenje "budzetiranog" (2.2k) vs originalnog (34k param) modela, sweep learning rate-ova
├── random_forest/
│   ├── model_random_forest.py           -- sklearn RandomForestClassifier baseline, rucne PR/ROC/AUC metrike
│   └── model_feature_distributions.py   -- fraud vs normal raspodela za svih 17 kolona
├── node2vec/
│   └── model_node2vec.py       -- node2vec od nule (biased walk + skip-gram u cistom PyTorch-u), embedding umesto sirovih feature-a
├── findings - number of layers comparison/
│   └── model_depth_comparison.py  -- 1 vs 2 vs 3 vs 5 blokova, jedna velika uporedna slika
├── findings_for_pptx/          -- gotove slike + notes.md, spremne za prezentaciju (nema koda)
└── plots/                      -- izlazni grafici raznih ranijih pokretanja
```

## Kako sve zajedno radi (za AI koji nastavlja rad)

1. **`common_data.py` je izvor istine za podatke.** `load_split()` uvek čita
   PUN graf iz `dgraphfin.npz` (apsolutna putanja u kodu), pravi balansiran
   subset (podrazumevano 60% normal / 40% fraud, SVIH 15,509 fraud čvorova +
   nasumičan uzorak normal), pa stratifikovan train/val split (80/20).
   `standardize()` radi z-score na osnovu SAMO train statistika.

2. **Sve `model_*.py` skripte u pod-folderima uvoze `common_data` preko**
   `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` — ovo
   pretpostavlja da je `common_data.py` **tačno jedan nivo iznad** te skripte
   (tj. direktno u `pavle/`). Ako se doda novi pod-folder sa skriptom koja
   treba `common_data` ili `model_budget_lr_sweep` (npr. `BudgetMLP`,
   `train_model`), mora ostati na toj istoj dubini (`pavle/<folder>/skripta.py`),
   inače import puca.

3. **Arhitekture su namerno male.** Postoji "10x pravilo": broj trening
   primera treba da bude bar 10x broj parametara modela, da se izbegne rizik
   od overfitting-a. `lab.py`-jeva `MLPNet` je zbog toga na 3 bloka (2,521
   parametara, ~31k trening primera → odnos ~12x). Ranija 5-slojna verzija
   (34k parametara) je namerno smanjena.

4. **Sve trening skripte prate isti obrazac**: `Linear -> BatchNorm1d -> ReLU
   -> Dropout(0.2)` blok, ponovljen N puta, pa čist `Linear` na kraju (sirov
   logit, ide u `BCEWithLogitsLoss`). `Adam` optimizator, `drop_last=True` u
   `DataLoader`-u (izbegava pucanje `BatchNorm1d` na batch-u velicine 1).

5. **Poznat, ponovljen nalaz kroz SVE pristupe** (logistička regresija, MLP
   raznih dubina/širina, Random Forest, node2vec embeddinzi): val loss/accuracy
   se dosledno zaglavljuje na ~0.58 loss / ~66-67% accuracy / ~0.60-0.62 F1 na
   balansiranom 60/40 skupu. Ovo je **informacioni plafon** (slab signal u
   feature-ima + slaba homofilija u grafu), NE problem arhitekture/kapaciteta
   — potvrđeno probom modela od 673 do 34,177 parametara, tri različita
   algoritma (logreg/MLP/RF), i node2vec strukturnim embedding-om (koji je
   ispao LOŠIJI od sirovih feature-a).

## Detaljan pisani izveštaj

Pun narativni izveštaj sa svim brojevima i zaključcima je u
`../IZVESTAJ.md` (jedan nivo iznad ovog foldera, u korenu
`compfin-project-petnica-2026/`).

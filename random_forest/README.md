# Random Forest eksperiment

Odvojen od glavne PyTorch neuronske mreže — ovde probamo **Random Decision Forest**
(sklearn `RandomForestClassifier`) na istom fraud/normal problemu, da vidimo kako se
klasičan ensemble-of-trees pristup poredi sa neuronskom mrežom na istim podacima.

## Fajlovi

- `random_forest.py` — glavna skripta. Učitava balansiran (60/40) train/val
  split (isti pristup kao `neural_network/lab.py`, radi fer poređenja), trenira
  Random Forest, i generiše sve plotove u `plots/`.
- `feature_distributions.py` — raspodela vrednosti po koloni, fraud vs normal
  (koje kolone/feature-e fraud čvorovi imaju sistematski drugačije od normal).

## Zašto Random Forest

Za razliku od neuronske mreže, Random Forest:
- ne zahteva standardizaciju feature-a (invarijantan na skalu)
- prirodno tretira `-1` missing kod kao samo još jednu vrednost po kojoj može da deli
  (stabla mogu naučiti "ako je feature==-1, idi ovim putem" bez posebne obrade)
- daje direktno tumačljivu **feature importance** metriku — korisno nam bilpo da dobijemo insight da li je 
  neka kolona zaista najvažnija vaznija.

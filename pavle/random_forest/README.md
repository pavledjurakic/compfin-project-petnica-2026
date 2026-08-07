# Random Forest eksperiment

Odvojen od glavne PyTorch neuronske mreže — ovde probamo **Random Decision Forest**
(sklearn `RandomForestClassifier`) na istom fraud/normal problemu, da vidimo kako se
klasičan ensemble-of-trees pristup poredi sa neuronskom mrežom na istim podacima.

## Fajlovi

- `model_random_forest.py` — glavna skripta. Učitava isti balansiran (60/40) train/val
  split preko `common_data.py` (deljeno sa NN eksperimentima, radi fer poređenja), trenira
  Random Forest, i generiše sve plotove u `plots/`.

## Sadržaj `plots/` foldera

- `plot_feature_importance.png` — koje od 17 kolona RF smatra najbitnijim za razdvajanje
  fraud/normal (direktno se nadovezuje na raniju analizu missing vrednosti po koloni)
- `plot_confusion_matrix.png` — konfuziona matrica na val skupu, prag=0.5
- `plot_roc_curve.png` — ROC kriva + AUC
- `plot_precision_recall_curve.png` — PR kriva (bitnija metrika s obzirom na neravnotežu)
- `plot_rf_vs_nn_comparison.png` — direktno poređenje RF-a sa MLP/logreg rezultatima iz
  glavne sesije (isti val skup, isti split)

## Zašto Random Forest

Za razliku od neuronske mreže, Random Forest:
- ne zahteva standardizaciju feature-a (invarijantan na skalu)
- prirodno tretira `-1` missing kod kao samo još jednu vrednost po kojoj može da deli
  (stabla mogu naučiti "ako je feature==-1, idi ovim putem" bez posebne obrade)
- daje direktno tumačljivu **feature importance** metriku — korisno da potvrdimo/opovrgnemo
  zaključke iz ranije analize missing vrednosti (npr. da li je kolona 10, jedina bez missing,
  zaista najvažnija)

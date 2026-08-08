# Neural network baseline

Logistic regression i mali MLP na sirovim 17 feature-a, treniran na balansiranom
(60/40 fraud/normal) subset-u punog grafa.

## Fajlovi

- `lab.py` — glavna skripta. `LogisticRegressionNet`, `MLPNet` (3 bloka
  Linear→BatchNorm1d→ReLU→Dropout, `hidden_dims=(48,24,12)`, ~2.5k parametara —
  namerno malo, po 10x pravilu za ovu veličinu trening skupa).
- `experiments/budget_vs_original.py` — poređenje "budžetiranog" (2.5k parametara)
  i predimenzionisanog (34k) modela kroz nekoliko learning rate vrednosti.
- `experiments/depth_comparison.py` — uticaj dubine mreže (1/2/3/5 blokova) na
  loss/accuracy/F1, sa objašnjenjem 10x pravila na samoj slici.

## Nalaz

Kroz sve testirane dubine/širine (673 do 34,177 parametara), rezultat se
dosledno zaglavljuje na ~0.58 val loss / ~66-67% accuracy — plafon je
informacioni (slab signal u feature-ima), ne arhitekturni.

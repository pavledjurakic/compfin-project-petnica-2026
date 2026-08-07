# node2vec eksperiment

Implementacija **node2vec** algoritma od nule (pristrasan 2nd-order random walk +
skip-gram sa negative sampling-om u čistom PyTorch-u, bez gotovih node2vec/gensim
biblioteka), da vidimo da li graf-embedding (umesto sirovih 17 feature-a) daje
bolji signal za fraud klasifikaciju.

## Fajlovi

- `model_node2vec.py` — sve u jednom: gradi podgraf, generiše walk-ove, trenira
  skip-gram embeddinge, pa ih koristi kao ulaz za isti 2-slojni klasifikator
  koji se pokazao dobro ranije (`BlockMLP` sa `hidden_dims=(48,24)`, unutar 10x
  parametarskog budžeta).

## Metodologija

1. **Podgraf**: svi fraud čvorovi (15,509) + njihovi direktni (1-hop) susedi iz
   punog grafa = 43,608 čvorova, ~32,474 ivica. Ovo je nužno — node2vec na celom
   grafu od 3.7M čvorova bi bio računski preskup za čist Python; podgraf i dalje
   sadrži SVE fraud čvorove i realnu topologiju oko njih.
2. **Random walk**: 2nd-order pristrasan walk sa parametrima **p=1, q=0.125**
   (po tvom empirijskom izboru za ovakve grafove — malo q znači da walk favorizuje
   "izlazak" dalje od početnog čvora umesto vraćanja, DFS-like ponašanje).
3. **Skip-gram + negative sampling**: standardan word2vec pristup, ovde primenjen
   na sekvence čvorova (walk = "rečenica", čvor = "reč"). Trenirano u PyTorch-u.
4. **Dimenzionalnost**: eksperiment "igranja" sa embedding dimenzijom —
   32, 64, i **128** (tvoj glavni izbor, industrijski standard).
5. **Klasifikacija**: za svaku dimenziju, embedding vektor (umesto `x`) ide u
   isti mali 2-slojni MLP, treniran/evaluiran na balansiranom 60/40 skupu
   (samo od čvorova koji su unutar podgrafa, jer samo oni imaju embedding).

## `plots/` sadržaj

- `plot_node2vec_dim_comparison.png` — jedna velika slika: loss/accuracy/F1 za
  svaku probanu dimenziju (32/64/128), plus poređenje sa baseline-om (sirovih 17
  feature-a) na ISTIM čvorovima, da vidimo da li embedding uopšte pomaže.

# 🔁 Cheat Sheet PyTorch — RNN / LSTM / GRU & classification de séquences

> Basée sur les notebooks `4_0_RNN-archi`, `4_2_char_rnn`, `4_3_rnn_classif_attn` (et `utils.py`).
> Convention : `import torch`, `from torch import nn`, `from torch.nn.utils.rnn import pad_sequence`

## Sommaire

1. [Modules récurrents : entrées / sorties](#1-modules-récurrents--entrées--sorties)
2. [RNN « from scratch »](#2-rnn--from-scratch-)
3. [Préparer des séquences : indices, padding, longueurs](#3-préparer-des-séquences--indices-padding-longueurs)
4. [Classifier une séquence (many-to-one)](#4-classifier-une-séquence-many-to-one)
5. [Récupérer le vrai dernier état / `pack_padded_sequence`](#5-récupérer-le-vrai-dernier-état--pack_padded_sequence)
6. [Attention avec masque](#6-attention-avec-masque)
7. [Classes déséquilibrées](#7-classes-déséquilibrées)
8. [Stabiliser l'entraînement](#8-stabiliser-lentraînement)
9. [Évaluer : top-k, matrice de confusion](#9-évaluer--top-k-matrice-de-confusion)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Modules récurrents : entrées / sorties

| Module | Paramètres essentiels | Sorties |
|---|---|---|
| `nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False)` | `input_size` = dim de $x_t$ ; `hidden_size` = dim de $h_t$ | `output, h_n` |
| `nn.LSTM(...)` | mêmes paramètres (pas de `nonlinearity`) | `output, (h_n, c_n)` ⚠️ tuple |
| `nn.GRU(...)` | idem | `output, h_n` |

| Tenseur | Forme (`batch_first=False`, défaut) | Forme (`batch_first=True`) |
|---|---|---|
| entrée `x` | `(T, B, input_size)` | `(B, T, input_size)` |
| `output` = tous les $h_t$ de la **dernière couche** | `(T, B, D·hidden)` | `(B, T, D·hidden)` |
| `h_n` (et `c_n`) = dernier état de **chaque couche** | `(D·num_layers, B, hidden)` | **toujours** `(D·num_layers, B, hidden)` |

- $D = 2$ si `bidirectional=True`, sinon 1. État initial optionnel : `model(x, h0)` (zéros par défaut).

```python
model = nn.RNN(input_size=100, hidden_size=50)
seq_x = torch.rand(23, 10, 100)          # T=23, batch=10, dim=100
seq_h, last_h = model(seq_x)             # (23, 10, 50), (1, 10, 50)
# pour classifier : last_h[-1] → (10, 50)  (dernière couche)

lstm = nn.LSTM(100, 50, num_layers=2, dropout=0.3, bidirectional=True)
out, (h_n, c_n) = lstm(seq_x)            # out (23, 10, 100) ; h_n (4, 10, 50)
h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)   # avant + arrière de la dernière couche → (10, 100)
```

| Choix | Conseil |
|---|---|
| RNN / GRU / LSTM | RNN simple : séquences courtes ; **LSTM/GRU par défaut** (GRU : moins de paramètres, souvent aussi bon). |
| `hidden_size` | 64–256 ; `num_layers` 1–3. |
| `dropout` | n'agit **qu'entre couches** (`num_layers > 1`). |
| `bidirectional=True` | classification / étiquetage (on a toute la séquence) ; **jamais** pour générer ou prédire le futur. |

---

## 2. RNN « from scratch »

Comprendre la récurrence : une cellule = deux couches linéaires + tanh, appliquées **pas à pas** dans une boucle **à l'intérieur** du `forward`.

```python
class RNN(nn.Module):                                 # version utils.py
    def __init__(self, inputdim, latentdim):
        super().__init__()
        self.inputdim, self.latentdim = inputdim, latentdim
        self.encoder = nn.Linear(inputdim, latentdim)   # W1 : x_t → h
        self.latent  = nn.Linear(latentdim, latentdim)  # W2 : h_{t-1} → h
    def one_step(self, x, h):
        return torch.tanh(self.encoder(x) + self.latent(h))
    def hzero(self, batch_size):
        return torch.zeros(batch_size, self.latentdim)
    def forward(self, x, h=None):                       # x : (T, B, dim)
        if h is None:
            h = self.hzero(x.shape[1]).to(x.device)
        hseq = []
        for t in range(x.shape[0]):
            h = self.one_step(x[t], h)
            hseq.append(h)
        return torch.stack(hseq)                        # (T, B, latent)

rnn, decoder = RNN(2, 10), nn.Linear(10, 10)            # le décodeur W3 est séparé
optim = torch.optim.Adam(chain(rnn.parameters(), decoder.parameters()), lr=1e-4)
yhat = decoder(rnn(x)[-1])                               # many-to-one : dernier état
```

> ⚠️ Passer toute la séquence `(T, B, d)` à une cellule non récurrente (`RNN_V1`) **ne plante pas** (les `Linear` s'appliquent sur la dernière dim) mais **ignore la récurrence** : chaque pas est traité indépendamment.
> 💡 `itertools.chain(m1.parameters(), m2.parameters())` : un seul optimiseur pour deux modules séparés.

---

## 3. Préparer des séquences : indices, padding, longueurs

| Étape | Code | Remarque |
|---|---|---|
| Caractère → indice | `all_letters.find(c)` ; `torch.tensor([...], dtype=torch.long)` | Normaliser l'Unicode (`unicodedata.normalize('NFD', s)`, enlever les accents). |
| Mot → indice (GloVe) | `word2id.get(w, OOVID)` | Mot inconnu → indice `OOV`. |
| Dataset | `__getitem__` → `(tenseur d'indices, label)` | séquences de **longueurs différentes**. |
| `collate_fn` | `pad_sequence(seqs, batch_first=False, padding_value=PAD)` + `lengths` | Renvoie `(x (T_max, B), lengths (B,), labels (B,))`. |
| Split | `random_split(ds, [n_train, n_test])` | |

```python
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    x = pad_sequence(sequences, batch_first=False, padding_value=PAD)   # (T_max, B)
    return x, lengths, torch.tensor(labels)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
```

| Représentation d'entrée | Code | Usage |
|---|---|---|
| One-hot figé via un Embedding | `self.emb = nn.Embedding(n, n)` ; `self.emb.weight.data = torch.eye(n)` ; `self.emb.weight.requires_grad = False` | Caractères (n ≈ 60) : one-hot sans le calculer à la main. |
| Embedding appris | `nn.Embedding(n_voc, d, padding_idx=PAD)` | Tokens / mots. |
| Embedding pré-entraîné (GloVe) | `nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float), freeze=True)` | ⚠️ GloVe chargé en numpy **float64** → convertir en float32. |

---

## 4. Classifier une séquence (many-to-one)

```python
class RNNClassif(nn.Module):
    def __init__(self, n_voc, emb_dim, hidden_size, n_classes, pad_idx):
        super().__init__()
        self.emb = nn.Embedding(n_voc, emb_dim, padding_idx=pad_idx)
        self.rec = nn.LSTM(emb_dim, hidden_size)               # ou nn.RNN / nn.GRU
        self.h2o = nn.Linear(hidden_size, n_classes)
    def forward(self, x, lengths=None):                        # x : (T, B)
        hidden, (last, c) = self.rec(self.emb(x))              # hidden (T,B,H) ; last (1,B,H)
        if lengths is not None:                                # vrai dernier état (hors pad)
            last = hidden[lengths - 1, torch.arange(x.size(1))]   # (B, H)
        else:
            last = last[-1]
        return self.h2o(last), hidden                          # scores bruts (B, n_classes)
```

> 👉 `nn.CrossEntropyLoss()` sur les scores bruts → **pas** de `LogSoftmax` dans le modèle (sinon utiliser `NLLLoss`).
> ⚠️ `self.h2o(last)` avec `last` de forme `(1, B, H)` donne `(1, B, C)` : `CrossEntropyLoss` attend `(B, C)` → `last[-1]` ou `.squeeze(0)`.

---

## 5. Récupérer le vrai dernier état / `pack_padded_sequence`

Avec du padding **à la fin**, `h_n` est l'état après avoir lu les `<pad>` : biaisé pour les séquences courtes.

| Solution | Code | Remarque |
|---|---|---|
| Indexation avancée | `hidden[lengths - 1, torch.arange(B)]` | Simple, `hidden` de forme `(T, B, H)`. |
| Boucle (TP) | `torch.stack([hidden[lengths[i]-1, i] for i in range(B)])` | Lisible mais lent. |
| **Séquences empaquetées** 🔎 | `pack_padded_sequence(emb, lengths.cpu(), enforce_sorted=False)` → `rec(packed)` → `h_n` | Le RNN **s'arrête** au vrai dernier pas de chaque séquence (pas de calcul sur le pad) ; `pad_packed_sequence(out)` pour retrouver `(T, B, H)`. |

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
packed = pack_padded_sequence(self.emb(x), lengths.cpu(), enforce_sorted=False)
out_packed, h_n = self.rec(packed)          # h_n[-1] = vrai dernier état (B, H)
out, _ = pad_packed_sequence(out_packed)    # (T, B, H), zéros sur le pad
```

> ⚠️ `lengths` doit être sur **CPU** (tenseur `int64` ou liste).

---

## 6. Attention avec masque

Remplacer « le dernier état » par une **moyenne pondérée apprise** de tous les états :

$$\alpha_t = \mathrm{softmax}_t\big(\langle a, h_t\rangle\big), \qquad e = \sum_t \alpha_t h_t$$

```python
self.attn = nn.Linear(hidden_size, 1, bias=False)          # vecteur d'attention a (la query)

def forward(self, x, lengths):
    hidden, _ = self.rec(self.emb(x))                       # (T, B, H)
    T, B = x.shape
    scores = self.attn(hidden).squeeze(-1)                  # (T, B) : <a, h_t>
    mask = torch.arange(T).unsqueeze(1).expand(T, B) < lengths.unsqueeze(0)   # True = vrai mot
    scores = scores.masked_fill(~mask.to(x.device), float('-inf'))            # pad → -inf
    alpha = torch.softmax(scores, dim=0)                    # softmax sur le TEMPS (dim 0)
    e = (alpha.unsqueeze(-1) * hidden).sum(0)               # (B, H)
    return self.h2o(e), alpha
```

| Variante | Calcul de $\alpha$ |
|---|---|
| Uniforme (Q1 du TP) | $\alpha_t = 1/L$ sur les vrais mots = moyenne masquée |
| Globale (Q2) | $\mathrm{softmax}(\langle a, h_t\rangle)$, $a$ appris |
| Dépendante de l'entrée 🔎 | query calculée à partir de la séquence (ex. $q = W\bar h$) ou d'un décodeur |

> ⚠️ `masked_fill` avec un masque **inversé** (`~mask`) : on met `-inf` là où il y a du **padding**. Softmax sur la **bonne dimension** (le temps). Vérifier : `alpha.sum(0) == 1` et `alpha == 0` sur le pad.
> 💡 Renvoyer `alpha` permet de **visualiser** quels mots ont compté pour la décision.

---

## 7. Classes déséquilibrées

Noms par langue : 8 434 russes contre 65 portugais → le modèle prédit « russe » partout.

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `torch.bincount(labels)` | | Effectif de chaque classe. |
| `WeightedRandomSampler(weights, num_samples, replacement=True)` | `weights` = **un poids par exemple** (pas par classe) | Sur-échantillonne les classes rares : batchs équilibrés. |
| `DataLoader(..., sampler=sampler)` | ⚠️ **incompatible avec `shuffle=True`** | |
| `nn.CrossEntropyLoss(weight=w_classes)` | un poids **par classe** (tenseur float, même device) | Alternative : pondérer la loss plutôt que les tirages. |

```python
counts = torch.bincount(torch.tensor(train_labels))
class_w = 1.0 / counts.float()                              # inverse de la fréquence
sample_w = class_w[torch.tensor(train_labels)]              # poids de CHAQUE exemple
sampler = WeightedRandomSampler(sample_w, num_samples=len(train_labels), replacement=True)
train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, collate_fn=collate_fn)
```

> 👉 Évaluer alors avec l'accuracy **équilibrée** / F1 macro / la matrice de confusion, pas l'accuracy brute.

---

## 8. Stabiliser l'entraînement

| Outil | Code | Quand |
|---|---|---|
| Gradient clipping | `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)` entre `backward()` et `step()` | Toujours avec des RNN (exploding gradient, NaN). |
| Surveiller la norme du gradient | `with torch.no_grad(): g2 = sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)` → `writer.add_scalar("grad", g2, step)` | Diagnostiquer explosion / disparition. |
| `tanh` / LSTM / GRU | | Saturation, vanishing. |
| Dropout | `nn.LSTM(..., num_layers≥2, dropout=0.3)` / `nn.Dropout` avant `h2o` | Sur-apprentissage. |
| Longueur max | tronquer les séquences (`seq[:500]`) | Mémoire et vanishing. |

---

## 9. Évaluer : top-k, matrice de confusion

```python
with torch.no_grad():
    yhat, _ = rnn(lineToTensor("Satoshi").unsqueeze(1).to(device))   # (T,) → (T, 1) : batch de 1
    top = yhat.squeeze().topk(5).indices                              # 5 langues les plus probables

confusion = torch.zeros(n_cat, n_cat)
for x, l, y in test_loader:
    guess = rnn(x.to(device), l)[0].argmax(1).cpu()
    for t, p in zip(y, guess): confusion[t, p] += 1                  # ⚠️ confusion[y, guess] += 1 ne cumule pas les doublons
confusion = confusion / confusion.sum(1, keepdim=True)               # normaliser par ligne (vraie classe)
plt.matshow(confusion.numpy())
```

> 🔎 `sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')` et `ConfusionMatrixDisplay` font la même chose proprement.

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Entrée `(B, T, d)` avec `batch_first=False` | résultats absurdes sans erreur (batch et temps inversés) | respecter `(T, B, d)` ou `batch_first=True` |
| `h_n` supposé `(B, T, H)` avec `batch_first=True` | erreurs de forme | `h_n` est **toujours** `(couches·D, B, H)` |
| `out, h = lstm(x)` | `h` est un tuple `(h_n, c_n)` | `out, (h, c) = lstm(x)` |
| Utiliser `h_n` avec du padding final | séquences courtes mal représentées | `hidden[lengths-1, arange(B)]` ou `pack_padded_sequence` |
| Softmax d'attention sans masque | poids sur les `<pad>` | `masked_fill(~mask, -inf)` |
| Softmax sur la mauvaise dimension | poids qui somment à 1 sur le batch | `dim` = dimension du **temps** |
| `LogSoftmax` + `CrossEntropyLoss` | double log-softmax | scores bruts avec CE, ou `NLLLoss` |
| Sortie `(1, B, C)` dans la CE | `Expected target size` | `last[-1]` / `.squeeze(0)` |
| `WeightedRandomSampler` + `shuffle=True` | `ValueError` | l'un ou l'autre |
| Poids par **classe** donnés au sampler | tirages incohérents | un poids **par exemple** |
| Embeddings GloVe float64 | `expected Float but found Double` | `torch.tensor(emb, dtype=torch.float)` |
| `dropout` avec `num_layers=1` | warning, aucun effet | empiler ou `nn.Dropout` explicite |
| Pas de clipping | loss → NaN | `clip_grad_norm_` |
| `lengths` sur GPU dans `pack_padded_sequence` | erreur | `lengths.cpu()` |
| `print` dans le `forward` (TP) | logs énormes pendant l'entraînement | retirer les `print` de debug |

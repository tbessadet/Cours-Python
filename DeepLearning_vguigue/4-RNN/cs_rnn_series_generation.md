# 📈 Cheat Sheet PyTorch — RNN pour séries temporelles & génération de texte

> Basée sur les notebooks `4_1_RNN-metro` (flux du métro de Hangzhou) et `4_4_token-generative` (génération de discours, tokens WordPiece).
> Convention : `import torch`, `from torch import nn`, `from torch.utils.data import Dataset, DataLoader`

## Sommaire

**I. Séries temporelles**
1. [Mettre en forme des séries en Dataset](#1-mettre-en-forme-des-séries-en-dataset)
2. [Classification de séries (many-to-one)](#2-classification-de-séries-many-to-one)
3. [Prévision (many-to-many, teacher forcing)](#3-prévision-many-to-many-teacher-forcing)
4. [Prévoir à plusieurs pas (génération autorégressive)](#4-prévoir-à-plusieurs-pas-génération-autorégressive)
5. [Métriques de régression pour séries](#5-métriques-de-régression-pour-séries)

**II. Génération de texte**

6. [Tokenizer WordPiece (Hugging Face `tokenizers`)](#6-tokenizer-wordpiece-hugging-face-tokenizers)
7. [Dataset de phrases et padding](#7-dataset-de-phrases-et-padding)
8. [Modèle de langue récurrent + loss masquée](#8-modèle-de-langue-récurrent--loss-masquée)
9. [Décoder : glouton, échantillonnage, beam search](#9-décoder--glouton-échantillonnage-beam-search)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Séries temporelles

## 1. Mettre en forme des séries en Dataset

Données métro : tenseur `(D jours, T=73 quarts d'heure, S=80 stations, 2 flux in/out)`.

| Étape | Code | Remarque |
|---|---|---|
| Charger | `matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))` | Sous-échantillonner : `matrix[:, :, :CLASSES, :DIM_INPUT]`. |
| **Normaliser** par le max | `stations_max = data.view(-1, S, 2).max(0)[0]` ; `data / stations_max` | ⚠️ calculer le max **sur le train** et le **réutiliser** pour le test (`stations_max=ds_train.stations_max`). |
| Fenêtres glissantes | `data[day, t:t+length, station]` | Chaque (jour, début, station) = un exemple → `len = S·D·(T − length)`. |
| Classification | `__getitem__ → (séquence (length, 2), station)` | prédire la station (= la « classe »). |
| Prévision | `__getitem__ → (data[d, t:t+L-1], data[d, t+1:t+L])` | $X$ et $Y$ décalés d'un pas, toutes les stations : `(L-1, S, 2)`. |

```python
ds_train = SampleMetroDataset(matrix_train[:, :, :10, :2], length=20)
ds_test  = SampleMetroDataset(matrix_test[:, :, :10, :2], length=20, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=32, shuffle=True)
x, y = next(iter(data_train))        # x : (32, 20, 2) → BATCH FIRST !
```

> ⚠️ Le DataLoader empile en `(B, T, d)` alors que le RNN de `utils.py` attend `(T, B, d)` → **`x.transpose(0, 1)`** (ou `permute(1, 0, 2)`).

---

## 2. Classification de séries (many-to-one)

```python
rnn, decoder = RNN(dim_input, latent), nn.Linear(latent, classes)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(chain(rnn.parameters(), decoder.parameters()), lr=1e-4)

for epoch in range(epochs):
    for x, y in data_train:
        x, y = x.transpose(0, 1).to(device), y.to(device)    # (T, B, d)
        h = rnn(x)                                           # (T, B, latent)
        yhat = decoder(h[-1])                                # décodeur branché sur le DERNIER état
        l = loss(yhat, y)
        optim.zero_grad(); l.backward(); optim.step()
```

> 💡 Le décodeur étant **séparé** du RNN, c'est à la boucle de choisir où le brancher : dernier état (classification) ou tous les états (prévision).
> 👉 Comparer accuracy train/test ; sur-apprentissage → early stopping, dropout, moins de dimensions latentes, plus de données (plus de stations / jours).

---

## 3. Prévision (many-to-many, teacher forcing)

À partir des flux jusqu'à $t$ pour toutes les stations, prédire $t+1$. Un **RNN commun** à toutes les stations : entrée $\mathbb R^{n\times 2}$ → sortie $\mathbb R^{n\times 2}$.

| Changement par rapport à la classification | Pourquoi |
|---|---|
| Décodeur `nn.Linear(latent, dim_input)` (ou `latent → S·2`) | on prédit des **valeurs**, pas une classe |
| **`nn.MSELoss`** (ou L1) | régression ; la cross-entropie n'a plus de sens |
| Décodage sur **tous** les états `decoder(h)` | supervision à **chaque** pas (many-to-many) |

```python
# Option A : traiter chaque station comme une série indépendante (RNN partagé)
for x, y in data_train:                         # x, y : (B, L-1, S, 2)
    B, T, S, d = x.shape
    x = x.permute(1, 0, 2, 3).reshape(T, B * S, d)       # (T, B·S, 2)
    y = y.permute(1, 0, 2, 3).reshape(T, B * S, d)
    yhat = decoder(rnn(x))                                # (T, B·S, 2) : une prédiction par pas
    l = nn.MSELoss()(yhat, y)
```

> 💡 La loss est calculée à chaque pas, mais la rétro-propagation n'a lieu qu'une fois la séquence vue : au temps $t$, le gradient combine l'erreur à $t$ **et** celles des pas suivants (BPTT, *backpropagation through time*).
> 👉 Horizon : pour prédire $t+2$, décaler $Y$ de 2 pas, ou utiliser la génération autorégressive (§4).

---

## 4. Prévoir à plusieurs pas (génération autorégressive)

On **réinjecte** la prédiction comme entrée du pas suivant (*free generation*) :

```python
def forecast(rnn, decoder, x, h=None, length=10):
    with torch.no_grad():
        if h is None:
            h = rnn.hzero(x.size(1)).to(x.device)
        h = rnn(x, h)[-1]                    # encoder l'historique
        x = decoder(h)                       # 1re prédiction (t+1)
        yhat = [x]
        for _ in range(length - 1):
            h = rnn.one_step(x, h)           # ⚠️ mettre à jour h (le code du TP l'oublie)
            x = decoder(h)
            yhat.append(x)
    return torch.stack(yhat)                 # (length, B, d)
```

> ⚠️ Les erreurs **s'accumulent** avec l'horizon : tracer la métrique en fonction de l'horizon ($t+1, t+2, \dots$).
> 💡 Entraîner d'abord en **teacher forcing**, puis mélanger des pas en génération libre (*curriculum / scheduled sampling*) améliore la prévision long terme.

---

## 5. Métriques de régression pour séries

| Métrique | Formule | À savoir |
|---|---|---|
| MSE / RMSE | $\frac1n\sum(y-\hat y)^2$ ; $\sqrt{\cdot}$ | sensible aux gros écarts ; RMSE dans l'unité des données |
| MAE | $\frac1n\sum\lvert y-\hat y\rvert$ | robuste |
| **MAPE** | $\frac{100}{n}\sum\left\lvert\frac{y-\hat y}{y}\right\rvert$ | ⚠️ **explose** quand $y \approx 0$ (heures creuses, nuit) |
| **SMAPE** | $\frac{100}{n}\sum\frac{\lvert y-\hat y\rvert}{(\lvert y\rvert+\lvert\hat y\rvert)/2}$ | bornée (0–200 %), mieux adaptée aux faibles affluences |

```python
def smape(y, yhat, eps=1e-8):
    return 100 * torch.mean(torch.abs(y - yhat) / ((y.abs() + yhat.abs()) / 2 + eps))
```

> 👉 Calculer les métriques sur des valeurs **dé-normalisées** (`× stations_max`) pour les interpréter, et comparer à une **baseline naïve** (prédire $y_{t+1} = y_t$, ou la moyenne du même créneau).

---

# II. Génération de texte

## 6. Tokenizer WordPiece (Hugging Face `tokenizers`)

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `Tokenizer(WordPiece(unk_token="[UNK]"))` | | Tokenizer en **sous-mots** (comme BERT). Autres modèles : `BPE`, `Unigram`. |
| `tokenizer.pre_tokenizer = Whitespace()` | | Découpage préalable sur espaces / ponctuation. |
| `WordPieceTrainer(vocab_size=10000, special_tokens=[...])` | `vocab_size` ; tokens spéciaux `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]` | Taille du vocabulaire = compromis longueur des séquences / taille de la couche de sortie. |
| `tokenizer.train([fichier], trainer)` | | Apprend le vocabulaire sur le corpus. |
| `tokenizer.encode(txt)` → `.ids`, `.tokens` | | `"francais"` → `['fr', '##anc', '##ais']` (`##` = suite de mot). |
| `tokenizer.decode(ids)` | | Reconstruit le texte (recolle les `##`). |
| `tokenizer.token_to_id("[PAD]")` | | Indice d'un token spécial (plus sûr que `encode("[PAD]").ids[0]`). |
| `tokenizer.save("tok.json")` / `Tokenizer.from_file(...)` | | Le tokenizer est **indissociable** du modèle entraîné : le sauvegarder avec lui. |

> 💡 Un tokenizer appris sur un corpus anglais découpe beaucoup plus le français ou le vocabulaire technique (plus de tokens par mot) : **inégalités entre langues**.

---

## 7. Dataset de phrases et padding

```python
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, maxsent=None, maxlen=None):
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +', ' ', p[:maxlen]).strip() + "."
                        for p in text.split(".") if len(p.strip()) > 0][:maxsent]
        self.phrasesnum = [torch.tensor(tokenizer.encode(p).ids) for p in self.phrases]
    def __len__(self): return len(self.phrases)
    def __getitem__(self, i): return self.phrasesnum[i]       # pré-tokenisé une fois pour toutes

def collate_fn(batch):
    lengths = torch.tensor([len(s) for s in batch])
    return pad_sequence(batch, batch_first=False, padding_value=PAD), lengths   # (T_max, B)

data = DataLoader(ds, collate_fn=collate_fn, batch_size=64, shuffle=True)
```

> 💡 Le « . » (token `EOS`) termine chaque phrase : le modèle apprend **quand s'arrêter**.

---

## 8. Modèle de langue récurrent + loss masquée

```python
class RNNLM(nn.Module):
    def __init__(self, input_size, hidden_size, n_voc):
        super().__init__()
        self.emb = nn.Embedding(n_voc, input_size)
        self.rec = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, n_voc)       # classification sur le vocabulaire
    def forward(self, x, hid=None):                        # x : (T, B)
        out, last = self.rec(self.emb(x), hid)             # out (T, B, H)
        return self.decoder(out), last                     # (T, B, n_voc) : une prédiction par pas
```

Teacher forcing : `y = x[1:]`, `x = x[:-1]` (cible = entrée décalée d'un pas).

| Loss masquée | Code |
|---|---|
| **Le plus simple** | `nn.CrossEntropyLoss(ignore_index=PAD)(yhat.view(-1, V), y.view(-1))` |
| À la main | `l = nn.CrossEntropyLoss(reduction='none')(yhat.view(-1, V), y.view(-1))` ; `mask = (y.view(-1) != PAD)` ; `(l * mask).sum() / mask.sum()` |

> ⚠️ Le `maskedCrossEntropy` du TP utilise `CrossEntropyLoss()` avec la réduction par défaut `'mean'` : elle renvoie **un scalaire déjà moyenné sur tout (pad inclus)** ; multiplier ensuite par le masque n'enlève rien. Il faut **`reduction='none'`** (ou `ignore_index`).
> ⚠️ `CrossEntropyLoss` attend `(N, C)` et `(N,)` : aplatir temps et batch (`view(-1, V)`), ou passer `yhat.permute(1, 2, 0)` `(B, C, T)` avec `y.T` `(B, T)`.

```python
for x, lengths in data:
    y, x = x[1:], x[:-1]
    yhat, _ = model(x.to(device))
    l = nn.CrossEntropyLoss(ignore_index=PAD)(yhat.view(-1, yhat.size(-1)), y.to(device).view(-1))
    optim.zero_grad(); l.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
```

---

## 9. Décoder : glouton, échantillonnage, beam search

| Méthode | Choix du token suivant | Résultat |
|---|---|---|
| **Glouton** (argmax) | `y[-1].argmax(-1)` | déterministe, répétitif (boucles « the the the ») |
| **Échantillonnage** | `torch.multinomial(torch.softmax(y[-1] / T, dim=-1), 1)` | varié ; température $T<1$ = plus sûr, $T>1$ = plus créatif |
| Top-k / top-p 🔎 | échantillonner parmi les $k$ meilleurs / la masse $p$ | compromis qualité / diversité |
| **Beam search** | garder les $k$ meilleures séquences (somme des log-probas) | phrases plus probables ; pour la traduction / le résumé |

```python
@torch.no_grad()
def generate(model, tokenizer, eos, start="", maxlen=200, temperature=1.0):
    x = torch.tensor(tokenizer.encode(start).ids).unsqueeze(1)        # (T, 1)
    y, h = model(x)                                                   # encoder le début
    for _ in range(maxlen):
        probs = torch.softmax(y[-1] / temperature, dim=-1)            # dernier pas, (1, V)
        c = torch.multinomial(probs, 1)                               # (1, 1)
        x = torch.cat((x, c), 0)
        if c.item() == eos: break
        y, h = model(c, h)                                            # 1 pas, en gardant l'état
    return tokenizer.decode(x.squeeze().tolist())
```

**Beam search** : à chaque pas, pour chaque candidat `(h, séquence, score)`, prendre les `k` meilleurs tokens (`torch.log_softmax(y, -1).topk(k)`), former les $k^2$ extensions, **garder les $k$ meilleures** ; un candidat terminé par `EOS` est conservé tel quel.

Normalisation de longueur (OpenNMT), hypothèses : score = somme des log-probas, $|Y|$ longueur :

$$lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}, \qquad \text{score}(Y) = \frac{\log P(Y)}{lp(Y)}, \quad \alpha \approx 0{,}7$$

> ⚠️ Passer l'**état caché** `h` d'un pas au suivant (et ne donner que le **nouveau** token), sinon on recalcule tout (lent) ou on perd le contexte. Pour un LSTM, `h` est le **tuple** `(h, c)`.
> 💡 Utiliser `log_softmax` et **additionner** les log-probas (multiplier des probas → *underflow*).

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Batch `(B, T, d)` donné à un RNN `(T, B, d)` | apprentissage incohérent, sans erreur | `x.transpose(0, 1)` |
| Normalisation calculée sur le test | fuite d'information, scores irréalistes | réutiliser `stations_max` du train |
| CrossEntropy pour une prévision de valeurs | non-sens | `MSELoss` / `L1Loss` |
| MAPE en heures creuses | métrique infinie / énorme | SMAPE, MAE |
| Pas de baseline naïve | impossible de juger le modèle | comparer à $\hat y_{t+1} = y_t$ |
| `forecast` sans mettre à jour `h` | prédictions qui stagnent | `h = rnn.one_step(x, h)` |
| Loss non masquée sur le padding | le modèle apprend à prédire `[PAD]` | `ignore_index=PAD` / `reduction='none'` + masque |
| `y = x[1:]` mais `x` non tronqué | formes `T` et `T-1` incompatibles | `x = x[:-1]` |
| Génération en argmax | répétitions en boucle | échantillonnage (température) / beam search |
| Produit de probabilités en beam search | underflow → 0 | somme des `log_softmax` |
| Tokenizer non sauvegardé avec le modèle | indices incompatibles au rechargement | `tokenizer.save(...)` |
| Pas de clipping | NaN | `clip_grad_norm_` |

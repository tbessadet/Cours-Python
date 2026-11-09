# 📝 Cheat Sheet PyTorch — Texte : tokenisation, embeddings de mots & CNN 1D

> Basée sur le notebook `3_3-Repr` (classification de sentiment IMDB).
> Convention : `import torch`, `from torch import nn`, `import torchtext`, `from torch.nn.utils.rnn import pad_sequence`

## Sommaire

1. [Tokeniser et construire un vocabulaire](#1-tokeniser-et-construire-un-vocabulaire)
2. [Padding et `collate_fn` pour du texte](#2-padding-et-collate_fn-pour-du-texte)
3. [Modèle moyen (sac d'embeddings)](#3-modèle-moyen-sac-dembeddings)
4. [Embeddings pré-entraînés & fine-tuning](#4-embeddings-pré-entraînés--fine-tuning)
5. [CNN 1D sur du texte](#5-cnn-1d-sur-du-texte)
6. [Portée (champ récepteur) et sous-séquences qui activent un filtre](#6-portée-champ-récepteur-et-sous-séquences-qui-activent-un-filtre)
7. [Tester et inspecter le modèle](#7-tester-et-inspecter-le-modèle)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Tokeniser et construire un vocabulaire

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torchtext.data.utils.get_tokenizer("basic_english")` | `"basic_english"`, `"spacy"` | Découpe une phrase en mots (minuscules, ponctuation séparée). |
| `torchtext.vocab.build_vocab_from_iterator(iter, specials=['<pad>','<oov>'], min_freq=20)` | `specials` en premier (indices 0, 1…) ; `min_freq` : fréquence minimale pour garder un mot | Vocabulaire mot → indice. `min_freq` élevé = vocabulaire plus petit (13 352 mots pour IMDB à 20). |
| `vocab['the']` / `vocab.get_itos()` | | Mot → indice ; liste indice → mot (pour reconstruire une phrase). |
| `vocab.set_default_index(vocab['<oov>'])` | 🔎 | Mots inconnus → `<oov>` au lieu d'une erreur. |
| `torchtext.datasets.IMDB(root, split=('train','test'))` | | 25k/25k critiques ; éléments `(label, texte)` (labels 1/2). |
| Générateur `yield` | `def yield_imdb(): for _, l in data: yield tokenizer(l)` | Produit les tokens à la volée sans tout stocker. |

```python
tokenizer = get_tokenizer("basic_english")
vocab = torchtext.vocab.build_vocab_from_iterator(
    (tokenizer(t) for _, t in train_list), specials=['<pad>', '<oov>'], min_freq=20)
vocab.set_default_index(vocab['<oov>'])
ids = torch.tensor([vocab[t] for t in tokenizer("the cat sat on the mat")])
```

> ⚠️ 🔎 `torchtext` n'est plus maintenu (dernière version 0.18, incompatible avec les PyTorch récents). Alternatives : un `Counter` + dictionnaire maison, ou les tokenizers **Hugging Face** (`AutoTokenizer`, sous-mots, cf. module 5) et `datasets.load_dataset("imdb")`.

---

## 2. Padding et `collate_fn` pour du texte

Les phrases n'ont pas la même longueur → on complète par `<pad>` (indice 0, vecteur nul) pour former un tenseur `(B, L_max)`.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `pad_sequence(liste_de_tenseurs, batch_first=False, padding_value=0)` | ⚠️ **`batch_first=True`** pour obtenir `(B, L)` | Sinon sortie `(L, B)`. Padding **à la fin**. |
| `collate_fn` | | Tokenise, indexe, padde, et renvoie `(textes, labels)`. |
| `nn.CrossEntropyLoss(ignore_index=pad_idx)` | | Ignorer les positions de padding quand on prédit **un label par token** (pas utile ici : un label par texte). |

```python
def collate_batch_imdb(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(1 if label == 1 else 0)                    # labels 0/1 (long)
        texts.append(torch.tensor([vocab[t] for t in tokenizer(text) if t in vocab]))
    return pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>']), torch.LongTensor(labels)

train_dl = DataLoader(train_list, batch_size=32, shuffle=True, collate_fn=collate_batch_imdb)
```

> 💡 🔎 Pour limiter le padding (et le temps de calcul), tronquer à une longueur max (`text[:500]`) ou regrouper les textes de longueurs proches (*bucketing*).

---

## 3. Modèle moyen (sac d'embeddings)

Représentation d'un texte = **moyenne** des embeddings de ses mots (hors `<pad>`), puis MLP.

$$\mathbf h = \frac{1}{\#\{t : x_t \neq \text{pad}\}}\sum_{t} E[x_t]$$

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Embedding(vocab_size, dim, padding_idx=pad)` | `padding_idx` | Le vecteur du `<pad>` est fixé à **0** et **jamais mis à jour**. |
| Masque | `mask = (x != pad).float()` | Compter les vrais mots par phrase. |
| `nn.EmbeddingBag(vocab_size, dim, mode='mean')` | 🔎 | Embedding + moyenne en une couche (avec `offsets`, sans padding). |

```python
class ModeleMoyen(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, layers, padding_idx, pre_trained=None):
        super().__init__()
        self.padding_idx = padding_idx
        if pre_trained is None:
            self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(pre_trained, freeze=True, padding_idx=padding_idx)
        couches, dim = [], input_dim
        for h in layers:                                  # ex. [50, 20]
            couches += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        couches.append(nn.Linear(dim, output_dim))        # output_dim = 2 avec CrossEntropy
        self.fc = nn.Sequential(*couches)

    def forward(self, x):                                 # x : (B, L)
        emb = self.embedding(x)                           # (B, L, d)
        n = (x != self.padding_idx).sum(1, keepdim=True).clamp(min=1)   # (B, 1)
        return self.fc(emb.sum(1) / n)                    # moyenne sur les vrais mots

modeleMoyen = ModeleMoyen(len(vocab), 100, 2, [50, 20], vocab['<pad>'])
```

> 💡 Pourquoi le pad « n'a pas beaucoup d'importance » ici : son vecteur est nul, il n'ajoute rien à la **somme** ; il ne fausse que le **dénominateur** si on divise par `L` au lieu du nombre de vrais mots.
> ⚠️ Classification binaire avec `CrossEntropyLoss` → **2 sorties** (sinon `BCEWithLogitsLoss` et 1 sortie).

---

## 4. Embeddings pré-entraînés & fine-tuning

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `torchtext.vocab.FastText('simple')` / `GloVe(name='6B', dim=100)` | | Vecteurs pré-entraînés (300d pour fastText simple) ; `fasttext[mot]` → vecteur (zéros si inconnu). |
| Matrice alignée sur **notre** vocabulaire | `torch.stack([fasttext[m] for m in vocab.get_itos()])` | Ligne $k$ = vecteur du mot d'indice $k$ : l'ordre doit correspondre à `vocab`. |
| `nn.Embedding.from_pretrained(W, freeze=True, padding_idx=pad)` | `freeze` | Gelé par défaut : seul le MLP apprend. |
| Fine-tuning | `model.embedding.weight.requires_grad = True` | Adapter les vecteurs à la tâche. ⚠️ `model.embedding.freeze = False` (vu dans le TP) **ne fait rien** : c'est l'attribut `requires_grad` qui compte. |

| Stratégie | Paramètres appris | Quand |
|---|---|---|
| End-to-end (aléatoire) | tout (≈ 1,3 M pour 13k mots × 100d) | beaucoup de données |
| Pré-entraîné **gelé** | le MLP seulement (quelques milliers) | peu de données, vocabulaire générique |
| Pré-entraîné **fine-tuné** | tout, à partir d'une bonne init | compromis, souvent le meilleur |

> 👉 Comparer les trois dans TensorBoard (loss/accuracy train et test) et réfléchir au **pourcentage de paramètres mis à jour**.

---

## 5. CNN 1D sur du texte

Idée : la convolution détecte des **motifs de quelques mots** (n-grammes : « un bon film », « très mauvais ») ; le pooling résume ; un **max global** sur la séquence donne un vecteur de taille fixe → classifieur linéaire.

| Module | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)` | `in_channels` = **dimension des embeddings** ; `kernel_size` = nb de mots couverts | Entrée `(B, C, L)` : les **canaux** sont les dimensions d'embedding, la **longueur** la séquence. |
| `nn.MaxPool1d(kernel_size, stride)` | | Résumé local. |
| Max global | `y.max(2)[0]` (ou `nn.AdaptiveMaxPool1d(1)`) | `(B, nfeat, L')` → `(B, nfeat)` : un score par filtre, quelle que soit la longueur. |

```python
class ModelCNN(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, nfeat, padding_idx, pre_trained=None):
        super().__init__()
        self.embedding = (nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx) if pre_trained is None
                          else nn.Embedding.from_pretrained(pre_trained, freeze=True, padding_idx=padding_idx))
        self.convolutions = nn.Sequential(
            nn.Conv1d(input_dim, 100, 3), nn.MaxPool1d(3, 1), nn.ReLU(),
            nn.Conv1d(100, 100, 4),       nn.MaxPool1d(4, 1), nn.ReLU(),
            nn.Conv1d(100, nfeat, 6),     nn.MaxPool1d(4, 1))
        self.fc = nn.Linear(nfeat, output_dim)
    def convolution_fn(self, x):
        return self.convolutions(self.embedding(x).transpose(1, 2))   # (B,L,d) → (B,d,L)
    def forward(self, x):
        return self.fc(self.convolution_fn(x).max(2)[0])              # max global sur la longueur
```

> ⚠️ **`transpose(1, 2)` obligatoire** : `nn.Embedding` renvoie `(B, L, d)` mais `Conv1d` attend `(B, d, L)`.
> ⚠️ Une séquence plus courte que la **portée** du réseau (19 mots ici) fait planter la convolution : padder à une longueur minimale ou ignorer ces batchs.
> 👉 Le CNN est **très lent sur CPU** (près de 3 h pour 10 epochs dans le TP) : GPU, ou réduire filtres / longueur max.

---

## 6. Portée (champ récepteur) et sous-séquences qui activent un filtre

Chaque opération $i$ (convolution ou pooling) a une largeur $w_i$ et un pas $s_i$. Récurrence de la **portée** $W$ (nombre de mots d'entrée vus par une sortie) et du **pas cumulé** $S$ :

$$W_{i+1} = W_i + (w_{i+1} - 1)\,S_i, \qquad S_{i+1} = S_i\, s_{i+1}, \qquad W_0 = S_0 = 1$$

La sortie $y_j$ correspond aux mots d'indices $[\,j\,S,\ j\,S + W\,[$ de l'entrée.

```python
class Scope:
    def __init__(self): self.width, self.stride = 1, 1
    def __call__(self, width, stride=1):
        self.width = (width - 1) * self.stride + self.width
        self.stride = self.stride * stride
        return self
# ModelCNN ci-dessus : conv3, pool3, conv4, pool4, conv6, pool4 (stride 1) → W = 19, S = 1
```

Retrouver les n-grammes qui activent le plus chaque filtre :

```python
with torch.no_grad():
    y = model.convolution_fn(batch_x)                       # (B, nfeat, L')
    B, nfeat, Lp = y.shape
    yt = y.transpose(0, 1).contiguous().view(nfeat, -1)     # (nfeat, B*L')
    vals, idx = yt.topk(K, 1)                               # K meilleures positions par filtre
    b, pos = idx // Lp, idx % Lp                            # batch et position
    segment = batch_x[b, pos*S : pos*S + W]                 # les W mots correspondants
    # " ".join(vocab.get_itos()[t] for t in segment)
```

> 💡 `heapq` (tas de taille K) garde efficacement le top-K sur tout le corpus.

---

## 7. Tester et inspecter le modèle

| Action | Code |
|---|---|
| Prédire une phrase | `x = torch.tensor([[vocab[t] for t in tokenizer("this movie is awful") if t in vocab]])` → `model(x).softmax(-1)` (forme `(1, L)` : **batch de 1**) |
| Trouver des erreurs | parcourir le train, comparer `model(x).argmax().item()` au label |
| Voir les embeddings | `writer.add_embedding(model.embedding.weight[:50], metadata=vocab.get_itos()[:50])` → onglet Projector |
| Mots proches | `F.cosine_similarity(E[vocab['great']].unsqueeze(0), E).topk(10)` 🔎 |

> 👉 Les embeddings appris **end-to-end** pour du sentiment ne capturent que la polarité (peu impressionnants en sémantique générale) ; les embeddings pré-entraînés (fastText, GloVe) ont une sémantique plus riche.

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| `pad_sequence` sans `batch_first=True` | tenseur `(L, B)`, formes fausses | `batch_first=True` |
| Pas d'index par défaut dans le vocab | `KeyError` / `RuntimeError` sur un mot inconnu | `set_default_index(vocab['<oov>'])` ou filtrer `if t in vocab` |
| Specials pas en tête | `<pad>` ≠ 0, embeddings du pad appris | `specials=['<pad>', '<oov>']` + `padding_idx` |
| Moyenne divisée par `L` (pads inclus) | représentation biaisée par la longueur | diviser par le nombre de vrais mots |
| `Conv1d` sur `(B, L, d)` | erreur de canaux | `.transpose(1, 2)` |
| Séquence trop courte pour le CNN | `Kernel size can't be greater than actual input size` | padder à la portée minimale |
| Labels IMDB 1/2 donnés tels quels | `CrossEntropyLoss` : cible hors bornes | ramener à 0/1 |
| 1 sortie avec `CrossEntropyLoss` | loss nulle / erreur | 2 sorties (ou BCEWithLogits + 1 sortie) |
| `embedding.freeze = False` | les embeddings restent gelés | `embedding.weight.requires_grad = True` |
| Matrice pré-entraînée pas alignée sur le vocab | embeddings attribués aux mauvais mots | construire `W` dans l'ordre de `vocab.get_itos()` |
| Phrase seule sans dimension batch | erreurs de forme | `torch.tensor([[...]])` |
| `torchtext` introuvable / incompatible | `ImportError` | figer les versions ou passer à Hugging Face |

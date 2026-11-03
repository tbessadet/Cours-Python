# 🧬 Cheat Sheet Apprentissage de représentations — Embeddings, recommandation, texte (moyenne & CNN 1D)

> Basée sur les notebooks `3_1-reco` (factorisation matricielle, MovieLens), `3_2-reco-triplet` (embeddings, *triplet loss*, jeux de société), `3_3-Repr` (texte IMDB : embeddings, pré-entraînés, CNN 1D).
> Conventions : `import torch`, `from torch import nn`, `import torch.nn.functional as F`, `from torch.utils.data import DataLoader`.
> Prérequis : `cs_pytorch_fondamentaux.md` (boucle d'entraînement, `DataLoader`, `eval`/`no_grad`). Séquences (RNN) : `cs_rnn_torch.md`.

## Sommaire
1. [`nn.Embedding` : la table de représentations](#1-nnembedding--la-table-de-représentations)
2. [Recommandation par factorisation matricielle](#2-recommandation-par-factorisation-matricielle)
3. [Batchs d'embeddings et `collate_fn`](#3-batchs-dembeddings-et-collate_fn)
4. [Perte avec régularisation et boucle avec *early stopping*](#4-perte-avec-régularisation-et-boucle-avec-early-stopping)
5. [Réseau de prédiction de notes (concaténation + MLP)](#5-réseau-de-prédiction-de-notes-concaténation--mlp)
6. [Triplet loss et apprentissage métrique](#6-triplet-loss-et-apprentissage-métrique)
7. [Visualiser des embeddings (t-SNE, TensorBoard)](#7-visualiser-des-embeddings-t-sne-tensorboard)
8. [Prétraitement du texte : tokenisation, vocabulaire, padding](#8-prétraitement-du-texte--tokenisation-vocabulaire-padding)
9. [Modèle « moyen » : embeddings moyennés + MLP](#9-modèle--moyen---embeddings-moyennés--mlp)
10. [Embeddings pré-entraînés et fine-tuning](#10-embeddings-pré-entraînés-et-fine-tuning)
11. [CNN 1D pour le texte et portée (*scope*)](#11-cnn-1d-pour-le-texte-et-portée-scope)
12. [📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. `nn.Embedding` : la table de représentations

Un embedding associe à chaque identifiant entier (utilisateur, item, token…) un **vecteur dense appris** de dimension $Z$. C'est un simple *lookup* dans une matrice $E\in\mathbb R^{V\times Z}$ (`V` = nb d'éléments).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None)` | `num_embeddings` = taille du vocabulaire/nb d'entités ; `embedding_dim` = $Z$ ; `padding_idx` = indice dont le vecteur reste à 0 et **n'est jamais mis à jour** ; `max_norm` = renormalise les vecteurs | Entrée : tenseur d'indices **`int64`** de forme quelconque `(…)` → sortie `(…, Z)`. Initialisation $\mathcal N(0,1)$ par défaut. |
| `nn.Embedding.from_pretrained(emb, freeze=True, padding_idx=None)` | `emb` = tenseur `(V, Z)` ; `freeze=True` → poids non appris | Charge des vecteurs existants (GloVe, FastText). **Gelé par défaut.** |
| `emb.weight` | — | La matrice `(V, Z)` (paramètre). `emb.weight.requires_grad = True` pour fine-tuner. |
| `nn.init.normal_(emb.weight, mean, std)` | `std=0.01` conseillé en factorisation | Initialisation en **petites valeurs** pour partir proche de zéro. |

```python
users = nn.Embedding(nb_users, 10)
nn.init.normal_(users.weight, 0, 0.01)
users(torch.tensor(5))            # -> (10,)     ATTENTION : un tenseur, pas un int Python
users(torch.tensor([5, 7]))       # -> (2, 10)
```

⚠️ L'entrée doit être un **tenseur d'entiers** (`users(5)` plante). `nn.Embedding` n'est **pas** `nn.Linear` : il ne multiplie pas, il indexe. Un one-hot suivi d'un `Linear` sans biais est mathématiquement équivalent, mais bien plus coûteux.

⚠️ Les indices doivent être **dans `[0, num_embeddings)`** : remapper les identifiants bruts en indices consécutifs (voir §2).

---

## 2. Recommandation par factorisation matricielle

Données : triplets $(u,i,r_{ui})$ (utilisateur, item, note). But : prédire $\hat r_{ui}$ pour une paire non observée.

**Préparer les données** (MovieLens-100k : `u.data`, séparateur tabulation) :

```python
ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["userId","movieId","rating","timestamp"])
user_map = {u: k for k, u in enumerate(ratings["userId"].unique())}   # ids bruts -> 0..N-1
item_map = {m: k for k, m in enumerate(ratings["movieId"].unique())}
ratings["userId"]  = ratings["userId"].map(user_map)
ratings["movieId"] = ratings["movieId"].map(item_map)
```

Découpage du TD : 1 ligne sur 5 en **test** (20 %), puis 20 % du reste en **validation** (après `shuffle`) → 64 000 / 16 000 / 20 000. Pour parcourir : `for idx, uid, mid, r, ts in df.itertuples()`.

### Modèles (équations)

| Modèle | Prédiction | Paramètres |
|---|---|---|
| **Baseline** (biais seuls) | $\hat r_{ui}=\mu+b_u+b_i$ | $\mu\in\mathbb R$, $b_u\in\mathbb R^{n_{users}}$, $b_i\in\mathbb R^{n_{items}}$ |
| **MF classique** (« SVD » RecSys) | $\hat r_{ui}=\mu+U_u^\top I_i$ | $\mu$, $U\in\mathbb R^{n_u\times Z}$, $I\in\mathbb R^{n_i\times Z}$ |
| **MF de Koren (2009)** | $\hat r_{ui}=\mu+b_u+b_i+U_u^\top I_i$ | $\mu$, $b_u$, $b_i$, $U$, $I$ |

Objectif minimisé (régularisation L2 sur les profils) :
$$\min_{U,I,b}\sum_{(u,i)}\big(r_{ui}-\hat r_{ui}\big)^2+\lambda\big(\lVert U_u\rVert^2+\lVert I_i\rVert^2+\dots\big)$$

- $r_{ui}$ : note observée ; $\hat r_{ui}$ : note prédite ; $\lambda$ : poids de régularisation (`reg`, ex. $10^{-3}$) ; $Z$ : taille latente (`num_feat`, ex. 25).

**Modèle PyTorch** :

```python
class ClassicMF(nn.Module):
    def __init__(self, n_users, n_items, latent):
        super().__init__()
        self.users = nn.Embedding(n_users, latent)
        self.items = nn.Embedding(n_items, latent)
        nn.init.normal_(self.users.weight, 0, 0.01); nn.init.normal_(self.items.weight, 0, 0.01)
        self.mean = nn.Parameter(torch.FloatTensor(1,).fill_(3.0))      # biais global appris
    def forward(self, user, item):                  # user, item : (B,) int64
        eu, ei = self.users(user), self.items(item)             # (B, Z)
        out = (eu * ei).sum(1) + self.mean                      # produit terme à terme puis somme sur Z -> (B,)
        return out, eu, ei, self.mean                           # on renvoie aussi les poids à régulariser

# Koren : ajouter self.bu = nn.Embedding(n_users, 1), self.bi = nn.Embedding(n_items, 1)
#         out = (eu*ei).sum(1) + self.mean + self.bu(user).squeeze(1) + self.bi(item).squeeze(1)
```

| Astuce | Explication |
|---|---|
| `nn.Parameter(tensor)` | Déclare un tenseur comme **paramètre** du module (visible par `.parameters()`, optimisé). |
| `(eu * ei).sum(1)` | Produit scalaire **par ligne** pour tout un batch, en évitant une boucle (`torch.bmm`/`einsum` possibles). |
| `bu(user).squeeze(1)` | `Embedding(n,1)` renvoie `(B,1)` : `squeeze(1)` → `(B,)` pour coller à `out`. |

💡 La baseline avec un `mu` appris, un biais par user et par item : sans réseau, seulement 3 scalaires par prédiction, et c'est déjà un **baseline très solide** (MSE ≈ 0,9 sur MovieLens-100k).

⚠️ **Utilisateurs/items inconnus** (absents du train) : la fonction `predict` doit gérer l'indice hors table (biais = 0) au lieu de planter.

---

## 3. Batchs d'embeddings et `collate_fn`

`DataLoader` peut itérer directement sur une **liste Python de tuples** `(user, item, rating)` ; le `collate_fn` transforme la liste d'exemples d'un batch en tenseurs.

```python
def tuple_batch(l):
    users, items, ratings = zip(*l)                     # transposition : liste de tuples -> 3 tuples
    return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(ratings)

prep_train = [(t.userId, t.movieId, t.rating) for t in train_ratings.itertuples()]
dl = DataLoader(prep_train, batch_size=16, shuffle=True, num_workers=0, collate_fn=tuple_batch)
```

- Chaîne complète : `Dataset (liste) ──DataLoader──▶ batch brut ──collate_fn──▶ tenseurs ──model.forward──▶ prédiction ──loss──▶ coût`.
- 🔎 `torch.tensor(users, dtype=torch.long)` est l'écriture moderne de `torch.LongTensor(users)`.

**Syntaxe `*args`** (utilisée pour passer un nombre variable de poids à régulariser) :
```python
pred, *params = model(users_t, items_t)      # params = liste de tous les autres retours
loss_func(pred, ratings_t, reg, *params)     # *params dépaquette la liste en arguments positionnels
```
Avantage : la même fonction de coût marche pour `ClassicMF` (3 retours en plus) et `KorenMF` (plus de retours).

---

## 4. Perte avec régularisation et boucle avec *early stopping*

```python
def loss_func(pred, ratings_t, reg, *params):
    mse = F.mse_loss(pred, ratings_t, reduction='sum')
    l2 = sum(torch.mean(p.norm(2, -1)) for p in params)        # norme L2 des vecteurs, moyennée
    return mse / pred.size(0) + reg * l2, mse                  # (coût optimisé, MSE brute pour l'évaluation)
```

| Élément | Explication |
|---|---|
| `F.mse_loss(pred, target, reduction)` | `reduction='sum'` renvoie la somme (pour recalculer une MSE moyenne propre sur tout le jeu en divisant par le nb d'exemples). |
| `p.norm(2, -1)` | Norme L2 le long du dernier axe (`p` de forme `(B,Z)` → `(B,)`). |
| Évaluation | Calculer la MSE (sans le terme de régularisation) sur **train / val / test** à chaque epoch pour comparer ; seule la validation sert aux choix d'hyperparamètres, le test n'est lu qu'à la fin. |

**Boucle générique train/test** (notebook `3_2`) : on alterne les phases dans une même boucle avec `torch.set_grad_enabled(phase=='train')`, `model.train()/eval()`, et on conserve le meilleur modèle.

```python
best_l, best_model, patience = 1e5, None, 0
for epoch in range(nb_epochs):
    for phase in ['test', 'train']:
        for games, users, notes in dataloaders[phase]:
            with torch.set_grad_enabled(phase == 'train'):
                model.train() if phase == 'train' else model.eval()
                yhat = model(games.to(device), users.to(device))
                l = loss(yhat.view(-1), notes.float().to(device).view(-1))
                if phase == 'train':
                    opt.zero_grad(); l.backward(); opt.step()
        if phase == 'test':
            if cum_loss < best_l:
                best_l, patience = cum_loss, 0
                best_model = copy.deepcopy(model.state_dict())      # copie profonde du meilleur état
            else:
                patience += 1
    if patience > early_stop: break
model.load_state_dict(best_model)
```

- `copy.deepcopy(model.state_dict())` : **copie** les poids (un simple `state_dict()` renvoie des références qui changent au fil de l'apprentissage).
- *Early stopping* : on arrête après `early_stop` epochs sans amélioration de la loss de test/validation.

⚠️ Le TD utilise ici le jeu de **test** comme critère d'arrêt : c'est une fuite (le meilleur modèle est choisi sur le test). En pratique, utiliser un jeu de **validation** distinct et ne lire le test qu'une fois.

⚠️ Nom de variable : initialiser le compteur (`early_cpt`/`patience`) **avant** la boucle, sinon `NameError` si le premier test n'améliore pas.

---

## 5. Réseau de prédiction de notes (concaténation + MLP)

Alternative non linéaire à la factorisation : deux embeddings (utilisateur, jeu) **concaténés** puis passés dans un MLP, sortie = note.

```python
class RecoRegNet(nn.Module):
    def __init__(self, n_games, n_users, latent_dim=50, act=nn.Tanh):
        super().__init__()
        self.games = nn.Embedding(n_games, latent_dim)
        self.users = nn.Embedding(n_users, latent_dim)
        self.mlp = nn.Sequential(nn.Linear(2*latent_dim, latent_dim), act(), nn.Linear(latent_dim, 1))
    def forward(self, igames, iusers):
        x = torch.cat([self.games(igames), self.users(iusers)], dim=1)   # (B, 2*latent)
        return self.mlp(x)                                               # (B, 1) -> .view(-1) avant la loss
```

Prétraitement spécifique du TD :

| Étape | Raison |
|---|---|
| **Filtrage itératif** (`filtre(data, ths_g, ths_u)`) : retirer jeux/users ayant < seuil de notes, **en boucle jusqu'à convergence** | Retirer des items change les comptes par utilisateur, et inversement ; on itère jusqu'à stabilité. Réduit la *sparsity* (rareté de la matrice). |
| **Normalisation des notes** dans $[0,1]$ : $(n-n_{min})/(n_{max}-n_{min})$ | Cible bornée, compatible avec une sortie sigmoïde / un `lr` raisonnable. |
| **Split stratifié par utilisateur** (80/20 dans les notes de **chaque** user) | Chaque utilisateur apparaît dans le train ET le test (sinon profils inconnus). |

💡 Comparer toujours à la **baseline** (moyenne) : si le réseau ne fait pas mieux, c'est que la factorisation/embedding n'apporte rien.

---

## 6. Triplet loss et apprentissage métrique

**But** : structurer l'espace des embeddings pour que des éléments « similaires » soient proches et des éléments « dissemblables » éloignés.

Pour un triplet (ancre $a$, positif $p$ proche de $a$, négatif $n$ éloigné) :
$$\mathcal L(a,p,n)=\max\big(0,\;d(a,p)-d(a,n)+m\big)$$

- $d$ : distance (euclidienne $\lVert\cdot\rVert_2$ par défaut) ; $m$ : *margin* (marge, 1.0 par défaut) ; on veut que le négatif soit plus loin que le positif d'au moins $m$. Si c'est déjà le cas, perte nulle (aucun gradient).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `nn.TripletMarginLoss(margin=1.0, p=2, swap=False, reduction='mean')` | `margin`, `p` = norme (2 = euclidienne) | Appel : `loss(anchor, positive, negative)` avec trois tenseurs `(B, D)`. |
| `F.triplet_margin_loss(a, p, n, …)` | idem | Version fonctionnelle. |
| `nn.CosineEmbeddingLoss`, `nn.MarginRankingLoss` | — | 🔎 Variantes pour un cosinus ou un classement par paires. |

**Exercice du TD** : des points 2D colorés (RGB aléatoire) sont les **paramètres** (`X.requires_grad=True`, pas de réseau). Un `Dataset` renvoie `(ancre, positif de même couleur dominante, négatif aléatoire)`. On optimise `X` lui-même avec la triplet loss : les points de même couleur se regroupent.

```python
X = torch.rand(N, 2, requires_grad=True)                 # les "embeddings" sont des paramètres libres
loss_fn = nn.TripletMarginLoss()
opt = torch.optim.SGD([X], lr=3e-1)                      # optimiser X directement
for epoch in range(200):
    for xr, xp, xn in dload:                             # (B,2) chacun
        l = loss_fn(xr, xp, xn)
        opt.zero_grad(); l.backward(); opt.step()
```

⚠️ Dans le `Dataset` du TD les triplets sont obtenus avec `data[ind]` sur un tenseur à gradient : l'**indexation conserve le lien** avec `X`, donc l'optimisation agit sur la matrice entière. Si on copie les coordonnées (`.detach()`/`.clone()` hors graphe), aucun point ne bouge.

⚠️ Le **négatif tiré au hasard** peut tomber dans la même classe que l'ancre (faux négatif) → bruit. En pratique : *hard negative mining* (choisir des négatifs proches mais de classe différente).

💡 Une triplet loss avec un `lr` élevé (0.3 dans le TD) pour des paramètres libres en 2D : les échelles ne sont pas celles d'un réseau.

---

## 7. Visualiser des embeddings (t-SNE, TensorBoard)

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `writer.add_embedding(mat, metadata=None, global_step=None)` | `mat` = `(V, Z)` ; `metadata` = liste d'étiquettes (une par ligne) | Envoie la matrice à TensorBoard → onglet **Projector** (t-SNE, PCA, UMAP interactifs). |
| `sklearn.manifold.TSNE(n_components=2, perplexity=30, random_state=0)` | `perplexity` (5–50) | Projection 2D non linéaire ; `fit_transform(emb.detach().cpu().numpy())`. |

```python
writer.add_embedding(model.games.weight, metadata=[dic_jeux[idx2games[i]] for i in range(len(idx2games))])
writer.flush()
```
⚠️ Les distances absolues d'un t-SNE n'ont pas de sens (seuls les voisinages comptent) ; varier `perplexity`/graine avant de conclure. `weight` a un gradient → `.detach()` pour numpy.

---

## 8. Prétraitement du texte : tokenisation, vocabulaire, padding

Pipeline : **texte → tokens → indices (`int64`) → padding → batch `(B, L)`**.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `get_tokenizer("basic_english")` (`torchtext.data.utils`) | nom du tokenizer | Découpe en mots minuscules en séparant la ponctuation. |
| `torchtext.vocab.build_vocab_from_iterator(it, specials=['<pad>','<oov>'], min_freq=…)` | `specials` = tokens spéciaux (ajoutés **en premier** → `<pad>` = indice 0) ; `min_freq` = fréquence minimale pour entrer dans le vocabulaire | Construit le vocabulaire. `vocab[token]` → indice ; `vocab.get_itos()` → liste indice→mot. `vocab.set_default_index(vocab['<oov>'])` pour qu'un mot inconnu renvoie `<oov>` au lieu d'une erreur. |
| `pad_sequence(list_of_tensors, batch_first=False, padding_value=0)` (`torch.nn.utils.rnn`) | `batch_first=True` → `(B, L)` ; `padding_value` = **indice du `<pad>`** | Complète à droite les séquences à la longueur max du batch. Par défaut sortie `(L, B)` (convention RNN). |
| `DataLoader(…, collate_fn=collate_batch)` | — | Le *collate* tokenise, indexe, pad et renvoie `(textes, labels)`. |

```python
def collate_batch_imdb(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(1 if label == 1 else 0)
        texts.append(torch.tensor([vocab[t] for t in tokenizer(text) if t in vocab]))   # mots hors vocabulaire retirés
    return pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>']), torch.LongTensor(labels)
```

💡 **Le padding** rend les phrases de même longueur pour former un tenseur. Convention : indice 0 + vecteur nul. Comme il est à la fin, il ne gêne pas l'inférence ; il faut seulement l'**ignorer** dans le calcul de la représentation (moyenne) ou de la perte (`ignore_index`, voir `cs_rnn_torch.md`).

⚠️ `pad_sequence` sans `padding_value` remplit avec **0** : cohérent seulement si `<pad>` a bien l'indice 0.

⚠️ Dans le texte, retirer les mots inconnus (comme ci-dessus) peut produire une séquence **vide** → `pad_sequence`/le modèle plantent ; sinon les remplacer par `<oov>`.

---

## 9. Modèle « moyen » : embeddings moyennés + MLP

Une phrase est représentée par la **moyenne des embeddings de ses tokens** (sac de mots dense) → MLP → classe. Ignore l'ordre des mots.

```python
class ModeleMoyen(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, layers, padding_idx, pre_trained=None):
        super().__init__()
        self.padding_idx = padding_idx
        if pre_trained is None:
            self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(pre_trained, freeze=True, padding_idx=padding_idx)
        dims, mods, d = layers, [], input_dim
        for h in dims: mods += [nn.Linear(d, h), nn.ReLU()]; d = h
        mods.append(nn.Linear(d, output_dim))
        self.fc = nn.Sequential(*mods)
    def forward(self, x):                                   # x : (B, L) indices
        e = self.embedding(x)                               # (B, L, Z)
        mask = (x != self.padding_idx).unsqueeze(-1)        # (B, L, 1), False sur le padding
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1) # somme / nb de VRAIS mots
        return self.fc(mean)                                # (B, output_dim) -> CrossEntropyLoss
```

- `output_dim` = **2** pour un problème binaire avec `CrossEntropyLoss`.
- Le TD propose : embeddings 100-D, couches 50 puis 20 neurones + ReLU, 10 epochs, Adam `1e-3` ; les courbes d'apprentissage et de validation sont très bonnes.

💡 Pourquoi le padding n'a « pas beaucoup d'importance » ici : avec `padding_idx`, son vecteur vaut 0 donc il ne change pas la **somme** ; il fausse seulement le **diviseur** si on divise par `L` au lieu du vrai nombre de mots.

Lecture des résultats : `model(input).argmax()` pour une phrase (`input` = `torch.tensor([[idx…]])`, batch de 1) ; parcourir le train pour trouver des erreurs, regarder `add_embedding(model.embedding.weight[:50], metadata=vocab.get_itos()[:50])` pour inspecter les mots appris.

---

## 10. Embeddings pré-entraînés et fine-tuning

| Élément | Explication |
|---|---|
| `torchtext.vocab.FastText('simple')` / `GloVe(name='6B', dim=50)` | Vecteurs pré-appris sur un grand corpus (300-D pour FastText simple). `ft[mot]` → vecteur. |
| `fast_vecs = torch.stack([ft[w] for w in vocab.get_itos()])` | Matrice `(V, 300)` alignée sur **l'ordre de votre vocabulaire**. |
| `nn.Embedding.from_pretrained(fast_vecs, freeze=True)` | Embeddings **figés** (rien n'est appris dans cette couche). |
| `emb.weight.requires_grad = True` | **Fine-tuning** : on les adapte à la tâche (apprend davantage mais risque de sur-apprentissage). |

Comparaisons du TD (même MLP) : *appris de zéro* vs *pré-entraînés gelés* vs *pré-entraînés + fine-tuning*. Réfléchir : **quel pourcentage des paramètres est mis à jour** quand l'embedding est gelé ? (seulement le MLP : très peu).

```python
m = ModeleMoyen(len(vocab), 300, 2, [50, 20], vocab["<pad>"], fast_vecs)
m.embedding.weight.requires_grad = True        # fine-tuning (sinon gelé)
```

⚠️ **Piège du TD** : écrire `modelCNN_fine.embedding.freeze = False` **ne fait rien** (attribut Python inventé). La bonne instruction est `embedding.weight.requires_grad = True` (ou `from_pretrained(..., freeze=False)`).

⚠️ Les vecteurs pré-entraînés doivent être **alignés** sur les indices du vocabulaire (même ordre). Mots absents de GloVe/FastText → vecteur nul ou aléatoire à gérer.

👉 Réflexe : commencer gelé (rapide, robuste), dégeler si assez de données et si la tâche s'éloigne du corpus d'origine.

---

## 11. CNN 1D pour le texte et portée (*scope*)

Les convolutions 1D recherchent des **motifs de n-grammes** (ex. « un bon film ») ; le pooling résume ; le **max global** extrait, par filtre, la meilleure activation sur toute la phrase.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)` | `in_channels` = dim d'embedding ; `out_channels` = nb de filtres ; `kernel_size` = nb de tokens couverts | Entrée **`(B, C, L)`** : les canaux sont les dimensions d'embedding. |
| `nn.MaxPool1d(kernel_size, stride)` | idem | Agrège `kernel_size` positions (stride 1 : la longueur diminue de `k-1`). |
| `x.max(2)[0]` / `F.adaptive_max_pool1d(x, 1)` | axe temporel | **Max global** : un nombre par filtre → vecteur de taille fixe indépendante de la longueur du texte. |

```python
class ModelCNN(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, nfeat, padding_idx, pre_trained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=padding_idx)
        self.convolutions = nn.Sequential(
            nn.Conv1d(input_dim, 100, 3), nn.MaxPool1d(3, 1), nn.ReLU(),
            nn.Conv1d(100, 100, 4),       nn.MaxPool1d(4, 1), nn.ReLU(),
            nn.Conv1d(100, nfeat, 6),     nn.MaxPool1d(4, 1))
        self.fc = nn.Linear(nfeat, output_dim)
    def convolution_fn(self, x):
        return self.convolutions(self.embedding(x).transpose(1, 2))   # (B,L,Z) -> (B,Z,L)
    def forward(self, x):
        y = self.convolution_fn(x)                  # (B, nfeat, L')
        return self.fc(y.max(2)[0])                 # max global sur le temps -> (B, nfeat) -> (B, classes)
```

⚠️ **`transpose(1, 2)`** indispensable : `Embedding` sort `(B, L, Z)` mais `Conv1d` veut `(B, Z, L)`.

⚠️ La phrase doit être **au moins aussi longue que la portée** du réseau (19 tokens ici), sinon les dimensions s'effondrent : le TD saute les batchs trop courts (`if batch[0].shape[1] < scope.width: continue`).

### Portée (*receptive field*) d'une pile de convolutions/poolings

On note $w_i$ la taille du noyau et $s_i$ le stride de l'opération $i$ ; $W_i$ = nombre de tokens d'entrée vus par une sortie après l'opération $i$ ; $S_i$ = décalage dans l'entrée correspondant à un décalage unitaire en sortie.

$$W_{i+1}=W_i+(w_{i+1}-1)\,S_i\qquad\qquad S_{i+1}=S_i\cdot s_{i+1}\qquad(W_0=1,\ S_0=1)$$

La sortie $y_j$ correspond aux tokens d'entrée d'indices $[\,j\cdot S,\; j\cdot S+W-1\,]$ (avec $W,S$ de la dernière opération).

*Exemple* : Conv(3) + Pool(3,1) + Conv(4) + Pool(4,1) + Conv(6) + Pool(4,1), tous de stride 1 → $W=1+2+2+3+3+5+3=19$, $S=1$ (portée 19 : « Scope (w=19, s=1) »).

**Analyser ce que détecte chaque filtre** (méthode du TD) : sous `no_grad`, pour chaque batch du train, calculer les activations `y` `(B, nfeat, L')`, garder pour chaque filtre les **K plus fortes** (`topk`, tas `heapq`), retrouver le batch/la position (`ix // L'`, `ix % L'`) puis le texte `x[b, pos*S : pos*S+W]` et le décoder avec `vocab.get_itos()`. On obtient, par filtre, les sous-séquences qui l'activent le plus (ex. des expressions positives ou négatives).

---

## 📌 Récapitulatif des pièges classiques

| # | Piège | Réflexe |
|---|---|---|
| 1 | `Embedding(5)` avec un `int` Python / indices `float` | `torch.tensor(5)` ou liste d'`int64` (`LongTensor`). |
| 2 | Ids bruts (utilisateurs/films) non consécutifs | Remapper en `0..N-1` avec un dictionnaire avant `Embedding`. |
| 3 | Utilisateur/item absent du train à l'inférence | Gérer le cas inconnu (biais = 0, vecteur moyen). |
| 4 | Forme `(B,1)` vs `(B,)` entre prédiction et cible dans la MSE | `.view(-1)` / `squeeze(1)` des deux côtés (broadcasting silencieux). |
| 5 | Régularisation L2 oubliée dans la perte, ou mélangée dans la MSE reportée | Retourner séparément la MSE brute et la perte régularisée. |
| 6 | Early stopping sur le test | Utiliser une validation séparée ; `deepcopy(state_dict)` pour garder le meilleur. |
| 7 | `pad_sequence` avec `padding_value` ≠ indice de `<pad>` | `padding_value=vocab['<pad>']`. |
| 8 | Moyenne des embeddings qui compte le padding | Diviser par le nombre de vrais mots (masque). |
| 9 | `CrossEntropyLoss` avec `output_dim=1` | 2 sorties pour le binaire. |
| 10 | `embedding.freeze = False` (sans effet) | `embedding.weight.requires_grad = True`. |
| 11 | Vecteurs pré-entraînés non alignés sur l'ordre du vocabulaire | `torch.stack([ft[w] for w in vocab.get_itos()])`. |
| 12 | `Conv1d` alimenté en `(B, L, Z)` | `transpose(1, 2)` pour obtenir `(B, Z, L)`. |
| 13 | Textes plus courts que la portée du CNN / séquences vides | Padding à une longueur minimale ; remplacer l'inconnu par `<oov>`. |
| 14 | Triplets obtenus par copie (hors graphe) | Indexer le tenseur-paramètre pour garder le gradient ; négatifs difficiles pour de meilleurs résultats. |
| 15 | Lire des distances sur un t-SNE | Seuls les voisinages sont fiables ; varier `perplexity`. |

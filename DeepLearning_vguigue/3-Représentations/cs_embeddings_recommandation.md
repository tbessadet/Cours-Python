# 🎯 Cheat Sheet PyTorch — Embeddings, recommandation & triplet loss

> Basée sur les notebooks `3_0-reco`, `3_1-reco`, `3_1-triplet-loss`, `3_2-reco-triplet`.
> Convention : `import torch`, `from torch import nn`, `import torch.nn.functional as F`, `from torch.utils.data import DataLoader`, `import pandas as pd`

## Sommaire

1. [Préparer des données d'interactions](#1-préparer-des-données-dinteractions)
2. [`nn.Embedding` : la table de représentations](#2-nnembedding--la-table-de-représentations)
3. [Modèle de base à 3 scalaires (autograd « à la main »)](#3-modèle-de-base-à-3-scalaires-autograd--à-la-main-)
4. [Factorisation matricielle : ClassicMF et Koren](#4-factorisation-matricielle--classicmf-et-koren)
5. [DataLoader sur des listes + `collate_fn`](#5-dataloader-sur-des-listes--collate_fn)
6. [Loss MSE + régularisation, boucle complète](#6-loss-mse--régularisation-boucle-complète)
7. [Recommandation neuronale (embeddings + MLP) & early stopping](#7-recommandation-neuronale-embeddings--mlp--early-stopping)
8. [Triplet loss](#8-triplet-loss)
9. [Visualiser des embeddings](#9-visualiser-des-embeddings)
10. [Syntaxe Python utile (`*args`, `zip(*…)`)](#10-syntaxe-python-utile-args-zip)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Préparer des données d'interactions

| Étape | Code | Explication |
|---|---|---|
| Charger MovieLens 100k | `pd.read_csv("data/ml-100k/u.data", sep="\t", names=["userId","movieId","rating","timestamp"])` | Triplets (user, item, note) + timestamp. |
| **Ré-indexer** de 0 à N−1 | `user_map = {u: k for k, u in enumerate(ratings.userId.unique())}` puis `ratings.userId.map(user_map)` | `nn.Embedding` exige des indices **contigus** dans `[0, n)`. Garder les dictionnaires pour retrouver les vrais ids (et ré-indexer les *side informations* avec **le même** dictionnaire). |
| Split train/val/test | `index % 5 == 0` → test ; `shuffle` puis 20 % → val | Ou un split **stratifié par utilisateur** (chaque user a des notes en train et en test) comme `prepare_dataloaders`. |
| Parcourir | `for index, uid, mid, r, ts in df.itertuples():` | Itération rapide sur les lignes d'un DataFrame. |
| Filtrer les données trop creuses | boucle itérative : garder les items avec > `ths_g` notes et users avec > `ths_u` avis **jusqu'à convergence** | Enlever un user change le compte des items, et inversement. |
| Normaliser les notes | `(n - note_min) / (note_max - note_min)` | Cible dans [0, 1] (penser à dé-normaliser pour interpréter). |

Densité (*sparsity*) : $\dfrac{\#\text{notes}}{n_{users}\times n_{items}}$ (6 % sur MovieLens 100k, souvent < 1 % en pratique).

---

## 2. `nn.Embedding` : la table de représentations

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None)` | nb d'objets, dimension latente $z$ (10–100) | Matrice `.weight` de forme `(n, z)`, **apprise**. Appel avec un **tenseur d'indices `long`** de forme quelconque `(*)` → `(*, z)`. |
| `torch.nn.init.normal_(emb.weight, 0, 0.01)` | moyenne, écart-type | Initialiser **petit** (le défaut est $\mathcal N(0,1)$, trop grand pour un produit scalaire de notes). |
| `nn.Embedding.from_pretrained(W, freeze=True)` | matrice `(n, z)` | Charger des vecteurs pré-appris (gelés par défaut). |
| `nn.Parameter(torch.FloatTensor(1).fill_(3))` | | Un scalaire appris (biais global $\mu$) enregistré dans le module. |

```python
users = nn.Embedding(nb_users, 10)
users(torch.tensor(5)).shape        # (10,)
users(torch.tensor([5, 7])).shape   # (2, 10)   batch d'indices
# users(5)                          # ❌ TypeError : int Python
```

**Produit scalaire par batch** (le point délicat) :

```python
u, i = users(ind_u), items(ind_i)   # (B, z) chacun
r_hat = (u * i).sum(1)              # (B,) : produit terme à terme puis somme sur z
# ❌ u @ i.T donnerait (B, B) : toutes les paires croisées (coûteux et faux)
```

---

## 3. Modèle de base à 3 scalaires (autograd « à la main »)

$$\hat r_{ui} = \mu + b_u + b_i$$

```python
mu = torch.tensor([3.5], requires_grad=True)
bu = [torch.tensor([0.1], requires_grad=True) for _ in range(num_users)]
bi = [torch.tensor([0.1], requires_grad=True) for _ in range(num_items)]

def predict(u, i):
    return mu + (bu[u] if u < num_users else 0) + (bi[i] if i < num_items else 0)  # user/item inconnu → 0

lr = 0.01
for epoch in range(5):
    for _, uid, mid, r, _ in train_ratings.itertuples():   # SGD exemple par exemple
        err = (predict(uid, mid) - r) ** 2
        err.backward()
        with torch.no_grad():
            for p in (mu, bu[uid], bi[mid]):               # seuls 3 paramètres concernés
                p -= lr * p.grad
                p.grad.zero_()
```

> 💡 Baseline **forte** (celle de la librairie *surprise*) : à battre par tout modèle plus complexe. Gérer les **utilisateurs/items inconnus** (cold start) en retombant sur $\mu$.

---

## 4. Factorisation matricielle : ClassicMF et Koren

| Modèle | Prédiction | Paramètres |
|---|---|---|
| ClassicMF (« SVD » en RecSys) | $\hat r_{ui} = \mu + U_u\cdot I_i$ | $\mu$, $U$ `(n_U, z)`, $I$ `(n_I, z)` |
| **Koren 2009** | $\hat r_{ui} = \mu + \mu_u + \mu_i + U_u\cdot I_i$ | + biais par user et par item |

Objectif (avec régularisation L2) :

$$\min_{U, I}\sum_{(u,i)\ \text{observés}} \big(r_{ui} - \hat r_{ui}\big)^2 + \lambda\big(\lVert U_u\rVert^2 + \lVert I_i\rVert^2\big)$$

```python
class ClassicMF(nn.Module):
    def __init__(self, nb_users, nb_items, latent_size):
        super().__init__()
        self.mean  = nn.Parameter(torch.FloatTensor(1).fill_(3))
        self.users = nn.Embedding(nb_users, latent_size)
        self.items = nn.Embedding(nb_items, latent_size)
        nn.init.normal_(self.users.weight, 0, 0.01)
        nn.init.normal_(self.items.weight, 0, 0.01)
    def forward(self, user, item):                         # user, item : (B,) long
        embed_u, embed_i = self.users(user), self.items(item)
        out = (embed_u * embed_i).sum(1) + self.mean
        return out, embed_u, embed_i, self.mean            # prédiction + poids à régulariser

class KorenMF(ClassicMF):
    def __init__(self, nb_users, nb_items, latent_size):
        super().__init__(nb_users, nb_items, latent_size)
        self.bu = nn.Embedding(nb_users, 1); self.bi = nn.Embedding(nb_items, 1)
        nn.init.normal_(self.bu.weight, 0, 0.01); nn.init.normal_(self.bi.weight, 0, 0.01)
    def forward(self, user, item):
        out, eu, ei, mu = super().forward(user, item)
        bu, bi = self.bu(user).squeeze(1), self.bi(item).squeeze(1)   # (B,1) → (B,)
        return out + bu + bi, eu, ei, bu, bi
```

> 👉 Hyper-paramètres du TP : `num_feat=25`, `batch_size=16`, `reg=0.001`, Adam (lr 1e-3). Augmenter $z$ sans régulariser → sur-apprentissage (erreur de validation qui remonte).

---

## 5. DataLoader sur des listes + `collate_fn`

| Outil | Explication |
|---|---|
| Dataset = **liste** de tuples | Une liste a `__getitem__` et `__len__` : elle suffit (*duck typing*). `prep_train = [(t.userId, t.movieId, t.rating) for t in df.itertuples()]` |
| Classe Dataset minimale | `__init__(data)`, `__getitem__(i)` → `(game, user, note)`, `__len__()`. |
| `collate_fn(batch)` | Reçoit la **liste** des `batch_size` exemples et renvoie le batch **en tenseurs**. Indispensable dès que les éléments ne sont pas déjà des tenseurs (ou pour du padding). |

```python
def tuple_batch(l):
    users, items, ratings = zip(*l)                  # liste de tuples → 3 tuples
    return torch.LongTensor(users), torch.LongTensor(items), torch.FloatTensor(ratings)

dl_train = DataLoader(prep_train, batch_size=16, shuffle=True, collate_fn=tuple_batch)
dl_test  = DataLoader(prep_test,  batch_size=16, shuffle=False, collate_fn=tuple_batch)
```

> 💡 Chaîne : `Dataset ==DataLoader==> batch brut ==collate_fn==> batch tenseurs ==model==> prédiction ==loss==> erreur`.

---

## 6. Loss MSE + régularisation, boucle complète

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `F.mse_loss(pred, target, reduction='mean')` | `'sum'` pour sommer | Version fonctionnelle de `nn.MSELoss`. |
| `p.norm(2, -1)` | ordre, dimension | Norme L2 de chaque vecteur (ligne) d'un batch d'embeddings → `(B,)`. |

```python
def loss_func(pred, ratings_t, reg, *params):        # *params : nombre variable de tenseurs
    mse = F.mse_loss(pred, ratings_t, reduction='sum')
    l2 = sum(torch.mean(p.norm(2, -1)) for p in params)
    return mse / pred.size(0) + reg * l2, mse

model = KorenMF(num_users, num_items, 25)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    model.train()
    for users_t, items_t, ratings_t in dl_train:
        pred, *params = model(users_t, items_t)      # récupère tous les poids renvoyés
        loss, _ = loss_func(pred, ratings_t, 0.001, *params)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        mse = sum(loss_func(*[model(u, i)[0], r, 0])[1].item() for u, i, r in dl_val) / len(prep_val)
    print(epoch, "val MSE", mse, "RMSE", mse ** 0.5)
```

> 💡 La régularisation ne porte que sur les embeddings **du batch** (pas toute la table) : c'est l'usage en SGD.
> 👉 Comparer en **RMSE** sur le test : baseline 3 scalaires ≈ 0,94, MF/Koren un peu mieux sur MovieLens 100k.

---

## 7. Recommandation neuronale (embeddings + MLP) & early stopping

Architecture : embedding user `(ℓ)` + embedding jeu `(ℓ)` → **concaténation** `(2ℓ)` → MLP `2ℓ → ℓ → 1`.

```python
class RecoRegNet(nn.Module):
    def __init__(self, n_games, n_users, latent_dim=50, act=nn.Tanh):
        super().__init__()
        self.games = nn.Embedding(n_games, latent_dim)
        self.users = nn.Embedding(n_users, latent_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), act(),
                                 nn.Linear(latent_dim, 1))
    def forward(self, igames, iusers):
        x = torch.cat([self.games(igames), self.users(iusers)], dim=-1)   # (B, 2ℓ)
        return self.mlp(x).squeeze(-1)                                    # (B,)
```

| Outil | Explication |
|---|---|
| `torch.set_grad_enabled(phase == 'train')` | Un seul code pour train et test : gradient actif seulement en train (équivaut à `no_grad` en test). |
| `copy.deepcopy(model.state_dict())` | Sauvegarder le **meilleur** modèle (sinon on garde une référence qui continue d'évoluer). |
| **Early stopping** | Compter les époques sans amélioration de la loss de test/validation ; arrêter après `early_stop` (5) ; recharger le meilleur `state_dict`. |
| `writer.add_histogram("test/pred", torch.tensor(all_notes), epoch)` | Comparer la distribution des prédictions à celle des vraies notes. |

> ⚠️ Faire l'early stopping sur un jeu de **validation**, pas sur le test final (sinon le score de test est optimiste).

---

## 8. Triplet loss

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `nn.TripletMarginLoss(margin=1.0, p=2.0)` | `margin` α, norme `p` | $\max(0, d(a,p) - d(a,n) + \alpha)$, moyenne sur le batch. Entrées `(B, z)` : ancre, positif, négatif. |
| `nn.TripletMarginWithDistanceLoss(distance_function=...)` | 🔎 ex. `1 - F.cosine_similarity` | Même principe avec une autre distance. |
| `nn.CosineEmbeddingLoss`, `nn.MarginRankingLoss` | 🔎 | Variantes par paires. |

$$\mathcal L(a,p,n) = \max\big(0,\ \lVert a - p\rVert_2 - \lVert a - n\rVert_2 + \alpha\big)$$

**Exercice des points colorés** (on apprend directement les **coordonnées** de 100 points 2D pour regrouper ceux de même couleur) :

```python
X = torch.rand(N, 2, requires_grad=True)            # les "embeddings" = les points eux-mêmes
data  = ColorsDataset(X, C)                          # renvoie (ancre, même couleur, point aléatoire)
dload = DataLoader(data, batch_size=16, shuffle=True)
loss, lr = nn.TripletMarginLoss(), 3e-1
for epoch in range(200):
    for xr, xp, xn in dload:
        l = loss(xr, xp, xn)
        l.backward()                                 # le gradient remonte jusqu'à X (indexation)
        with torch.no_grad():
            X -= lr * X.grad                         # MAJ de TOUTE la matrice X
            X.grad.zero_()
plt.scatter(X[:, 0].detach(), X[:, 1].detach(), c=C.numpy())
```

> 💡 Avec un vrai modèle, les trois éléments passent dans **le même** encodeur (poids partagés) : `loss(f(a), f(p), f(n))`.
> 👉 Choix des négatifs : aléatoires (simple), *hard negatives* (les plus proches de l'ancre, apprentissage plus rapide mais instable), *semi-hard*. `X.grad` se remplit seulement aux lignes tirées dans le batch.

---

## 9. Visualiser des embeddings

| Outil | Paramètres | Explication |
|---|---|---|
| `writer.add_embedding(mat, metadata=labels, global_step=1)` | `mat` `(n, z)`, `metadata` = liste de noms | Onglet **Projector** de TensorBoard : PCA, **t-SNE**, UMAP interactifs, recherche de voisins. `writer.flush()` ensuite. |
| `sklearn.manifold.TSNE(n_components=2, perplexity=30)` | `perplexity` 5–50 | `.fit_transform(model.games.weight.detach().numpy())` puis `plt.scatter`. |
| `F.cosine_similarity(E[i].unsqueeze(0), E)` + `topk` | 🔎 | Les items les plus proches d'un item : vérifier que l'espace a du sens. |

> ⚠️ `.weight` a `requires_grad=True` : `.detach()` avant numpy / sklearn. t-SNE ne préserve que le **voisinage local** (distances entre clusters non interprétables).

---

## 10. Syntaxe Python utile (`*args`, `zip(*…)`)

| Syntaxe | Exemple | Usage |
|---|---|---|
| Déballage étoilé | `pred, *params = model(u, i)` | `params` = liste des sorties restantes. |
| Arguments variables | `def loss_func(pred, y, reg, *params)` puis `loss_func(pred, y, reg, *params)` | Fonction robuste à un nombre quelconque de paramètres à régulariser (ClassicMF : 3, Koren : 4). |
| Ignorer | `_, *res = f()` | Jeter la première valeur. |
| Transposer une liste de tuples | `a, b, c = zip(*[(1, 'x', 3.), (2, 'y', 4.)])` | `(1, 2)`, `('x', 'y')`, `(3., 4.)` : cœur des `collate_fn`. |
| `collections.Counter(liste)` | `Counter(x[1] for x in data)` | Compter les notes par item/user (filtrage). |

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Ids non contigus | `IndexError: index out of range in self` | ré-indexer de 0 à N−1 avec un dictionnaire |
| `emb(5)` avec un `int` | `TypeError` | `emb(torch.tensor(5))` / indices `long` |
| `u @ i.T` pour un batch | sortie `(B, B)` | `(u * i).sum(1)` |
| Init par défaut $\mathcal N(0,1)$ | prédictions énormes, convergence lente | `nn.init.normal_(w, 0, 0.01)` |
| Biais `(B,1)` ajouté à une sortie `(B,)` | broadcast `(B,B)` | `.squeeze(1)` |
| Liste de tuples sans `collate_fn` | batch = liste de tenseurs mal empilés | `collate_fn=tuple_batch` |
| Pas de régularisation avec grand $z$ | val MSE qui remonte | L2 sur les embeddings du batch / `weight_decay` |
| Side info ré-indexée avec un autre mapping | profils incohérents | réutiliser `user_map` / `item_map` |
| `best = model.state_dict()` sans copie | « meilleur » modèle qui change | `copy.deepcopy(...)` |
| Early stopping sur le test | score final optimiste | utiliser un jeu de validation |
| Triplet loss sans marge (α = 0) | tous les points s'effondrent | garder `margin > 0` |
| `.numpy()` sur `.weight` | `Can't call numpy() on Tensor that requires grad` | `.detach().numpy()` |

# 🔥 Cheat Sheet PyTorch — Tenseurs, Autograd, `nn`, `optim`, `DataLoader`, boucle d'entraînement

> Basée sur les notebooks `0-tensors`, `1_1-pytorch-struct`, `1_2-pytorch-grad`, `1_3-pytorch-neuralnet`, `1_4-pytorch-chain` (régression *California housing*, classification MNIST).
> Conventions : `import torch`, `from torch import nn`, `import torch.nn.functional as F`, `from torch.utils.data import TensorDataset, DataLoader, random_split`.
> ℹ️ L'extraction de features avec un CNN pré-entraîné est déjà dans `cs_pytorch.md` — pas répétée ici.

## Sommaire
1. [Tenseurs : création, types, indexation](#1-tenseurs--création-types-indexation)
2. [Opérations, broadcasting et formes](#2-opérations-broadcasting-et-formes)
3. [Autograd et graphe de calcul](#3-autograd-et-graphe-de-calcul)
4. [Descente de gradient « à la main » puis `torch.optim`](#4-descente-de-gradient-à-la-main-puis-torchoptim)
5. [Modules `nn` : couches, activations, pertes](#5-modules-nn--couches-activations-pertes)
6. [Construire un réseau : `Sequential` vs `nn.Module`](#6-construire-un-réseau--sequential-vs-nnmodule)
7. [`Dataset` / `DataLoader`](#7-dataset--dataloader)
8. [Boucle d'entraînement complète (train / eval / `no_grad`)](#8-boucle-dentraînement-complète)
9. [Régularisation : L2, Dropout, BatchNorm](#9-régularisation--l2-dropout-batchnorm)
10. [GPU, checkpointing, TensorBoard](#10-gpu-checkpointing-tensorboard)
11. [📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Tenseurs : création, types, indexation

Un `torch.Tensor` ≈ `numpy.ndarray` mais **typé strictement** (pas de conversion automatique) et exploitable sur GPU.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torch.tensor(data)` | `data` (liste/tuple imbriqué), `dtype=`, `requires_grad=False`, `device=` | Crée un tenseur à partir de données Python. **Copie** les données. Le dtype est déduit (`float` → `float32`, `int` → `int64`). |
| `torch.zeros / ones / empty (*shape)` | `*shape` (entiers ou tuple), `dtype=` | Remplis de 0 / 1 / valeurs mémoire non initialisées (`empty` : ne jamais lire avant d'écrire). |
| `torch.rand / randn (*shape)` | `*shape`, `dtype=` | Loi uniforme $[0,1)$ / loi normale $\mathcal N(0,1)$. |
| `torch.zeros_like(x)` (`ones_like`, `rand_like`, `empty_like`) | `x` | Même forme **et même dtype/device** que `x`. |
| `t.normal_(mean, std)` / `t.random_(a, b)` | `mean`, `std` / bornes | Remplissage **in-place** (suffixe `_`) par une loi. |
| `torch.manual_seed(s)` | `s` (entier) | Fixe la graine → reproductibilité. À rappeler avant chaque tirage à reproduire. |
| `torch.from_numpy(a)` / `t.numpy()` | — | Pont numpy ↔ torch, **mémoire partagée** (modifier l'un modifie l'autre). `.numpy()` exige un tenseur CPU sans gradient (`t.detach().cpu().numpy()`). |
| `t.item()` / `t.tolist()` | — | Tenseur à 1 élément → scalaire Python / tenseur → liste imbriquée. |
| `t.to(dtype_ou_device)` | `torch.float32`, `"cuda"`… | Convertit le type ou le device. Raccourcis : `.float()`, `.double()`, `.long()`, `.int()`, `.bool()`. |
| `t.shape` / `t.size(dim)` / `t.dim()` / `t.numel()` | — | Forme (`torch.Size`) / taille d'un axe / nb de dimensions / nb total d'éléments. |

**dtypes à connaître** : `torch.float32` (`float`, défaut des modèles), `float64` (`double`, défaut de numpy), `int64` (`long`, **requis pour les indices/labels de classe**), `int32`, `uint8` (images brutes), `bool` (masques).

```python
x = torch.tensor([[1., 2., 3.], [2., 3., 4.]])      # (2, 3) float32
x[0, 0]            # tenseur 0-D ; x[0, 0].item() -> float Python
x[:, 0]            # toute la 1re colonne ; x[0, [0, 2]] -> colonnes 0 et 2
x[0, ::2]          # slicing début:fin(exclu):pas, comme numpy
a = torch.rand(6, 6)
a[a <= 0.5] = 0; a[a > 0.75] = 3          # masque booléen = indexation conditionnelle
(a == 0).sum()                             # comptage direct
```

💡 `torch.tensor(...)` **copie**, `torch.from_numpy(...)` **partage**. `b = a` ne copie rien (simple étiquette) → utiliser `b = a.clone()`.

---

## 2. Opérations, broadcasting et formes

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `a + b`, `a * b`, `a ** 2`, `a / b` | — | Opérations **terme à terme** (avec broadcasting). `*` n'est **pas** un produit matriciel. |
| `a @ b` / `a.mm(b)` / `torch.matmul(a, b)` | — | Produit matriciel (`matmul` gère aussi les batchs). `a.dot(b)` : produit scalaire de deux **vecteurs 1-D** uniquement. |
| `a.t()` / `a.T` / `a.transpose(d0, d1)` | `d0, d1` | Transposition (`.t()` 2-D seulement). |
| `sum / mean / std / max / argmax / prod` | `dim=` (int), `keepdim=False` | Réductions. Sans `dim` → scalaire global. `max(dim)` renvoie `(valeurs, indices)`. |
| `torch.cat((a, b), dim)` | tuple de tenseurs, `dim` | Concatène **le long d'un axe existant**. `torch.stack` crée un **nouvel** axe. |
| `t.view(*shape)` / `t.reshape(*shape)` | `shape` (un `-1` = « devine ») | Change la forme sans changer le nombre d'éléments. `view` exige une mémoire contiguë ; `reshape` copie si besoin. |
| `t.squeeze(dim)` / `t.unsqueeze(dim)` | `dim` | Supprime un axe de taille 1 / ajoute un axe de taille 1 (ex. `unsqueeze(0)` → batch de 1). `squeeze` sans effet si l'axe ≠ 1. |
| `t.permute(*dims)` | nouvel ordre des axes | Réordonne les axes (ex. `(N,H,W,C)` → `(N,C,H,W)`). |
| `t.clone()` / `t.detach()` | — | Copie indépendante / vue **détachée du graphe** (partage la mémoire). |
| `torch.nn.functional.one_hot(idx, num_classes)` | `idx` en `int64` | Encodage one-hot ; renvoie un `int64` (le convertir en `float` pour une MSE). |
| `torch.clamp(t, min, max)`, `abs`, `exp`, `log`, `sin`… | — | Fonctions élémentaires. Version in-place : suffixe `_` (`t.add_(u)`, `t.clamp_()`…). |

**Broadcasting** (mêmes règles que numpy) : on compare les formes **de droite à gauche** ; chaque axe doit être égal, ou valoir 1, ou être absent.

```python
a = torch.ones(4, 3, 2)
a * torch.rand(3, 2)        # OK  (axe 0 absent)
a * torch.rand(3, 1)        # OK  (axe 2 = 1)
a * torch.rand(3)           # ERREUR -> b.unsqueeze(1) donne (3,1) et ça marche
```

⚠️ **Piège n°1 du cours : le broadcasting silencieux.** `(5,1) - (5,)` donne une matrice `(5,5)` (pas un vecteur) : la somme passe de 5 à 25 sans aucune erreur. Dans une fonction de coût, c'est une erreur de calcul invisible. Réflexe : `yhat.view(-1)` ou `y.view(-1, 1)` pour que **les deux formes soient identiques**, et `print(x.shape)` en cas de doute.

⚠️ **Pas de conversion à la volée** : `a @ [[1],[1],[1]]` ou `a * [2., 2., 2.]` (liste Python) lèvent une erreur de type. Convertir en tenseur du même dtype. Mélanger `float32` et `float64` (ou `int`) dans une opération échoue aussi.

⚠️ **`view`/`reshape` renvoient souvent une vue** : modifier la source modifie le résultat. `clone()` si on veut une copie.

---

## 3. Autograd et graphe de calcul

PyTorch construit **dynamiquement** un graphe acyclique des opérations, puis rétropropage les dérivées partielles (règle de la chaîne). Toute opération est une `Function` avec `forward` (calcul) et `backward` (dérivées locales).

| Élément | Paramètres essentiels | Explication |
|---|---|---|
| `requires_grad=True` | à la création (`torch.tensor(…, requires_grad=True)`) ou `t.requires_grad = True` | Active le suivi : toute opération utilisant ce tenseur est enregistrée. **Seuls les tenseurs flottants** peuvent avoir un gradient. Par défaut `False`. |
| `z.backward()` | `gradient=None` (obligatoire si `z` n'est pas scalaire), `retain_graph=False` | Rétropropage depuis `z` (doit être un **scalaire**) : remplit `.grad` de toutes les feuilles avec `requires_grad`. |
| `t.grad` | — | Gradient accumulé (même forme que `t`). `None` tant qu'aucun `backward`. Seules les **feuilles** (tenseurs créés par l'utilisateur) le conservent. |
| `t.grad.zero_()` / `optim.zero_grad()` | — | Remise à zéro du gradient. |
| `t.detach()` | — | Copie détachée du graphe (plus de suivi). `t.data` fait pareil mais est déconseillé (contourne les contrôles). |
| `with torch.no_grad():` | — | Désactive la construction du graphe dans le bloc (évaluation, mise à jour manuelle de poids). |
| `torch.set_grad_enabled(bool)` | `bool` | Version « conditionnelle » de `no_grad` (utile pour fusionner train/test dans une boucle). |

```python
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
z = 2*a + b
z.backward()
a.grad.item(), b.grad.item()      # 2.0, 1.0
```

⚠️ **Le gradient s'accumule** : un 2ᵉ `backward()` **ajoute** au `.grad` existant (il ne l'écrase pas). C'est pourquoi on met `zero_grad()` à **chaque itération**. Le cours le fait observer volontairement avec la MSE : le 2ᵉ gradient est le double du 1ᵉʳ.

⚠️ Sans `requires_grad`, `backward()` lève une erreur (« element 0 of tensors does not require grad »).

💡 Gradient de la MSE $\frac1{2N}\sum_i\lVert\hat y_i-y_i\rVert^2$ : $\partial/\partial\hat y_i = \frac1N(\hat y_i-y_i)$ et $\partial/\partial y_i = -\frac1N(\hat y_i-y_i)$ — utile pour **vérifier** le résultat d'autograd.

---

## 4. Descente de gradient « à la main » puis `torch.optim`

Régression linéaire $f(x,w,b)=x\,w^\top+b$, coût $\frac1N\sum_i\lVert f(x^i)-y^i\rVert^2$.

```python
w = torch.randn(1, d, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)
for it in range(EPOCHS):
    loss = ((x @ w.T + b).view(-1) - y).pow(2).mean()   # formes identiques !
    loss.backward()
    with torch.no_grad():            # mise à jour hors graphe
        w -= EPS * w.grad            # -= / += : in-place OBLIGATOIRE
        b -= EPS * b.grad
    w.grad.zero_(); b.grad.zero_()
```

| Méthode | Explication |
|---|---|
| `w.data -= EPS * w.grad` | Ancienne variante : `.data` évite l'enregistrement dans le graphe. |
| `w = w - EPS*w.grad` | ❌ **Faux** : crée un **nouveau** tenseur (hors des feuilles) qui n'est plus connecté / sans `requires_grad` → le modèle ne s'entraîne plus. |

### `torch.optim`

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `optim.SGD(params, lr, momentum=0, weight_decay=0, nesterov=False)` | `params` = `model.parameters()` ou `[w, b]`, `lr` = pas | Descente de gradient (mini-batch). `momentum≈0.9` amortit les oscillations. |
| `optim.Adam(params, lr=1e-3, betas=(0.9,0.999), weight_decay=0)` | idem | Pas adaptatif par paramètre : choix par défaut en pratique, peu sensible au `lr`. |
| `optim.AdamW(params, lr, weight_decay=0.01)` | idem | Adam avec *weight decay* correctement découplé (préférable à `Adam(weight_decay=…)` pour du L2). |
| `opt.zero_grad()` | — | Remet à zéro tous les `.grad` des paramètres. |
| `opt.step()` | — | Applique la mise à jour avec les gradients courants. |
| `opt.state_dict()` / `load_state_dict()` | — | Pour le checkpointing (moments d'Adam inclus). |

Ordre standard dans la boucle : **`zero_grad()` → forward → loss → `backward()` → `step()`**.

💡 Si la loss ne converge pas ou explose : réduire `lr` (le TD joue sur `EPS=1e-7`…`1e-2` : sur des données **non normalisées**, un `lr` trop grand diverge). Normaliser les entrées (`(x - x.mean(0)) / x.std(0)`) règle l'essentiel.

⚠️ Créer l'optimiseur **après** `model.to(device)` et **après** avoir fixé quels paramètres sont gelés, sinon il n'optimise pas les bons tenseurs.

---

## 5. Modules `nn` : couches, activations, pertes

Un module encapsule une fonction + ses **paramètres appris** (liste automatique via `.parameters()` / `.named_parameters()`).

### Couches

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Linear(in_features, out_features, bias=True)` | dimensions | $y = xW^\top + b$ ; accepte `(N, in)` ou `(*, in)`. Poids de forme `(out, in)`. |
| `nn.Flatten(start_dim=1)` | `start_dim` | Aplati tout sauf le batch. Équivalent propre de `x.view(x.size(0), -1)`. |
| `nn.Dropout(p=0.5)` | `p` = proba d'annuler une activation | Régularisation ; **actif seulement en `.train()`** (voir §9). |
| `nn.BatchNorm1d(num_features)` | nb de features (2-D) / canaux (3-D) | Normalise par batch ; mode train ≠ eval. |
| `nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)` | taille du vocab, dim | Table de lookup `indices (int64) → vecteurs` (voir `cs_representations_reco_texte.md`). |

### Activations

| Classe | Formule / rôle |
|---|---|
| `nn.ReLU()` | $\max(0,x)$ — défaut des couches cachées. |
| `nn.Tanh()` | $\tanh x\in(-1,1)$. |
| `nn.Sigmoid()` | $\sigma(x)=\frac1{1+e^{-x}}\in(0,1)$ — sortie binaire. |
| `nn.Softmax(dim)` | $\mathrm{softmax}(x)_i=\dfrac{e^{x_i}}{\sum_j e^{x_j}}$ — distribution de probabilité. `dim=1` pour `(N, C)`. |
| `nn.LogSoftmax(dim)` | $\log\mathrm{softmax}$ — stable numériquement, entrée de `NLLLoss`. |

### Fonctions de coût (*loss*)

| Classe | Paramètres essentiels | Quand / formule |
|---|---|---|
| `nn.MSELoss(reduction='mean')` | `reduction` ∈ `'mean'`,`'sum'`,`'none'` | Régression. $\frac1N\sum(\hat y-y)^2$. Prédiction et cible **de même forme**. |
| `nn.L1Loss()` | idem | Régression robuste aux outliers (erreur absolue). |
| `nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)` | `weight` (tenseur de poids par classe), `ignore_index` (classe ignorée, ex. padding) | **Classification multi-classe.** Entrée : **logits bruts** `(N, C)` ; cible : indices `int64` `(N,)`. Combine `LogSoftmax` + `NLLLoss`. |
| `nn.NLLLoss()` | — | $-x_y$ : attend des **log-probabilités** (donc après `LogSoftmax`). |
| `nn.BCELoss()` / `nn.BCEWithLogitsLoss()` | `pos_weight` (version logits) | Binaire / multi-label. `BCELoss` attend des probabilités (après `Sigmoid`) ; `BCEWithLogitsLoss` attend des **logits** (plus stable → à préférer). |

Définition (négative log-vraisemblance) : $\mathrm{NLL}(y,\mathbf x)=-x_y$ où $y$ est l'indice de la bonne classe et $\mathbf x$ le vecteur de log-probabilités.

⚠️ **Ne pas mettre de `Softmax` en sortie du réseau quand on utilise `CrossEntropyLoss`** (déjà inclus → softmax appliqué deux fois, apprentissage dégradé). Le softmax n'est ajouté « à la main » que pour *lire* des probabilités à l'inférence (ex. saliency).

⚠️ **Problème binaire avec `CrossEntropyLoss`** : mettre **2 sorties** (pas 1) et des labels `0/1` en `long`. Avec 1 sortie → `BCEWithLogitsLoss` (cibles `float`).

⚠️ `MSELoss` pour de la classification demande des cibles one-hot en `float` ; c'est possible mais moins adapté que la cross-entropie.

👉 Le TD compare : MSE+one-hot (moins bon) → CrossEntropy (le standard) ; Tanh → ReLU (converge plus vite, évite la saturation).

---

## 6. Construire un réseau : `Sequential` vs `nn.Module`

```python
# (a) Conteneur séquentiel : suffisant pour un enchaînement linéaire de couches
net = nn.Sequential(nn.Linear(d, 5), nn.Tanh(), nn.Linear(5, 1))

# (b) Module personnalisé : dès qu'il y a du branchement, des retours multiples, etc.
class DeuxCouches(nn.Module):
    def __init__(self):
        super().__init__()                      # OBLIGATOIRE en premier
        self.un   = nn.Linear(d, 5)             # sous-modules = attributs
        self.act  = nn.Tanh()
        self.deux = nn.Linear(5, 1)
    def forward(self, x):                       # on appelle net(x), jamais net.forward(x)
        return self.deux(self.act(self.un(x)))

# (c) Construire une liste de couches dynamiquement
layers = []
dim = in_features
for h in [100, 100, 100]:
    layers += [nn.Linear(dim, h), nn.ReLU()]
    dim = h
layers.append(nn.Linear(dim, out_features))
net = nn.Sequential(*layers)                    # * = dépaquetage de la liste
```

| Outil | Explication |
|---|---|
| `model.parameters()` / `model.named_parameters()` | Itère sur les paramètres (utile pour l'optimiseur, les L2, compter les paramètres). |
| `sum(p.numel() for p in model.parameters() if p.requires_grad)` | Nombre de paramètres **entraînables**. |
| `model.reset_parameters()` (couches simples) | Réinitialise les poids. |
| `nn.init.xavier_uniform_(w)` / `normal_(w, mean, std)` | Initialisations manuelles (Xavier : bon défaut avec tanh/sigmoïde ; Kaiming pour ReLU). |
| `model.name = "…"` | Simple attribut Python, pratique pour nommer les runs (TensorBoard, sauvegardes). |

⚠️ Les sous-modules doivent être des **attributs** (`self.xxx = …`) pour que leurs paramètres soient vus par `.parameters()`. Une liste Python de couches ne suffit pas → `nn.ModuleList([...])`. Un tenseur non-paramètre à déplacer avec le modèle → `self.register_buffer("nom", tenseur)`.

---

## 7. `Dataset` / `DataLoader`

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `TensorDataset(*tensors)` | tenseurs de même 1ʳᵉ dimension | Renvoie `(x[i], y[i], …)`. |
| `Dataset` (à sous-classer) | `__len__`, `__getitem__(i)` | Seules ces deux méthodes sont nécessaires. Une simple **liste Python** marche aussi (duck typing). |
| `DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=None, sampler=None, num_workers=0, drop_last=False, pin_memory=False)` | `batch_size`, `shuffle`, `collate_fn` | Itérateur sur des mini-batchs. `shuffle=True` pour le train, **`False` pour le test**. `collate_fn` assemble la liste d'exemples en batch (indispensable pour du texte à longueurs variables). |
| `random_split(dataset, [n_train, n_val])` | longueurs (entiers) | Découpe aléatoire. Les longueurs doivent sommer à `len(dataset)`. |

```python
data = TensorDataset(x, y)
n_train = int(0.8 * len(data))
train_ds, valid_ds = random_split(data, [n_train, len(data) - n_train])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=16)            # pas de shuffle
len(train_loader)                                             # nb de batchs = ceil(N / batch_size)
```

💡 **Toujours normaliser** (centrer-réduire) entrées et, en régression, les cibles : `x = (x - x.mean(0)) / x.std(0)`. Les statistiques se calculent sur le **train** seulement (fuite de données sinon).

⚠️ Le **dernier batch peut être plus petit** → pour une moyenne exacte de la loss/accuracy, pondérer par `len(x)` : `cumloss += loss.item()*len(x)` puis diviser par le nombre total d'exemples.

---

## 8. Boucle d'entraînement complète

Squelette standard (celui de toute la suite du cours) :

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.to(device)                                  # AVANT l'optimiseur
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()                                       # mode apprentissage
    cumloss, cumacc, count = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        yhat = model(x)                                 # forward
        l = loss_fn(yhat, y)
        l.backward()                                    # backward
        optim.step()                                    # MAJ des poids
        cumloss += l.item() * len(x)
        cumacc  += (yhat.argmax(1) == y).float().sum().item()
        count   += len(x)
    # --- validation ---
    model.eval()                                        # dropout/BN en mode inférence
    with torch.no_grad():                               # pas de graphe
        ...                                             # même boucle, sans backward/step
```

| Élément | Explication |
|---|---|
| `model.train()` / `model.eval()` | Change le **comportement des modules** (Dropout, BatchNorm, certains RNN). N'agit pas sur le graphe. |
| `torch.no_grad()` | Désactive la **construction du graphe** (gain de mémoire/temps). Ne change pas le comportement de Dropout/BN. |
| `loss.item()` | Valeur scalaire Python. Sans `.item()`, accumuler la loss **garde le graphe en mémoire** (fuite mémoire). |
| `yhat.argmax(1)` / `yhat.max(1).indices` | Classe prédite. Accuracy : `(yhat.argmax(1) == y).float().mean()`. |

⚠️ **`no_grad` ≠ `eval`** : ce sont deux mécanismes indépendants, et on a besoin des **deux** en évaluation.

⚠️ Dans l'accuracy du cours : `y` doit être de forme `(N,)` (indices) ; si `y` est one-hot on compare `argmax(y, 1)`.

⚠️ Bug classique du TD `1_4` : dans la boucle de validation, l'accumulation `cumloss += …` doit être **dans** le `for` sur les batchs (sinon seul le dernier batch compte).

⚠️ `1_3` : dans la boucle sur `loader`, calculer la loss sur `bx` (le mini-batch) et non sur tout `housing_x` — sinon le DataLoader ne sert à rien.

💡 Sur-apprentissage : train ↓ mais validation ↑ → solutions : plus de données, régularisation (§9), réseau plus petit, *early stopping* (s'arrêter dès que la loss de validation cesse de s'améliorer, en gardant `copy.deepcopy(model.state_dict())` du meilleur modèle).

---

## 9. Régularisation : L2, Dropout, BatchNorm

| Technique | Mise en œuvre | Explication |
|---|---|---|
| **Pénalisation L2** (*weight decay*) | `loss = loss + λ·Σ‖W‖²` ou `optim.AdamW(..., weight_decay=λ)` | Favorise des poids faibles. Valeurs testées : $\lambda\in\{0,10^{-5},10^{-4},10^{-3},10^{-2}\}$. L1 : $\lVert W\rVert_1$ (favorise des poids exactement nuls). |
| `nn.Dropout(p)` | À placer **après** la couche linéaire/activation (ex. `Linear → Dropout → ReLU`) | Met aléatoirement à 0 une fraction `p` des activations pendant l'apprentissage (≈ moyenne de nombreux sous-réseaux, réduit la variance). Pas de dropout en sortie. |
| `nn.BatchNorm1d(n)` | `Linear → BatchNorm1d → ReLU` | Centre-réduit les activations avec les statistiques **du batch** en train, et une **moyenne/variance glissante** en eval. Stabilise et accélère l'apprentissage. |

```python
# Pénalisation L2 sur les poids uniquement (pas les biais)
l2 = sum((p**2).sum() for n, p in model.named_parameters() if n.endswith(".weight"))
loss = ce(yhat, y) + lam * l2
```

⚠️ **Dropout et BatchNorm exigent `model.train()` / `model.eval()`** : oublier `eval()` en test donne des prédictions bruitées et non reproductibles.

⚠️ `BatchNorm` avec des batchs de taille 1 (ou très petits) est instable en train.

---

## 10. GPU, checkpointing, TensorBoard

### GPU
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mac Apple Silicon : torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)                          # modèle (en place)
x, y = x.to(device), y.to(device)                 # tenseurs : to() RENVOIE une copie -> réaffecter !
```
⚠️ **Tout doit être sur le même device** (modèle, entrées, labels), sinon erreur. `t.device` donne le device d'un tenseur ; pour un modèle, regarder `next(model.parameters()).device`. Pour afficher/convertir : `t.detach().cpu().numpy()`.

### Checkpointing (sauvegarde/reprise)
```python
def save_state(epoch, model, optim, fichier):
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': optim.state_dict()}, fichier)

def load_state(fichier, model, optim):
    epoch = 0
    if os.path.isfile(fichier):
        s = torch.load(fichier)                   # map_location="cpu" si pas de GPU
        model.load_state_dict(s['model_state'])
        optim.load_state_dict(s['optim_state'])
        epoch = s['epoch']
    return epoch
start = load_state(f, model, optim)
for epoch in range(start, EPOCHS): ...
```
- `state_dict()` = dictionnaire `nom → tenseur` (poids seuls, pas l'architecture) → pour recharger il faut **recréer le même modèle** puis `load_state_dict`.
- Sauvegarder aussi l'optimiseur (moments d'Adam) pour **reprendre** un entraînement ; pour l'inférence seule le modèle suffit.
- 🔎 Depuis PyTorch 2.6, `torch.load` utilise `weights_only=True` par défaut : charger un dict contenant des objets Python arbitraires (ex. un `preprocessor`) exige `torch.load(f, weights_only=False)` (uniquement sur un fichier de confiance).

### TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"/tmp/logs/{model.name}")      # 1 dossier par run
writer.add_scalar("loss/train", loss_value, epoch)      # tag, valeur, step ("/" = regroupement)
writer.add_histogram("linear/0/weight", layer.weight, epoch)
writer.add_embedding(emb_matrix, metadata=labels)       # projection t-SNE/PCA
```
Lancement : `tensorboard --logdir /tmp/logs` (navigateur) ; dans un notebook : `%load_ext tensorboard` puis `%tensorboard --logdir /tmp/logs`. Utiliser un **nom de run différent** pour chaque expérience afin de comparer les courbes.

---

## 📌 Récapitulatif des pièges classiques

| # | Piège | Réflexe |
|---|---|---|
| 1 | Broadcasting silencieux `(N,1) - (N,)` → `(N,N)` | Aligner les formes (`view(-1)`), vérifier `.shape` dans les pertes. |
| 2 | `dtype` : float64 (numpy) vs float32 (modèle) ; labels non `long` | `.float()` sur les entrées, `.long()` sur les labels de classe. |
| 3 | Oublier `zero_grad()` → gradients cumulés | `optim.zero_grad()` à chaque itération. |
| 4 | `w = w - lr*w.grad` casse le graphe | Mise à jour in-place (`-=`) sous `torch.no_grad()`. |
| 5 | `Softmax` + `CrossEntropyLoss` | Sortie = logits bruts ; le softmax est déjà dans la loss. |
| 6 | Binaire avec `CrossEntropyLoss` et 1 seule sortie | 2 sorties, ou `BCEWithLogitsLoss` avec 1 sortie. |
| 7 | Oublier `model.eval()` + `torch.no_grad()` en validation | Les deux, systématiquement. |
| 8 | Accumuler `loss` au lieu de `loss.item()` | `.item()` (sinon fuite mémoire). |
| 9 | Modèle / données sur des devices différents | `.to(device)` sur modèle **et** batchs ; `to()` sur tenseur renvoie une copie. |
| 10 | Optimiseur créé avant `model.to(device)` / avant le gel de paramètres | Créer l'optimiseur en dernier. |
| 11 | Dernier batch plus petit → moyenne biaisée | Pondérer par `len(x)`. |
| 12 | `shuffle=True` sur le test, ou normalisation calculée sur tout le jeu | Shuffle en train seulement ; stats sur le train. |
| 13 | `b = a` ne copie pas ; `view`/`from_numpy`/`detach` partagent la mémoire | `clone()` pour copier. |
| 14 | Données non normalisées → divergence | Centrer-réduire entrées (et cibles en régression), ajuster `lr`. |
| 15 | `torch.load` d'objets Python (PyTorch ≥ 2.6) | `weights_only=False` si fichier de confiance. |

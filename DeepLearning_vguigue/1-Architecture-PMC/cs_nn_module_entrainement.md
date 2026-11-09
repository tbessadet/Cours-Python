# 🧠 Cheat Sheet PyTorch — Construire et entraîner un réseau (`torch.nn`)

> Basée sur les notebooks `1_1-pytorch-neuralnet` et `1_2-pytorch-chain`.
> Convention : `import torch`, `from torch import nn`, `from torch.utils.data import TensorDataset, DataLoader, random_split`

## Sommaire

**I. Construire le réseau**
1. [Couches et activations](#1-couches-et-activations)
2. [Assembler : `nn.Sequential` et `nn.Module`](#2-assembler--nnsequential-et-nnmodule)
3. [Fonctions de coût](#3-fonctions-de-coût)

**II. Entraîner**

4. [Données : Dataset, DataLoader, split](#4-données--dataset-dataloader-split)
5. [Optimiseurs et learning rate](#5-optimiseurs-et-learning-rate)
6. [La boucle d'apprentissage standard](#6-la-boucle-dapprentissage-standard)
7. [Classification multi-classes (MNIST)](#7-classification-multi-classes-mnist)

**III. Régulariser**

8. [Pénalisation L1/L2, Dropout, BatchNorm](#8-pénalisation-l1l2-dropout-batchnorm)

**IV. Outils**

9. [GPU / device](#9-gpu--device)
10. [Checkpointing (sauvegarde / chargement)](#10-checkpointing-sauvegarde--chargement)
11. [TensorBoard et tqdm](#11-tensorboard-et-tqdm)
12. [Lambda et one-hot](#12-lambda-et-one-hot)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Construire le réseau

## 1. Couches et activations

| Module | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Linear(in_features, out_features, bias=True)` | dimensions d'entrée / sortie | Couche affine $\mathbf z = \mathbf x W^\top + \mathbf b$. Entrée `(batch, in)` → sortie `(batch, out)`. Poids initialisés aléatoirement (uniforme de Kaiming). `.weight` de forme `(out, in)`, `.bias` `(out,)`. `reset_parameters()` ré-initialise. |
| `nn.Tanh()` | — | Activation dans $]-1, 1[$, centrée. Sature aux extrêmes. |
| `nn.Sigmoid()` | — | Activation dans $]0, 1[$ ; pour une **probabilité** en sortie binaire. Sature → *vanishing gradient* en couches cachées. |
| `nn.ReLU()` | `inplace=False` | $\max(0, x)$ : **le défaut** en couches cachées (gradient 1 si actif, pas de saturation). |
| `nn.Softmax(dim=1)` | `dim` = axe des classes | Transforme des scores en distribution (somme 1). ⚠️ **Toujours préciser `dim`** (1 pour un batch `(N, K)`). |
| `nn.LogSoftmax(dim=1)` | idem | Log-probabilités, à associer à `nn.NLLLoss`. |
| `nn.Dropout(p=0.5)` | `p` = proba d'éteindre | Voir §8. |
| `nn.BatchNorm1d(num_features)` | nb de features | Voir §8. |
| `nn.Flatten(start_dim=1)` | 🔎 | Aplatit `(N, 1, 28, 28)` → `(N, 784)` dans un `Sequential`. |

> 👉 Choix d'activation : **ReLU** en couches cachées ; en **sortie** : rien (régression ou `CrossEntropyLoss`), `Sigmoid` (binaire / multi-label avec `BCELoss`), `Softmax` seulement pour afficher des probabilités.

---

## 2. Assembler : `nn.Sequential` et `nn.Module`

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Sequential(m1, m2, ...)` | modules dans l'ordre | Enchaîne les modules ; le plus rapide pour un réseau « en ligne ». `nn.Sequential(*layers)` construit depuis une **liste**. |
| Sous-classe de `nn.Module` | `__init__` + `forward` | Pour une architecture libre (branches, plusieurs entrées…). **Toujours `super().__init__()`** en premier. Le `backward` est **automatique**. |
| `model(x)` | — | Appelle `forward` (+ hooks). Ne pas appeler `model.forward(x)` directement. |
| `model.parameters()` | — | Itérateur sur tous les paramètres (à donner à l'optimiseur). |
| `model.named_parameters()` | — | Paires `(nom, tenseur)` : `'0.weight'`, `'2.bias'`… Pratique pour cibler les `.weight` (pénalisation). |
| `model.train()` / `model.eval()` | — | Mode apprentissage / inférence (change Dropout, BatchNorm). |
| `model.name = "..."` | attribut libre | Astuce du cours pour nommer les runs TensorBoard / fichiers. |

```python
# Réseau générique : liste de dimensions cachées → Sequential
class LinearMultiClass(nn.Module):
    def __init__(self, in_features, out_features, dims=[16], activation=nn.Tanh):
        super().__init__()
        layers, dim = [], in_features
        for newdim in dims:                       # [32, 20] → in→32→20→out
            layers += [nn.Linear(dim, newdim), activation()]
            dim = newdim
        layers.append(nn.Linear(dim, out_features))   # pas d'activation finale
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

net = LinearMultiClass(28*28, 10, dims=[64]*2, activation=nn.ReLU)
```

> ⚠️ Passer la **classe** d'activation (`nn.ReLU`) et l'**instancier** dans la boucle (`activation()`) : chaque couche a son propre module.
> ⚠️ Des couches stockées dans une **liste Python** (`self.l = [nn.Linear(...)]`) ne sont **pas enregistrées** → paramètres absents de `.parameters()`. Utiliser `nn.Sequential` ou `nn.ModuleList`.

---

## 3. Fonctions de coût

| Classe | Entrées attendues | Explication / quand l'utiliser |
|---|---|---|
| `nn.MSELoss(reduction='mean')` | `yhat` et `y` **de même forme**, float | Régression. $\frac1N\sum(\hat y - y)^2$. `reduction='sum'` / `'none'` possibles. |
| `nn.L1Loss()` | idem | Erreur absolue moyenne, plus robuste aux valeurs aberrantes. |
| `nn.CrossEntropyLoss(weight=None, label_smoothing=0.)` | `yhat` = **scores bruts** `(N, K)` ; `y` = **indices** `(N,)` `long` | **Le standard en multi-classes** : combine `LogSoftmax` + `NLLLoss`. **Pas de Softmax** avant. `weight` : pondérer des classes déséquilibrées. |
| `nn.NLLLoss()` | `yhat` = **log-probas** (`LogSoftmax`), `y` indices | Version « manuelle » de la CE. |
| `nn.BCELoss()` | `yhat` dans $]0,1[$ (après `Sigmoid`), `y` float 0/1 | Binaire / multi-label. |
| `nn.BCEWithLogitsLoss()` | `yhat` = scores bruts | 🔎 Sigmoid + BCE fusionnés : **plus stable**, à préférer. |

Cross-entropie (multi-classes), hypothèses : $\mathbf z \in \mathbb R^K$ scores, $k$ indice de la vraie classe :

$$\mathrm{CE} = -\log\frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} = -z_k + \log\sum_{j=1}^{K} e^{z_j}$$

- $z_k$ : score de la bonne classe ; $K$ : nombre de classes.

> ⚠️ **Formes** : `MSELoss(net(x), y)` avec `net(x)` en `(N,1)` et `y` en `(N,)` → broadcast `(N,N)` silencieux (+ warning). Aligner avec `.view(-1)` ou `y.view(-1, 1)`.

---

# II. Entraîner

## 4. Données : Dataset, DataLoader, split

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `TensorDataset(X, Y)` | tenseurs de même 1re dimension | Dataset dont l'élément `i` est `(X[i], Y[i])`. `len(ds)`, `ds[5]`. |
| `Dataset` (classe à hériter) | `__len__`, `__getitem__(i)` | Pour des données sur disque / prétraitements à la volée. |
| `DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=False, collate_fn=None)` | `batch_size` (16–300 dans le TP), **`shuffle=True` pour le train**, `False` pour test/valid | Itérateur de **mini-batchs** `(bx, by)` empilés. `len(loader)` = nombre de batchs = $\lceil N / \text{batch\_size}\rceil$. `num_workers>0` : chargement parallèle. |
| `random_split(ds, [n_train, n_valid])` | longueurs (ou fractions) ; `generator=torch.Generator().manual_seed(0)` | Découpe aléatoire train/valid/test. |
| `next(iter(loader))` | — | Récupérer **un** batch pour inspecter les formes. |

```python
all_data   = (all_data - all_data.mean(0)) / all_data.std(0)     # normaliser : toujours bon
all_labels = (all_labels - all_labels.mean()) / all_labels.std()
ds = TensorDataset(all_data, all_labels)
n_train = int(0.8 * len(ds))
train_data, valid_data = random_split(ds, [n_train, len(ds) - n_train])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
bx, by = next(iter(train_loader)); print(bx.shape, by.shape)    # (32, 8) (32,)
```

> 💡 Une **époque** = un passage complet sur les données = `len(loader)` mises à jour ; chaque exemple est vu **une fois par époque**.
> ⚠️ 🔎 Calculer moyenne/écart-type de normalisation **sur le train seulement** en pratique (sinon fuite d'information vers le test).

---

## 5. Optimiseurs et learning rate

| Classe | Paramètres essentiels | Réglage |
|---|---|---|
| `torch.optim.SGD(params, lr, momentum=0, weight_decay=0)` | `lr` | Très sensible à `lr` : sur des données non normalisées, il faut un `lr` minuscule (1e-7 sur *housing* brut) ; sur données normalisées, 1e-2 à 1e-1. `momentum=0.9` recommandé. |
| `torch.optim.Adam(params, lr=1e-3, weight_decay=0)` | `lr=1e-3` | **Le réflexe** : robuste, peu de réglage. |
| `torch.optim.AdamW(params, lr, weight_decay=1e-2)` | `weight_decay` | Adam + pénalisation L2 découplée (meilleure régularisation). |
| `torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs, steps_per_epoch)` | `max_lr` | Monte puis descend le lr ; `scheduler.step()` **après** `optim.step()`, à chaque batch. |
| `torch.optim.lr_scheduler.StepLR(opt, step_size, gamma=0.1)` | 🔎 | Divise le lr tous les `step_size` epochs (`scheduler.step()` par epoch). |
| `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` | 🔎 `max_norm=1.0` | *Gradient clipping* : entre `backward()` et `step()`. |

> ⚠️ Créer l'optimiseur **après** `model.to(device)` (il doit référencer les paramètres déjà déplacés).

---

## 6. La boucle d'apprentissage standard

```python
EPOCHS, BATCH_SIZE = 50, 32
net = nn.Sequential(nn.Linear(d, 5), nn.Tanh(), nn.Linear(5, 1)).to(device)
MyLoss = nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in tqdm(range(EPOCHS)):
    net.train()                                   # mode apprentissage
    cumloss = 0
    for xbatch, ybatch in train_loader:
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)
        loss = MyLoss(net(xbatch).view(-1), ybatch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        cumloss += loss.item()                     # .item() : float, pas de graphe
    summary.add_scalar("loss/train", cumloss / len(train_loader), epoch)

    if epoch % 10 == 0:                            # validation périodique
        net.eval()
        with torch.no_grad():                      # pas de graphe → rapide, économe
            vloss = sum(MyLoss(net(x.to(device)).view(-1), y.to(device)).item()
                        for x, y in valid_loader)
        summary.add_scalar("loss/validation", vloss / len(valid_loader), epoch)
```

| Mécanisme | Agit sur | Quand |
|---|---|---|
| `torch.no_grad()` | le **graphe de calcul** (comme si `requires_grad=False`) | **Toute** évaluation / inférence |
| `model.eval()` / `model.train()` | le **comportement des modules** (Dropout, BatchNorm, RNN) | Avant valid/test / avant l'entraînement |

> 💡 Les deux sont **indépendants et complémentaires** : en évaluation, on fait **les deux**.
> 👉 Moyenner la loss par batch (`cumloss / len(loader)`) ou par exemple (`cumloss += l.item()*len(x)` puis `/count`) pour une grandeur interprétable.

---

## 7. Classification multi-classes (MNIST)

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `torchvision.datasets.MNIST(root, train=True, transform=None, download=True)` | `train` : 60 000 / 10 000 images | Chiffres manuscrits 28×28, labels 0–9. |
| `transforms.Compose([...])` | liste de transformations | Chaîne de prétraitements appliquée à chaque image. |
| `transforms.ToTensor()` | — | Image PIL → tenseur `float` `(C, H, W)` dans [0, 1]. |
| `transforms.Normalize(mean, std)` | tuples par canal : `(0.1307,), (0.3081,)` pour MNIST | Centre-réduit par canal. (Le TP utilise `(0.,), (1.,)` = identité.) |

```python
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)

def accuracy(yhat, y):                              # y en indices OU en one-hot
    if y.dim() == 1 or y.size(1) == 1:
        return (yhat.argmax(1) == y.view(-1)).float().mean()
    return (yhat.argmax(1) == y.argmax(1)).float().mean()

for x, y in train_loader:
    x = x.view(x.size(0), -1).to(device)            # (N,1,28,28) → (N,784)
    yhat = model(x)                                 # scores bruts (N,10)
    l = nn.CrossEntropyLoss()(yhat, y.to(device))   # y : indices long
```

| Option de coût | Sortie du réseau | Étiquettes |
|---|---|---|
| **CE** (recommandé) | scores bruts, **pas** de Softmax | indices `long` `(N,)` |
| MSE | `nn.Softmax(dim=1)(model(x))` | `one_hot(y, 10).float()` `(N, 10)` |

> 💡 Pour **mettre en évidence le sur-apprentissage**, le TP n'utilise que 1 % du train (`TRAIN_RATIO = 0.01`) : l'écart train/test devient visible.

---

# III. Régulariser

## 8. Pénalisation L1/L2, Dropout, BatchNorm

| Technique | Mise en œuvre | Réglage / remarques |
|---|---|---|
| **L2 manuelle** | `l += l2_coef * sum((p**2).sum() for n, p in model.named_parameters() if n.endswith(".weight"))` | Tester `l2_coef` ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2}. Ne pénaliser que les **poids** (pas les biais). |
| **L2 via l'optimiseur** | `Adam(..., weight_decay=1e-4)` / `AdamW` | Équivalent pratique, sans code. |
| **L1** | `l += l1_coef * sum(p.abs().sum() ...)` | Rend des poids exactement nuls (parcimonie). |
| **`nn.Dropout(p)`** | après une couche (`Linear → Dropout → ReLU` dans le TP) | `p` = 0.1–0.5. Actif **seulement en `train()`** ; en `eval()` il est désactivé (et les activations sont déjà ré-échelonnées par $1/(1-p)$ à l'entraînement). |
| **`nn.BatchNorm1d(dim)`** | `Linear → BatchNorm1d → ReLU` | Normalise chaque feature avec les stats du **batch** en train, les **moyennes glissantes** en eval. Accélère et stabilise. Batch de taille 1 impossible en train. |
| `nn.LayerNorm(dim)` | 🔎 | Normalise chaque exemple : indépendant du batch (standard des Transformers). |

```python
def get_net(in_f, out_f, dims, dropout=0., batchnorm=False):
    layers, dim = [], in_f
    for newdim in dims:
        layers.append(nn.Linear(dim, newdim))
        if batchnorm: layers.append(nn.BatchNorm1d(newdim))
        if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        dim = newdim
    layers.append(nn.Linear(dim, out_f))
    return nn.Sequential(*layers)

net = get_net(784, 10, [100, 100, 100], dropout=0.2)
```

> 👉 Diagnostic : **loss train ↓ et loss test ↑** = sur-apprentissage → régulariser (dropout, L2, plus de données, réseau plus petit, early stopping).
> 💡 `writer.add_histogram(f'linear/{i}/weight', module.weight, epoch)` : visualiser l'effet de la L2 (poids resserrés autour de 0).

---

# IV. Outils

## 9. GPU / device

| Outil | Explication |
|---|---|
| `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Code agnostique. Mac : `"mps" if torch.backends.mps.is_available()`. |
| `model = model.to(device)` | Déplace **les paramètres** (renvoie le module). **Avant** de créer l'optimiseur. |
| `x, y = x.to(device), y.to(device)` | Pour **chaque batch** (`.to` n'est **pas** inplace sur un tenseur : réaffecter). |
| `t.device` | Device d'un tenseur ; pour un modèle : `next(model.parameters()).device`. |
| `model.to("cpu")` | Avant sauvegarde / analyse numpy. |

> ⚠️ Tous les tenseurs d'une opération doivent être sur le **même device** (`RuntimeError: Expected all tensors to be on the same device`).

---

## 10. Checkpointing (sauvegarde / chargement)

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `model.state_dict()` / `optim.state_dict()` | — | Dictionnaires des paramètres (et des états internes de l'optimiseur : moments d'Adam…). |
| `torch.save(obj, fichier)` | dict, chemin | Sérialise (pas besoin de pickle). |
| `torch.load(fichier, map_location=device, weights_only=True)` | `map_location` pour charger un modèle GPU sur CPU | Relit. 🔎 `weights_only=True` : chargement sûr (défaut dans les versions récentes). |
| `model.load_state_dict(state)` | dict | Recharge les poids dans un modèle **de même architecture**. |

```python
def save_state(epoch, model, optim, fichier):
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': optim.state_dict()}, fichier)

def load_state(fichier, model, optim):
    epoch = 0
    if os.path.isfile(fichier):
        state = torch.load(fichier)
        model.load_state_dict(state['model_state'])
        optim.load_state_dict(state['optim_state'])
        epoch = state['epoch']
    return epoch

start_epoch = load_state(fichier, net, optim)
for epoch in range(start_epoch, EPOCHS):          # reprise là où on s'était arrêté
    ...
    if epoch % 10 == 0: save_state(epoch, net, optim, fichier)
```

> 👉 Sauvegarder **modèle + optimiseur + epoch** pour reprendre un entraînement ; le modèle seul suffit pour l'inférence.
> ⚠️ `/tmp` n'existe pas sous Windows : `os.makedirs("./model", exist_ok=True)` ou `Path(...).mkdir(parents=True, exist_ok=True)`.

---

## 11. TensorBoard et tqdm

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `SummaryWriter(log_dir)` | un dossier **par run** (`f"logs/{model.name}-{time.asctime()}".replace(":","_")`) | Écrit les logs ; TensorBoard compare les runs de dossiers frères. |
| `writer.add_scalar(tag, valeur, global_step)` | `tag` hiérarchique `"loss/train"`, `"loss/test"` | Courbe d'une grandeur ; les tags partageant un préfixe sont groupés. |
| `writer.add_scalars(tag, {"train": a, "test": b}, step)` | dict | Plusieurs courbes sur un graphe. |
| `writer.add_histogram(tag, tenseur, step)` | poids d'une couche | Distribution des poids au fil des epochs. |
| Lancement | `tensorboard --logdir chemin/logs` dans un terminal, puis ouvrir l'URL ; ou `%load_ext tensorboard` + `%tensorboard --logdir logs` dans le notebook | |
| `tqdm(iterable)` | `from tqdm import tqdm` | Barre de progression. `loop.set_description(f"loss : {v:.4f}")` affiche la loss courante. |

> ⚠️ Sous Windows, les `:` de `time.asctime()` sont interdits dans les noms de dossiers → `.replace(":", "_")`.

---

## 12. Lambda et one-hot

| Outil | Exemple | Explication |
|---|---|---|
| `lambda args: expr` | `monpoly = lambda x: 2*x**2 + 4` | Fonction anonyme d'une ligne ; utile pour paramétrer une transformation (`transf_lab`, `transf_out`). |
| `torch.nn.functional.one_hot(t, num_classes)` | `one_hot(torch.tensor(2), 3)` → `[0, 0, 1]` | Entier(s) → vecteur(s) one-hot **entiers** (`long`) ; ajouter `.float()` pour une loss. ⚠️ Prend un **tenseur** `long`, pas un `int` Python. |

```python
to_one_hot = lambda x: one_hot(x, 3).float()
to_one_hot(torch.tensor([0, 2]))   # tensor([[1., 0., 0.], [0., 0., 1.]])
```

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Softmax avant `CrossEntropyLoss` | apprentissage lent / plafonné | donner les **scores bruts** |
| Labels float ou one-hot pour `CrossEntropyLoss` (classique) | erreur de type / de forme | indices `long` `(N,)` |
| `nn.Softmax()` sans `dim` | warning, normalisation sur le mauvais axe | `nn.Softmax(dim=1)` |
| Formes `(N,1)` vs `(N,)` dans la MSE | loss fausse (broadcast), warning | `.view(-1)` |
| Évaluation sans `no_grad` | mémoire / temps gaspillés | `with torch.no_grad():` |
| Oubli de `model.eval()` avec Dropout/BatchNorm | résultats de test bruités / faux | `eval()` avant test, `train()` avant apprentissage |
| Optimiseur créé avant `model.to(device)` | paramètres non mis à jour / erreurs | `.to(device)` puis optimiseur |
| Batch pas déplacé sur le device | `Expected all tensors to be on the same device` | `x.to(device)` à chaque batch |
| `shuffle=False` sur le train | apprentissage biaisé par l'ordre | `shuffle=True` (train uniquement) |
| Couches dans une liste Python | paramètres non entraînés | `nn.Sequential(*layers)` / `nn.ModuleList` |
| `cumloss += loss` (sans `.item()`) | graphe conservé, mémoire qui gonfle | `loss.item()` |
| Données non normalisées + SGD | loss gigantesque, divergence ou lr ridicule | normaliser (moyenne 0, écart-type 1) |
| `torch.load` d'un modèle GPU sans GPU | erreur CUDA | `map_location="cpu"` |
| Noms de fichiers avec `:` sous Windows | erreur de création de dossier | `.replace(":", "_")` |
| `one_hot(2, 3)` | `TypeError: must be Tensor, not int` | `one_hot(torch.tensor(2), 3)` |

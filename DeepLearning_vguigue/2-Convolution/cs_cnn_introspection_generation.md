# 🔍 Cheat Sheet PyTorch — Introspection des CNN & génération d'images

> Basée sur les notebooks `2_2-CNN-visu`, `2_3-CNN` (parties visualisation) et `2_4-CNN-deepdreams`.
> Convention : `import torch`, `from torch import nn`, `import torch.nn.functional as F`, `from torchvision import transforms`

## Sommaire

**I. Introspection (explication post-hoc)**
1. [Visualiser filtres et cartes d'activation](#1-visualiser-filtres-et-cartes-dactivation)
2. [Carte de saillance (gradient / pixels)](#2-carte-de-saillance-gradient--pixels)
3. [Class Activation Map (CAM)](#3-class-activation-map-cam)
4. [Choisir une méthode d'explication](#4-choisir-une-méthode-dexplication)

**II. Jouer avec le gradient sur l'image**

5. [Hooks : récupérer une couche interne](#5-hooks--récupérer-une-couche-interne)
6. [Optimiser l'image d'entrée (schéma général)](#6-optimiser-limage-dentrée-schéma-général)
7. [DeepDream](#7-deepdream)
8. [Reconstruction d'image depuis un embedding](#8-reconstruction-dimage-depuis-un-embedding)
9. [Transfert de style](#9-transfert-de-style)
10. [Pré/post-traitement des images](#10-prépost-traitement-des-images)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Introspection

## 1. Visualiser filtres et cartes d'activation

| Élément | Comment l'obtenir | Lecture |
|---|---|---|
| **Poids d'un filtre** | `conv.weight[j, 0]` (filtre `j`, canal d'entrée 0) → `(K, K)` | Détecteurs de motifs élémentaires (bords, orientations) ; peu lisibles au-delà de la 1re couche. |
| **Carte d'activation** (*feature map*) | exécuter `model.features` **module par module** et garder les sorties | Où le filtre réagit dans l'image ; la topologie est conservée. |
| **Sortie du pooling** | idem après `MaxPool2d` / `AvgPool2d` | Max : cartes contrastées ; Avg : cartes floues. |

```python
def analyse_conv(model, img, nb_filtres=16):
    x = img.unsqueeze(0).to(device)               # batch de 1
    img_conv, img_pool = [], []
    for m in model.features._modules.values():    # forward pas à pas
        x = m(x)
        if isinstance(m, nn.Conv2d):
            img_conv.append((x.squeeze(0), m.weight))
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            img_pool.append(x.squeeze(0))
    # puis plt.subplots(3*len(img_conv), nb_filtres) :
    # ligne 1 = w[j,0], ligne 2 = sortie conv[j], ligne 3 = sortie pool[j]
    # (.detach().cpu() avant np.array / imshow)
```

> 👉 Comparer un réseau à max-pooling et un à average-pooling sur la même image ; faire varier taille / nombre de filtres.

---

## 2. Carte de saillance (gradient / pixels)

Principe : gradient du score de la classe d'intérêt **par rapport aux pixels** ; un gradient fort = pixel influent.

$$M_{ij} = \max_{ch}\left|\frac{\partial S_c}{\partial I_{ch,i,j}}\right|$$

```python
def getSaliency(model, img, label):
    model.eval(); model.zero_grad()
    img = img.to(device).clone().requires_grad_(True)       # gradient SUR L'IMAGE
    probas = nn.Softmax(dim=1)(model(img.unsqueeze(0)))     # le softmax est dans la loss, pas dans le réseau
    probas[0, label].backward()                             # backward sur UNE classe
    sal = img.grad.abs()
    if sal.dim() > 2:
        sal = sal.max(dim=0)[0]                             # max sur les canaux (RGB)
    plt.imshow(sal.cpu(), cmap="seismic", interpolation="bilinear")
    return sal
```

| Paramètre / choix | Effet |
|---|---|
| `label` | la vraie classe (pourquoi ce label ?) ou une autre (qu'est-ce qui ressemble à un « chat » ?). |
| softmax vs score brut | 🔎 le score brut (logit) évite que le gradient dépende des autres classes ; le TP utilise la proba. |
| `.abs()` puis `max` canaux | on ne garde que l'intensité de l'influence. |

> ⚠️ Gradient **sur l'image** : il faut `requires_grad=True` sur l'entrée ; les poids du modèle peuvent être gelés.

---

## 3. Class Activation Map (CAM)

Principe : remplacer la tête dense par **Global Average Pooling + une couche linéaire** (filtres → classes), ré-entraîner **uniquement** cette tête, puis pondérer les cartes de la dernière couche convolutive par les poids de la classe.

$$\mathrm{CAM}_c(x, y) = \sum_k w^c_k\, f_k(x, y)$$

```python
# 1. Geler l'extracteur et remplacer la tête (MNIST : sortie features = 16×16×16)
set_parameter_requires_grad(convnet, False)
convnet.classifier = nn.Sequential(nn.Unflatten(1, (16, 16, 16)),   # remise en forme des cartes
                                   nn.AvgPool2d(16, 1),            # GAP : 1 valeur par filtre
                                   nn.Flatten(1),
                                   nn.Linear(16, 10))              # filtres → classes
train(convnet, 5, train_loader, test_loader)                       # seule la tête apprend

# 2. Calcul de la carte
def generate_cam(model, img, target_class=None):
    with torch.no_grad():
        feats = model.features(img.to(device))                     # (1, K, h, w)
        out = F.softmax(model(img.to(device)), -1)
    if target_class is None:
        target_class = out.argmax(-1).item()
    w = list(model.classifier._modules.values())[-1].weight.data   # (n_classes, K)
    cam = (feats * w[target_class].view(1, -1, 1, 1)).sum(1)       # somme pondérée des cartes
    cam = F.interpolate(cam.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    plt.imshow(cam.cpu().squeeze(), cmap="bwr", vmin=-4, vmax=4)
```

| Réseau | Remplacement de tête |
|---|---|
| ConvNet MNIST | `Unflatten(1,(16,16,16))` → `AvgPool2d(16,1)` → `Linear(16,10)` |
| ConvCIFAR(32) | `Unflatten(1,(32,20,20))` → `AvgPool2d(20,1)` → `Linear(32,10)` |
| AlexNet | `Unflatten(1,(256,6,6))` → `AvgPool2d(6,1)` → `Linear(256,10)` (AlexNet aplatit entre `features` et `classifier`) |

> 💡 🔎 Plus simple : `nn.AdaptiveAvgPool2d(1)` + `nn.Flatten()` + `nn.Linear(K, n_classes)` en lisant directement la sortie de `features`.
> 💡 Pour voir la carte sur l'image : `F.interpolate` (*upscaling* bilinéaire) jusqu'à la taille d'entrée.

---

## 4. Choisir une méthode d'explication

| Méthode | Besoin | Modifie le modèle ? | Coût | Ce qu'on voit |
|---|---|---|---|---|
| Filtres / activations | accès aux couches | non | 1 forward | ce que détectent les filtres |
| **Saillance** | gradient | non | 1 forward + 1 backward | pixels influents (bruité) |
| **CAM** | GAP + linéaire | **oui** (nouvelle tête à entraîner) | 1 forward | régions par classe (grossières) |
| Grad-CAM 🔎 | gradient sur la dernière conv | non | 1 forward + 1 backward | régions par classe, toute archi CNN |
| Occlusion | prédictions | non | 1 forward par position | zones décisives (lent) |
| LIME 🔎 (`pip install lime`) | prédictions | non | centaines de forwards | super-pixels explicatifs, modèle-agnostique |

---

# II. Jouer avec le gradient sur l'image

## 5. Hooks : récupérer une couche interne

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `dict(model.named_modules())[nom]` | nom de couche, ex. `'inception3a'`, `'inception4b'`, `'inception5a'` (GoogLeNet) | Récupérer un sous-module par son nom (`print(model)` pour les lister). |
| `module.register_forward_hook(fn)` | `fn(module, input, output)` | Appelée à **chaque forward** du module : on y stocke `output`. Renvoie un *handle*. |
| `handle.remove()` | — | **Retirer** le crochet après usage (sinon il s'accumule). |

```python
activations = {}
def hook_fn(module, inp, out):
    activations['feat'] = out                       # pas de .detach() : on veut le gradient
target = dict(model.named_modules())['inception4b']
hook = target.register_forward_hook(hook_fn)
model(img)                                          # remplit activations['feat']
hook.remove()
```

> 💡 Les hooks évitent de réécrire le `forward` d'un gros modèle pré-entraîné pour accéder à ses couches cachées.

---

## 6. Optimiser l'image d'entrée (schéma général)

On **gèle le modèle** (`model.eval()`) et on traite **l'image** comme le paramètre à optimiser :

```python
x = img.clone().requires_grad_(True)               # ou torch.randn_like(img)
optimizer = torch.optim.Adam([x], lr=0.1)           # l'optimiseur reçoit L'IMAGE
for i in range(n_iter):
    optimizer.zero_grad()
    model(x)                                        # le hook capture les activations
    loss = critere(activations['feat'])
    loss.backward()
    optimizer.step()
    x.data = torch.clamp(x.data, -2, 2)             # garder des pixels plausibles (espace normalisé)
```

| Tâche | Image de départ | Loss (sur la couche cible) |
|---|---|---|
| DeepDream | l'image réelle | `-activations.norm()` (maximiser) |
| Reconstruction | bruit | `‖cible − activations‖` |
| Transfert de style | bruit (ou l'image de contenu) | `α·L_content + β·L_style` |

> 🔎 Pour ne pas calculer de gradients inutiles sur les poids : `for p in model.parameters(): p.requires_grad_(False)`.

---

## 7. DeepDream

**Ascension de gradient** : on modifie l'image pour **amplifier** ce qu'une couche détecte déjà.

```python
img.requires_grad_(True)
optimizer = torch.optim.Adam([img], lr=0.1)
for i in range(30):
    optimizer.zero_grad()
    model(img)
    loss = -activations['feat'].norm()      # MOINS : l'optimiseur minimise, on veut maximiser
    loss.backward(); optimizer.step()
    img.data = torch.clamp(img.data, -2, 2)
```

| Réglage | Effet |
|---|---|
| couche peu profonde (`inception3a`) | motifs simples (traits, textures) |
| couche profonde (`inception4d`, `5a`) | motifs complexes (yeux, têtes d'animaux) |
| `lr`, nb d'itérations | intensité de l'effet « psychédélique » |

---

## 8. Reconstruction d'image depuis un embedding

Question : la représentation d'une couche suffit-elle à reconstruire l'image ?

```python
with torch.no_grad():
    model(img); target = activations['feat'].clone()       # encodage de référence (figé)
noise = torch.randn_like(img).requires_grad_(True)          # départ : bruit blanc gaussien
optimizer = torch.optim.AdamW([noise], lr=5e-1)
scheduler = OneCycleLR(optimizer, max_lr=5e-1, steps_per_epoch=1, epochs=1000)
for i in range(1000):
    optimizer.zero_grad()
    model(noise)
    loss = (target - activations['feat']).norm()           # rapprocher les embeddings
    loss.backward(); optimizer.step(); scheduler.step()
    noise.data = torch.clamp(noise.data, -2, 2)
```

> 💡 Couche peu profonde (`inception3a`) → reconstruction fidèle (bon encodage du **contenu**) ; couche profonde → seule la sémantique est gardée, les détails se perdent.

---

## 9. Transfert de style

Deux hooks : une couche pour le **contenu** (`inception3a`), une (ou plusieurs) pour le **style** (`inception5a` dans le TP ; 🔎 Gatys utilise plusieurs couches).

$$G_{ij} = \sum_k F_{ik}F_{jk}, \qquad \mathcal L = \alpha\,\lVert F^{c}(x) - F^{c}(\text{contenu})\rVert^2 + \beta\,\lVert G(F^{s}(x)) - G(F^{s}(\text{style}))\rVert^2$$

- $F$ : activations aplaties `(C, H·W)` ; $G$ : matrice de Gram `(C, C)` ; $\alpha, \beta$ : poids contenu/style.

```python
acts = {}
def save(name):
    def fn(m, i, o): acts[name] = o
    return fn
mods = dict(model.named_modules())
h1 = mods['inception3a'].register_forward_hook(save('content'))
h2 = mods['inception5a'].register_forward_hook(save('style'))

def gram(f):                                   # f : (1, C, H, W)
    _, C, H, W = f.shape
    f = f.view(C, H * W)
    return f @ f.T / (C * H * W)               # normalisation pour équilibrer les échelles

with torch.no_grad():
    model(img);   target_content = acts['content'].clone()
    model(style); target_gram    = gram(acts['style']).clone()

x = img.clone().requires_grad_(True)          # partir du contenu converge plus vite que du bruit
optimizer = torch.optim.Adam([x], lr=0.05)
alpha, beta = 1.0, 1e3                         # à régler : β grand = plus de style
for i in range(500):
    optimizer.zero_grad()
    model(x)
    loss = alpha * (acts['content'] - target_content).pow(2).mean() \
         + beta  * (gram(acts['style']) - target_gram).pow(2).sum()
    loss.backward(); optimizer.step()
    x.data.clamp_(-2, 2)
h1.remove(); h2.remove()
```

> 👉 Si le résultat ressemble trop au contenu : augmenter `beta` ; s'il est illisible : baisser `beta` ou le `lr`. Les images de style et de contenu n'ont pas besoin d'avoir la même taille (la Gram est `(C, C)`).

---

## 10. Pré/post-traitement des images

| Étape | Code | Remarque |
|---|---|---|
| Charger | `Image.open(path).convert("RGB")` | `convert` évite les images RGBA / niveaux de gris. |
| Prétraiter (ImageNet) | `Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])` puis `.unsqueeze(0)` | Même normalisation que l'entraînement du modèle. |
| Classer | `F.softmax(model(x)[0], dim=0)` puis `torch.topk(probas, 5)` | Labels ImageNet dans `imagenet_classes.txt` (une classe par ligne). |
| Dé-normaliser | `t*std.view(3,1,1) + mean.view(3,1,1)` puis `clamp(0, 1)` | Indispensable avant affichage. |
| Vers PIL | `transforms.ToPILImage()(t.detach().cpu().squeeze(0))` | Puis `.crop((50, 50, 200, 200))` pour zoomer, `.save(...)`. |
| Copier une image | `copy.deepcopy(img)` / `img.clone()` | Garder l'original avant de l'optimiser. |

```python
def deprocess(t):
    t = t.detach().cpu().squeeze(0)
    t = t * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return transforms.ToPILImage()(t.clamp(0, 1))
```

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Oublier `requires_grad` sur l'image | `img.grad` vaut `None` | `img.requires_grad_(True)` (sur une copie : `clone()`) |
| DeepDream sans le signe `−` | l'image « s'éteint » au lieu de s'amplifier | `loss = -activations.norm()` |
| Passer les poids du modèle à l'optimiseur | on modifie le réseau, pas l'image | `optim.Adam([img])` |
| Hook non retiré | activations écrasées / fuite mémoire, hooks multiples | `hook.remove()` |
| `.detach()` dans le hook | plus de gradient vers l'image | stocker `output` tel quel (détacher seulement la **cible**) |
| Cible calculée avec gradient | graphe conservé inutilement | calculer la cible sous `no_grad()` + `.clone()` |
| Mauvaise forme dans `Unflatten` de la CAM | erreur de taille | utiliser la forme réelle de `model.features(x)` |
| Tête CAM non ré-entraînée / extracteur non gelé | carte sans sens / features modifiées | geler, remplacer, entraîner la tête |
| Affichage sans dé-normaliser | couleurs absurdes | `deprocess` |
| `plt.imshow` d'un tenseur avec gradient / sur GPU | erreur | `.detach().cpu()` |
| Pixels qui divergent | image saturée | `clamp` dans l'espace normalisé (≈ [−2, 2]) |
| Modèle en mode `train()` | Dropout/BatchNorm perturbent l'explication | `model.eval()` |
| `iter(loader).next()` | `AttributeError` (Python 3) | `next(iter(loader))` |

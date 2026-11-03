# 🖼️ Cheat Sheet CNN en PyTorch — Convolutions, introspection, augmentation, fine-tuning

> Basée sur les notebooks `2_0-CNN-tuto`, `2_1-CNN` (MNIST), `2_2-CNN-visu` (saillance, CAM), `2_3-CNN-finetuning` (CIFAR-10, data augmentation, AlexNet pré-entraîné).
> Conventions : `import torch`, `from torch import nn`, `import torchvision`, `from torchvision import transforms, models`.
> Prérequis (boucle d'entraînement, `DataLoader`, `eval/no_grad`…) : voir `cs_pytorch_fondamentaux.md`. Extraction de features pré-entraînées : `cs_pytorch.md`.

## Sommaire
1. [Format des tenseurs image et chargement des données](#1-format-des-tenseurs-image-et-chargement-des-données)
2. [Convolution : `Conv2d` et paramètres](#2-convolution--conv2d-et-paramètres)
3. [Pooling](#3-pooling)
4. [Calculer les tailles de sortie](#4-calculer-les-tailles-de-sortie)
5. [Architecture type `features` / `classifier`](#5-architecture-type-features--classifier)
6. [Entraînement et comptage des paramètres](#6-entraînement-et-comptage-des-paramètres)
7. [Introspection : filtres, cartes d'activation](#7-introspection--filtres-cartes-dactivation)
8. [Saliency map](#8-saliency-map)
9. [Class Activation Map (CAM)](#9-class-activation-map-cam)
10. [Data augmentation](#10-data-augmentation)
11. [Modèles pré-entraînés : transfert et fine-tuning](#11-modèles-pré-entraînés--transfert-et-fine-tuning)
12. [📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Format des tenseurs image et chargement des données

Convention PyTorch : **`(N, C, H, W)`** = (batch, canaux, hauteur, largeur). Une image seule : `(C, H, W)` ; il faut `unsqueeze(0)` pour la passer dans un réseau. Pour `plt.imshow` : `img.permute(1, 2, 0)` (→ `H, W, C`) ou `img.squeeze(0)` pour 1 canal.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torchvision.datasets.MNIST(root, train, transform, download)` | `root` = dossier, `train=True/False`, `transform`, `download=True` | Chiffres manuscrits 28×28, 1 canal, 10 classes (60 000 train / 10 000 test). |
| `torchvision.datasets.CIFAR10(root, train, transform, download)` | idem | 60 000 images couleur 32×32×3, 10 classes (50 000 train / 10 000 test). Attribut `.classes` = noms des classes. |
| `transforms.ToTensor()` | — | Image PIL/ndarray `(H,W,C)` en `[0,255]` → tenseur `(C,H,W)` en `[0,1]` (`float32`). |
| `transforms.Normalize(mean, std)` | listes de longueur `C` | `(x - mean) / std` par canal. À appliquer **après** `ToTensor`. |
| `transforms.Compose([...])` | liste de transformations | Enchaîne les prétraitements. |
| `DataLoader(ds, batch_size, shuffle, pin_memory=True)` | — | `pin_memory=True` accélère le transfert CPU → GPU. |

```python
tf = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10('/tmp/data', train=True,  download=True, transform=tf)
test_set  = torchvision.datasets.CIFAR10('/tmp/data', train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, pin_memory=True)
```

⚠️ Le label renvoyé par `dataset[i]` est un `int` Python ; le `DataLoader` le transforme en tenseur `int64` — c'est ce que `CrossEntropyLoss` attend.

---

## 2. Convolution : `Conv2d` et paramètres

Chaque filtre (noyau) $K\in\mathbb R^{C_{in}\times k\times k}$ glisse sur l'image et calcule une somme pondérée locale ; **les poids sont partagés** sur toutes les positions (peu de paramètres, invariance par translation). `out_channels` filtres → `out_channels` cartes de caractéristiques (*feature maps*).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')` | `in_channels` = canaux d'entrée (1 pour MNIST, 3 RGB) ; `out_channels` = nombre de filtres ; `kernel_size` = taille du noyau (int ou tuple) ; `stride` = pas de déplacement ; `padding` = bord de zéros ajouté ; `dilation` = écart entre les cases du noyau (filtre « troué ») | Poids de forme `(out_channels, in_channels, k, k)`, biais de forme `(out_channels,)`. Sortie `(N, out_channels, H', W')`. |
| `nn.Conv1d(in_channels, out_channels, kernel_size, …)` | idem, entrée `(N, C, L)` | Convolution sur séquences (texte : `C` = dimension d'embedding). Voir `cs_representations_reco_texte.md`. |
| `nn.init.xavier_uniform_(conv.weight)` | — | Initialisation de Glorot ; évite des gradients trop faibles/forts au départ. |

**Effet des paramètres** (exemple du TD `2_0` : image 5×5 valant 0…24, noyau 2×2 `[[1,2],[1,1]]`) :

| Paramètre | Effet | Quand l'utiliser |
|---|---|---|
| `padding=p` | Ajoute `p` pixels de zéros sur chaque bord → la sortie rétrécit moins. `padding=(k-1)//2` (k impair, stride 1) garde la taille (*same*). | Garder la résolution, ne pas perdre les bords. |
| `stride=s` | Déplace la fenêtre de `s` pixels → sortie ≈ divisée par `s`. | Sous-échantillonner à la place d'un pooling. |
| `dilation=d` | Le noyau couvre une zone plus large sans plus de paramètres (taille effective `d(k-1)+1`). | Augmenter le champ réceptif à coût constant. |
| `bias=True` | Un biais par filtre. | **Toujours le laisser** (le TD le retire uniquement pour vérifier un calcul à la main). |

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
conv.weight = nn.Parameter(torch.tensor([[[[1., 2.], [1., 1.]]]]))   # (out, in, k, k)
out = conv(x.unsqueeze(0))        # x : (5,5) -> (1,5,5) = (C,H,W) ; Conv2d accepte aussi une entrée sans dimension batch
# 1re sortie = (x[:2,:2] * poids).sum()  -> calcul manuel pour vérifier
```

💡 **Empilement** : les premières couches détectent des motifs simples (bords, coins), les suivantes les combinent en motifs plus complexes ; le **champ réceptif** grandit avec la profondeur, le `stride` et le pooling.

⚠️ `Conv2d` attend un **float** ; entrée sans dimension canal ou batch → erreur de dimensions. Ne pas confondre `in_channels` (profondeur de l'entrée) avec la taille de l'image.

---

## 3. Pooling

Résume une petite fenêtre par une valeur : réduit la résolution, apporte de l'invariance locale à la translation, ajoute **0 paramètre**.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `nn.MaxPool2d(kernel_size, stride=None, padding=0)` | `kernel_size` ; **`stride` vaut `kernel_size` par défaut** | Garde le max de la fenêtre : conserve la détection la plus forte (« le motif est-il présent ? »). Standard. |
| `nn.AvgPool2d(kernel_size, stride=None, padding=0)` | idem | Moyenne de la fenêtre : lisse, garde l'intensité moyenne. |
| `nn.AdaptiveAvgPool2d(output_size)` | taille de sortie fixe (ex. `1` ou `(6,6)`) | Pooling dont la taille de sortie est imposée quelle que soit l'entrée (permet des images de tailles variables). Avec `1` = *Global Average Pooling* (GAP). |
| `nn.MaxPool1d(kernel_size, stride)` | idem | Version séquences. |

⚠️ **Stride par défaut = `kernel_size`** : `MaxPool2d(3)` sous-échantillonne par 3, alors que `MaxPool2d(3, 1)` (comme dans le TD) glisse d'un pixel et ne réduit que de `k-1`. Préciser `stride` explicitement.

👉 Le TD compare MaxPool et AvgPool sur MNIST : max-pooling → cartes plus contrastées ; avg-pooling → cartes plus floues.

---

## 4. Calculer les tailles de sortie

Pour une dimension (hauteur ou largeur) d'entrée $n$, noyau $k$, padding $p$, stride $s$, dilatation $d$ :

$$n_{out}=\left\lfloor \frac{n+2p-d\,(k-1)-1}{s}\right\rfloor+1$$

Cas courant ($d=1$) : $n_{out}=\left\lfloor\dfrac{n+2p-k}{s}\right\rfloor+1$. Le pooling suit la même formule avec son `kernel_size`/`stride`.

- $n$ : taille d'entrée ; $k$ : taille du noyau ; $p$ : padding ; $s$ : stride ; $d$ : dilatation.

**Exemples du cours** (stride 1, pas de padding) :

| Réseau | Calcul | Sortie |
|---|---|---|
| MNIST 28×28 : Conv5 → MaxPool3 → Conv5 → MaxPool3 | 28 → 24 → 22 → 18 → 16 | `16 filtres × 16 × 16` = **4096** (`Linear(16*16*16, 120)`) |
| CIFAR 32×32 : même architecture | 32 → 28 → 26 → 22 → 20 | `nb_channels × 20 × 20` (`Linear(20*20*nb_channels, 120)`) |
| AlexNet (entrée 224) | — | `256 × 6 × 6` avant le classifieur |

💡 **Astuce pour ne pas calculer à la main** : passer un tenseur factice dans `features` et lire sa forme :
```python
with torch.no_grad():
    dim = model.features(torch.zeros(1, 3, 32, 32)).numel()    # taille aplatie à donner au 1er Linear
```

---

## 5. Architecture type `features` / `classifier`

Convention du cours : un sous-réseau **convolutif** `self.features` (extraction) + un sous-réseau **dense** `self.classifier` (décision). Le raccord se fait en **aplatissant** les cartes.

```python
class ConvNet(nn.Module):
    def __init__(self, pooling=nn.MaxPool2d, nb_channels=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, nb_channels, kernel_size=5), nn.ReLU(), pooling(3, 1),
            nn.Conv2d(nb_channels, nb_channels, kernel_size=5), nn.ReLU(), pooling(3, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(nb_channels*16*16, 120), nn.ReLU(), nn.Linear(120, 10)   # 10 logits
        )
        for m in self.features:                       # init Xavier des convolutions
            if isinstance(m, nn.Conv2d): nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))    # (N, C, H, W) -> (N, C*H*W)
```

- Ordre standard d'un bloc : **`Conv → (BatchNorm) → ReLU → Pool`**.
- Sortie = **logits** (pas de softmax) → `nn.CrossEntropyLoss`.
- 🔎 Pratique moderne : `nn.Flatten()` dans le `Sequential`, `padding='same'` + petits noyaux 3×3, `BatchNorm2d(n)` après chaque convolution, `Dropout` dans le classifieur.

---

## 6. Entraînement et comptage des paramètres

Boucle identique à `cs_pytorch_fondamentaux.md §8` (Adam `lr=1e-3`, `CrossEntropyLoss`, `model.train()` / `model.eval()` + `no_grad`). Sauvegarde : `torch.save({'model_state': model.state_dict()}, fichier)` puis, pour recharger, **recréer le même modèle** et `load_state_dict`.

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

| Paramètres d'une couche | Formule |
|---|---|
| `Conv2d(C_in, C_out, k)` | $C_{out}(C_{in}k^2+1)$ (le $+1$ = biais) |
| `Linear(n_in, n_out)` | $n_{out}(n_{in}+1)$ |
| Pooling, ReLU, Flatten | 0 |

💡 Comparaison du TD sur CIFAR-10 : CNN 2 conv. (32 filtres) ≈ **66 %** d'accuracy vs MLP dense `3072 → 500 → 100 → 10` ≈ **51 %**, pour bien moins de paramètres côté convolution : le **partage de poids** exploite la structure spatiale.

⚠️ Dans la boucle, `y` doit être `int64` de forme `(N,)`. L'accuracy du TD fait `argmax(1).view(N,-1) == y.view(-1,1)` — vérifier que les formes coïncident (sinon broadcasting silencieux).

⚠️ Sur Mac M1 (`mps`), `argmax` a posé problème dans le TD : utiliser `torch.max(yhat, 1).indices`.

---

## 7. Introspection : filtres, cartes d'activation

Principe : exécuter `features` **module par module** et stocker les sorties pour les visualiser.

```python
def analyse_conv(model, img, nb_filtres=16):
    x = img.unsqueeze(0).to(device)
    for m in model.features:                       # parcours des modules
        x = m(x)
        if isinstance(m, nn.Conv2d):               # poids + sortie de la conv
            poids = m.weight                       # (C_out, C_in, k, k)  -> imshow(poids[j,0])
            sortie = x.squeeze(0)                  # (C_out, H', W')      -> imshow(sortie[j])
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            sortie_pool = x.squeeze(0)
```

- Afficher : `plt.imshow(t.detach().cpu().numpy(), cmap="gray")` — `detach()` obligatoire car les poids/sorties ont `requires_grad`.
- Lecture : les **poids** montrent ce que cherche chaque filtre (contours orientés…) ; la **sortie** montre où le filtre réagit ; la sortie du **pooling** montre la version réduite.

| Alternative | Explication |
|---|---|
| `module.register_forward_hook(fn)` | 🔎 Capture automatique des sorties d'une couche pendant `model(x)` (pratique pour les réseaux non séquentiels). |

---

## 8. Saliency map

**But** : savoir quels **pixels d'entrée** influencent le plus la décision pour une classe. On utilise le **gradient du score de la classe par rapport à l'image** : un gradient fort ⇒ changer légèrement ce pixel change beaucoup la sortie.

$$S(i,j)=\max_{c}\left|\frac{\partial\,p_{y}(x)}{\partial x_{c,i,j}}\right|$$

- $p_y(x)$ : probabilité (softmax) de la classe $y$ d'intérêt ; $x_{c,i,j}$ : pixel du canal $c$ en $(i,j)$ ; max sur les canaux.

```python
def get_saliency(model, img, label):
    model.eval(); model.zero_grad()
    img = img.clone().to(device).requires_grad_(True)       # gradient sur l'IMAGE
    probs = torch.softmax(model(img.unsqueeze(0)), dim=1)   # softmax remis à la main
    probs[0, label].backward()                              # rétropropagation d'une seule classe
    sal = img.grad.abs()                                    # |gradient|
    return sal.max(dim=0)[0] if sal.dim() > 2 else sal      # max sur les canaux
# affichage : plt.imshow(sal.cpu(), cmap="seismic", interpolation="bilinear")
```

⚠️ Le softmax n'est **pas** dans le réseau (il est dans la loss) → il faut le réappliquer ici, sinon on dérive le logit brut.

⚠️ `img.requires_grad = True` impose que `img` soit une **feuille** de type float ; passer le tenseur sur le device **avant** d'activer le gradient (ou utiliser `clone().to(device).requires_grad_(True)`).

💡 Limite : la saillance brute est souvent bruitée ; elle indique la sensibilité locale, pas une explication « humaine ».

---

## 9. Class Activation Map (CAM)

**But** : voir **quelles régions** de l'image ont fait réagir les filtres utiles à la classe.

**Modification du réseau** : remplacer le classifieur par *Global Average Pooling → un seul `Linear(nb_filtres, nb_classes)`*, puis ré-entraîner **uniquement** cette partie (features gelées).

$$\mathrm{CAM}_c(i,j)=\sum_{k} w^{c}_{k}\,A_k(i,j)$$

- $A_k(i,j)$ : sortie du filtre $k$ de la dernière couche conv. en $(i,j)$ ; $w^c_k$ : poids du `Linear` reliant le filtre $k$ à la classe $c$.

```python
def set_requires_grad(model, b):
    for p in model.parameters(): p.requires_grad = b

set_requires_grad(convnet, False)                          # on gèle features
convnet.classifier = nn.Sequential(                        # nouvelles couches : requires_grad=True par défaut
    nn.Unflatten(1, (16, 16, 16)),                         # (N, 4096) -> (N, 16, 16, 16)
    nn.AvgPool2d(16, 1),                                   # GAP : (N,16,1,1)
    nn.Flatten(1),                                         # (N,16)
    nn.Linear(16, 10))
train(convnet, 10, train_loader, test_loader)

def generate_cam(model, img, target=None):
    with torch.no_grad():
        feats = model.features(img)                        # (1, K, h, w)
        if target is None: target = model(img).argmax(1).item()
        w = list(model.classifier.children())[-1].weight   # (n_classes, K)
        cam = (feats * w[target].view(1, -1, 1, 1)).sum(1) # somme pondérée -> (1, h, w)
    return cam  # puis F.interpolate(cam.unsqueeze(0), size=(H, W), mode="bilinear") pour superposer
```

⚠️ `nn.Unflatten` est nécessaire parce que l'ancien `forward` aplatit les cartes avant le classifieur (`x.view(N,-1)`) : il faut redonner la forme `(C,H,W)` pour faire le GAP. Pour AlexNet : `Unflatten(1,(256,6,6))` + `AvgPool2d(6,1)`.

⚠️ Le `forward` ré-aplatit : ne pas appliquer deux fois `view`. Le GAP sans précision de taille → utiliser `AdaptiveAvgPool2d(1)` pour être indépendant de la taille.

💡 Compromis : CAM sacrifie un peu de performance (classifieur linéaire seulement) pour gagner en interprétabilité spatiale. Différence avec la saillance : la CAM est **grossière mais porte sur les régions**, la saillance est **fine mais bruitée** (par pixel).

---

## 10. Data augmentation

Générer des **variantes** des images d'entraînement pour apprendre des invariances (retournement, échelle, luminosité…) et réduire le sur-apprentissage.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `transforms.RandomHorizontalFlip(p=0.5)` | `p` | Miroir horizontal aléatoire. Peu pertinent pour des chiffres/texte, très utile pour animaux/véhicules. |
| `transforms.RandomResizedCrop(size, scale=(0.08,1.0), ratio=(3/4,4/3))` | `size` = taille de sortie (32 pour CIFAR), `scale` = fraction de surface conservée | Recadrage aléatoire redimensionné. ⚠️ `scale` par défaut très agressif (jusqu'à 8 % de l'image) : monter `scale=(0.6,1)` pour de petites images. |
| `transforms.RandomCrop(size, padding=4)` | taille, padding | Alternative douce sur CIFAR. |
| `transforms.RandomRotation(degrees)` | angle max | Rotation aléatoire. |
| `transforms.ColorJitter(brightness, contrast, saturation, hue)` | amplitudes | Variations de couleur/luminosité. |

```python
tf_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                               transforms.RandomResizedCrop(32),
                               transforms.ToTensor()])          # augmentations AVANT ToTensor (opèrent sur PIL)
tf_test  = transforms.Compose([transforms.ToTensor()])          # aucune augmentation aléatoire en test
```

⚠️ **Augmenter uniquement le jeu d'entraînement**, jamais test/validation (sinon évaluation bruitée). Les transformations aléatoires sont recalculées à chaque epoch → il faut **plus d'epochs** pour en profiter (le TD passe à 30).

⚠️ Les augmentations qui travaillent sur images PIL se placent **avant** `ToTensor()` ; `Normalize` se place **après**.

---

## 11. Modèles pré-entraînés : transfert et fine-tuning

Un réseau entraîné sur ImageNet (1000 classes) a appris des features générales : on le réutilise en adaptant sa dernière couche.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `models.alexnet(weights=…)` (idem `resnet18`, `vgg16`, `mobilenet_v2`…) | `weights=models.AlexNet_Weights.DEFAULT` (ou `None` pour un réseau vierge) | Charge l'architecture + poids ImageNet. ⚠️ `pretrained=True` est déprécié ; `weights=True` du TD ne fonctionne pas dans les versions récentes. |
| `print(model)` | — | Affiche l'arborescence (`features`, `avgpool`, `classifier`) pour repérer **quelle couche modifier**. |
| `model.classifier[6] = nn.Linear(in_f, n_classes)` | `in_f = model.classifier[6].in_features` | Remplace la dernière couche (1000 → 10). Les nouvelles couches sont `requires_grad=True` par défaut. |
| `p.requires_grad = False` | — | **Gèle** des paramètres (pas de mise à jour). |

Deux stratégies :

| Stratégie | Que fait-on | Quand |
|---|---|---|
| **Extraction de features** (*feature extraction*) | Tout geler sauf la nouvelle couche de sortie | Peu de données, peu de calcul, domaine proche d'ImageNet. |
| **Fine-tuning complet** | Dégeler tout (ou les dernières couches) avec un **petit `lr`** (1e-4…1e-5) | Plus de données / domaine différent. |

```python
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
for p in alexnet.parameters(): p.requires_grad = False          # gel
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)   # nouvelle sortie (entraînable)

# Prétraitement identique à l'entraînement d'origine : redimensionner + normaliser ImageNet
tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, alexnet.parameters()), lr=1e-3)
```

Résultat du TD : AlexNet fine-tuné sur CIFAR-10 ≈ **83 %** (contre ≈ 66 % pour le petit CNN appris de zéro).

Tester sur des images du web : `Image.open(f).resize((32,32), Image.Resampling.LANCZOS)` → `tf(img).unsqueeze(0).to(device)` → `model(x).argmax(1)` ; comparer avec `dataset.classes`.

⚠️ **Redimensionner (224) + normaliser avec les statistiques d'ImageNet** est obligatoire : sinon les features n'ont plus de sens. Un réseau pré-entraîné reçoit des images 224×224 ; upscaler du 32×32 coûte cher en temps (d'où des entraînements longs).

⚠️ Avec des couches gelées, mettre quand même `model.eval()` en inférence ; avec BatchNorm gelée, penser à garder ces couches en mode `eval()` pendant le fine-tuning si le batch est petit.

⚠️ Après avoir remplacé le classifieur par un « GAP + Linear » (pour la CAM), les anciennes couches denses sont **perdues** : recharger le modèle pré-entraîné pour revenir en arrière.

👉 Réflexe courant : tout geler → entraîner la tête quelques epochs → dégeler progressivement avec un `lr` faible.

---

## 📌 Récapitulatif des pièges classiques

| # | Piège | Réflexe |
|---|---|---|
| 1 | Format `(N,H,W,C)` (numpy/PIL) au lieu de `(N,C,H,W)` | `permute(0,3,1,2)` ; `ToTensor` le fait pour les images PIL. |
| 2 | Dimension d'entrée du 1ᵉʳ `Linear` mal calculée | Formule $\lfloor (n+2p-k)/s\rfloor+1$ ou test avec un tenseur factice. |
| 3 | `MaxPool2d(k)` : le stride vaut `k` par défaut | Toujours préciser `stride`. |
| 4 | `Softmax` dans le réseau + `CrossEntropyLoss` | Logits en sortie ; softmax uniquement pour l'analyse (saillance). |
| 5 | Saliency : oublier `requires_grad` sur l'image ou le softmax | `img.requires_grad_(True)` + softmax manuel + `backward` sur **une** classe. |
| 6 | CAM sans `Unflatten` / sans geler `features` | `Unflatten` → GAP → `Linear` ; `requires_grad=False` sur les features. |
| 7 | Augmentation appliquée au test | Deux `Compose` : un pour le train, un pour le test. |
| 8 | `RandomResizedCrop` agressif sur de petites images | Réduire la plage `scale`. |
| 9 | Transfert : taille / normalisation différentes de l'entraînement d'origine | `Resize(224)` + `Normalize` ImageNet. |
| 10 | `weights=True` / `pretrained=True` (obsolètes) | `weights=models.<Modele>_Weights.DEFAULT`. |
| 11 | Optimiseur qui reçoit des paramètres gelés | Passer `filter(lambda p: p.requires_grad, …)`, créer l'optimiseur après le gel. |
| 12 | Afficher un tenseur avec gradient (`imshow`) | `.detach().cpu().numpy()`. |
| 13 | Comparer des modèles sans tenir compte du nombre de paramètres | Afficher `count_parameters` pour chaque architecture. |
| 14 | `argmax` instable sur `mps` (M1) | `torch.max(x, 1).indices`. |

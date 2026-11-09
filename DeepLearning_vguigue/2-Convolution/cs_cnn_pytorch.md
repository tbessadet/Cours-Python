# 🖼️ Cheat Sheet PyTorch — CNN, données image & fine-tuning

> Basée sur les notebooks `2_0-CNN-tuto`, `2_1-CNN`, `2_3-CNN` (et la classe `models/model.py`).
> Convention : `import torch`, `from torch import nn`, `import torchvision`, `from torchvision import transforms, models`

## Sommaire

1. [Couches de convolution](#1-couches-de-convolution)
2. [Couches de pooling](#2-couches-de-pooling)
3. [Calculer les dimensions](#3-calculer-les-dimensions)
4. [Construire un CNN : `features` + `classifier`](#4-construire-un-cnn--features--classifier)
5. [Inspecter un modèle](#5-inspecter-un-modèle)
6. [Données image : torchvision](#6-données-image--torchvision)
7. [Data augmentation](#7-data-augmentation)
8. [Boucle d'entraînement CNN (AdamW + OneCycleLR)](#8-boucle-dentraînement-cnn-adamw--onecyclelr)
9. [Modèles pré-entraînés & fine-tuning](#9-modèles-pré-entraînés--fine-tuning)
10. [Prédire sur une image externe](#10-prédire-sur-une-image-externe)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Couches de convolution

| Module | Paramètres essentiels | Explication |
|---|---|---|
| `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros')` | `in_channels` : 1 (gris), 3 (RGB) ou nb de filtres précédent ; `out_channels` : **nb de filtres** ; `kernel_size` : 3, 5, `(3,5)` ; `padding` : entier ou `'same'` (🔎 garde la taille si stride 1) | Entrée `(N, C_in, H, W)` → sortie `(N, C_out, H', W')`. Poids `.weight` de forme `(C_out, C_in, K, K)`. **Toujours garder le biais** en pratique (le TP le retire seulement pour vérifier les calculs à la main). |
| `nn.Conv1d(...)` | mêmes paramètres | Signaux 1D / séquences : entrée `(N, C, L)`. |
| `nn.init.xavier_uniform_(conv.weight)` | tenseur de poids | Initialisation (Glorot) pour éviter les problèmes de gradient. 🔎 `nn.init.kaiming_normal_` plutôt avec ReLU. |
| `conv.weight = nn.Parameter(t)` | tenseur de la bonne forme | Fixer les poids à la main (tests, filtres connus). |

```python
x = torch.arange(25.).view(5, 5)
conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
conv.weight = nn.Parameter(torch.tensor([[[[1., 2.], [1., 1.]]]]))   # (C_out, C_in, K, K)
conv(x.unsqueeze(0))        # (1, 4, 4) ; 1re case = 0*1 + 1*2 + 5*1 + 6*1 = 13
```

| Paramètre | Effet sur l'exemple 5×5, noyau 2 |
|---|---|
| défaut | 4×4 |
| `padding=1` | 6×6 (une bordure de zéros tout autour) |
| `stride=2` | 2×2 (la fenêtre saute d'une case sur deux) |
| `dilation=2` | 3×3 (filtre « à trous » : champ 3×3 avec 4 poids) |

> ⚠️ `Conv2d` accepte `(C, H, W)` (non batché) ou `(N, C, H, W)`. Une image MNIST seule `(28, 28)` doit devenir `(1, 28, 28)` (`unsqueeze(0)`) ; un batch du DataLoader est déjà `(N, 1, 28, 28)`.

---

## 2. Couches de pooling

| Module | Paramètres essentiels | Explication |
|---|---|---|
| `nn.MaxPool2d(kernel_size, stride=None, padding=0)` | ⚠️ **`stride` vaut `kernel_size` par défaut** | Max par fenêtre ; **pas de paramètres** ; canal par canal (pas de `in/out_channels`). |
| `nn.AvgPool2d(kernel_size, stride=None)` | idem | Moyenne par fenêtre (plus « lisse » que le max). |
| `nn.AdaptiveAvgPool2d(output_size)` | 🔎 `1` → `(N, C, 1, 1)` | Taille de sortie fixée quelle que soit l'entrée : *Global Average Pooling* (GAP) avec `1`. |
| `nn.Flatten(start_dim=1)` / `nn.Unflatten(1, (C, H, W))` | | Passer de cartes `(N,C,H,W)` à un vecteur `(N, C·H·W)` et inversement, **dans** un `Sequential`. |

```python
nn.MaxPool2d(2)(x.unsqueeze(0))   # [[6, 8], [16, 18]]  (stride=2 implicite)
nn.AvgPool2d(2)(x.unsqueeze(0))   # [[3, 5], [13, 15]]
```

> 👉 Usage courant : `MaxPool2d(2)` (divise H et W par 2). Le TP utilise `kernel_size=3, stride=1` (réduit seulement de 2 pixels) pour garder des cartes lisibles à visualiser.

---

## 3. Calculer les dimensions

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K-1) - 1}{S} \right\rfloor + 1 \quad\overset{D=1}{=}\quad \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1$$

- $K$ : `kernel_size` ; $P$ : `padding` ; $S$ : `stride` ; $D$ : `dilation`. Même formule pour le pooling.

Exemple du TP (MNIST 28×28, `ConvNet`) :

| Couche | Taille |
|---|---|
| entrée | 1×28×28 |
| `Conv2d(1,16,5)` | 16×24×24 |
| `MaxPool2d(3, stride=1)` | 16×22×22 |
| `Conv2d(16,16,5)` | 16×18×18 |
| `MaxPool2d(3, 1)` | **16×16×16** → `Linear(16*16*16, 120)` |

CIFAR (3×32×32, `ConvCIFAR(32)`) : 32 → 28 → 26 → 22 → **20** → `Linear(20*20*32, 120)`.

> 💡 **Astuce sans calcul** : passer un faux batch et lire la forme : `model.features(torch.zeros(1, 1, 28, 28)).shape`. 🔎 Ou utiliser `nn.LazyLinear(120)` qui déduit `in_features` au premier appel.

---

## 4. Construire un CNN : `features` + `classifier`

Convention (compatible avec les fonctions d'introspection) : `self.features` = partie convolutive, `self.classifier` = partie dense.

```python
class ConvNet(nn.Module):
    def __init__(self, pooling=nn.MaxPool2d):            # pooling paramétrable
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool1 = pooling(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.pool2 = pooling(3, 1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.features   = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                        self.conv2, nn.ReLU(), self.pool2)
        self.classifier = nn.Sequential(nn.Linear(16*16*16, 120), nn.ReLU(),
                                        nn.Linear(120, 10))     # scores bruts → CrossEntropyLoss
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))          # aplatir en gardant le batch

convnet     = ConvNet().to(device)
convnet_avg = ConvNet(nn.AvgPool2d).to(device)
```

> ⚠️ Aplatir avec `x.view(x.size(0), -1)` (ou `flatten(1)`), **jamais** `x.view(-1)` : on mélangerait les exemples du batch.
> 👉 Pattern standard : `[Conv → (BatchNorm2d) → ReLU → Pool] × n → Flatten → Linear → ReLU → (Dropout) → Linear`. Augmenter le nombre de filtres quand la résolution baisse (32 → 64 → 128).

---

## 5. Inspecter un modèle

| Outil | Explication |
|---|---|
| `print(model)` | Arborescence des modules (rapide et efficace). |
| `for name, p in model.named_parameters(): print(name, p.size(), p.requires_grad)` | Liste des paramètres ; utile pour **geler / dégeler** des couches. |
| `from torchsummary import summary; summary(model, input_size=(1, 28, 28))` | Tableau couche par couche : forme de sortie + nb de paramètres (`pip install torchsummary`). ⚠️ `input_size` **sans** dimension de batch. |
| `sum(p.numel() for p in model.parameters() if p.requires_grad)` | Nombre de paramètres **entraînables**. |
| `model.features._modules.values()` / `model.classifier[6]` | Accéder aux sous-modules (par position dans un `Sequential`). |

> 💡 Comparer les tailles : sur CIFAR, le MLP `3072→500→100→10` a ~1,6 M paramètres pour ~51 % d'accuracy ; le petit CNN à 32 filtres ~66 % : la convolution est bien plus efficace par paramètre… sauf dans sa première couche dense (`20*20*32*120` ≈ 1,5 M) → d'où l'intérêt d'un pooling plus agressif / GAP.

---

## 6. Données image : torchvision

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `torchvision.datasets.MNIST(root, train, transform, download=True)` | | 60k/10k chiffres 28×28 en niveaux de gris. |
| `torchvision.datasets.CIFAR10(root, train, transform, download=True)` | `.classes` : `['airplane', 'automobile', 'bird', ...]` | 50k/10k images couleur 32×32, 10 classes. |
| `DataLoader(ds, batch_size, shuffle, pin_memory=True, num_workers)` | `pin_memory=True` accélère le transfert vers GPU | Batchs `(N, C, H, W)` + labels `(N,)` `long`. |
| `transforms.ToTensor()` | | PIL/ndarray `(H, W, C)` [0,255] → tenseur `(C, H, W)` float [0,1]. |
| `transforms.Normalize(mean, std)` | un tuple par canal ; ImageNet : `[0.485, 0.456, 0.406]`, `[0.229, 0.224, 0.225]` | $(x - \mu)/\sigma$ par canal. **Obligatoire avec les mêmes valeurs que le pré-entraînement** pour un modèle pré-entraîné. |
| `transforms.Resize(n)` / `transforms.CenterCrop(224)` | `Resize(224)` : petit côté à 224 | Adapter la taille à l'entrée attendue (224×224 pour AlexNet/VGG/ResNet). |

> ⚠️ Affichage : `plt.imshow(img.permute(1, 2, 0))` pour une image couleur `(C,H,W)` → `(H,W,C)` ; `img.squeeze()` pour une image en gris `(1,H,W)`. Si l'image a été normalisée, la **dé-normaliser** avant (`img*std + mean`).

---

## 7. Data augmentation

Insérer les transformations aléatoires dans le `Compose` **avant** `ToTensor()`, **uniquement pour le train** :

| Transformation | Paramètres | Effet |
|---|---|---|
| `transforms.RandomHorizontalFlip(p=0.5)` | `p` | Miroir gauche-droite (⚠️ pas pour des chiffres/du texte). |
| `transforms.RandomResizedCrop(size, scale=(0.08, 1.0))` | `size=32` ; 🔎 `scale=(0.6, 1.0)` moins agressif | Recadrage aléatoire puis redimensionnement. |
| `transforms.RandomCrop(32, padding=4)` | 🔎 | Le classique CIFAR (petites translations). |
| `transforms.RandomRotation(degrees)` | ex. `15` | Rotation aléatoire. |
| `transforms.ColorJitter(brightness, contrast, saturation, hue)` | ex. `0.2` | Variations de couleur / luminosité. |
| `transforms.AutoAugment()` / `RandAugment()` | 🔎 politiques toutes faites | Politiques d'augmentation automatiques (cf. diapo du module 1). |

```python
transformTrain = transforms.Compose([transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(32),
                                     transforms.ToTensor()])
transformTest  = transforms.Compose([transforms.ToTensor()])          # PAS d'aléatoire en test
train_aug = torchvision.datasets.CIFAR10('/tmp/data', train=True, download=False, transform=transformTrain)
```

> 💡 L'augmentation régularise : la loss de train baisse moins vite, mais l'écart train/test se réduit. Il faut souvent **plus d'epochs** (30 dans le TP).

---

## 8. Boucle d'entraînement CNN (AdamW + OneCycleLR)

```python
def train(model, epochs, train_loader, test_loader):
    writer = SummaryWriter(f"{TB_PATH}/{model.name}")
    max_lr = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr,
                           steps_per_epoch=len(train_loader), epochs=epochs)
    model = model.to(device); loss = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epochs)):
        model.train(); cumloss, cumacc, count = 0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            yhat = model(x); l = loss(yhat, y)
            l.backward(); optimizer.step(); scheduler.step()   # scheduler : à CHAQUE batch
            cumloss += l.item() * len(x)                         # pondérer : le dernier batch est plus petit
            cumacc  += (yhat.argmax(1) == y).float().sum().item()
            count   += len(x)
        writer.add_scalar('loss/train', cumloss / count, epoch)
        writer.add_scalar('accuracy/train', cumacc / count, epoch)
        model.eval()
        with torch.no_grad(): ...                                # même calcul sur test_loader
```

| Choix | Réglage |
|---|---|
| `OneCycleLR` | `steps_per_epoch=len(train_loader)`, `epochs` identiques à la boucle, sinon erreur « step au-delà du total ». `max_lr` ≈ 10× le lr habituel (1e-2 avec AdamW). |
| `AdamW` | `weight_decay=1e-2` par défaut. |
| Mac M1 (`mps`) | le TP remplace `argmax` par `torch.max(yhat, 1).indices` (bug historique). |

---

## 9. Modèles pré-entraînés & fine-tuning

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `models.alexnet(weights="DEFAULT")` | 🔎 `weights=models.AlexNet_Weights.IMAGENET1K_V1` ; l'ancien `pretrained=True` / `weights=True` est déprécié | Charge l'architecture **et** les poids ImageNet. Idem `models.vgg16`, `models.resnet18`, `models.resnet50`… |
| `torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)` | dépôt, nom | Charger un modèle depuis PyTorch Hub. |
| `weights.transforms()` | 🔎 | Le prétraitement exact attendu par le modèle. |
| `set_parameter_requires_grad(model, False)` | `for p in model.parameters(): p.requires_grad = False` | **Geler** le réseau. À faire **avant** de remplacer la tête (la nouvelle couche est créée avec `requires_grad=True`). |
| Remplacer la tête | AlexNet/VGG : `model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)` ; ResNet : `model.fc = nn.Linear(model.fc.in_features, 10)` | Adapter le nombre de classes (1000 → 10). |
| `optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))` | 🔎 | N'optimiser que ce qui est dégelé (sinon Adam ignore juste les `grad=None`). |

```python
alexnet = models.alexnet(weights="DEFAULT")
set_parameter_requires_grad(alexnet, False)                        # 1. geler
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)  # 2. nouvelle tête
transformAlex = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])  # 3. prétraitement ImageNet
train(alexnet, 10, alex_trainloader, alex_testloader)              # ~83 % sur CIFAR (vs ~66 % CNN maison)
```

| Situation | Stratégie |
|---|---|
| Peu de données, domaine proche d'ImageNet | geler tout, n'entraîner que la tête (*feature extraction*) |
| Données moyennes | dégeler aussi les derniers blocs, lr faible (1e-4) |
| Beaucoup de données / domaine éloigné | tout fine-tuner (lr faible) ou entraîner de zéro |

> ⚠️ Recharger des poids sauvegardés d'un modèle modifié : **reconstruire exactement la même architecture** (même remplacement de tête) avant `load_state_dict`.

---

## 10. Prédire sur une image externe

```python
from PIL import Image
im = Image.open("dog.jpg").convert("RGB")
im = im.resize((32, 32), Image.Resampling.LANCZOS)           # si le modèle a appris sur du 32×32
x = transformAlexTest(im).unsqueeze(0).to(device)            # (1, 3, 224, 224) : batch de 1
model.eval()
with torch.no_grad():
    probas = torch.softmax(model(x), dim=1)
top5_prob, top5_id = torch.topk(probas[0], 5)                # 5 classes les plus probables
print([(classes[i], p.item()) for p, i in zip(top5_prob, top5_id)])
```

> 👉 Toujours appliquer **le même prétraitement qu'en test** (taille, normalisation) et ne pas oublier `unsqueeze(0)` + `eval()`.

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Mauvais `in_features` après les convolutions | `mat1 and mat2 shapes cannot be multiplied` | calculer la taille (§3) ou tester un faux batch / `LazyLinear` |
| `x.view(-1)` pour aplatir | formes incohérentes, exemples mélangés | `x.view(x.size(0), -1)` / `nn.Flatten()` |
| `MaxPool2d(3)` en croyant `stride=1` | cartes 3× plus petites | `stride` vaut `kernel_size` par défaut : le préciser |
| `in_channels=1` sur des images RGB | `expected input to have 1 channels, but got 3` | `in_channels=3` |
| Image seule sans dimension de batch | erreur de dimension / BatchNorm qui plante | `img.unsqueeze(0)` |
| Augmentation aléatoire sur le test | évaluation bruitée | transformations aléatoires **train uniquement** |
| Modèle pré-entraîné sans la normalisation ImageNet / mauvaise taille | précision médiocre | `Resize(224)` + `Normalize(mean, std)` ImageNet |
| Geler **après** avoir remplacé la tête | la nouvelle tête est gelée aussi → rien n'apprend | geler puis remplacer |
| `HorizontalFlip` sur des chiffres | confusion (2/5, 6/9…) | augmentations cohérentes avec la tâche |
| `plt.imshow` sur `(C, H, W)` | `Invalid shape` | `.permute(1, 2, 0)` (+ dé-normaliser) |
| Accumuler `cumloss += l` (tenseur) | mémoire qui gonfle | `l.item()` |
| `OneCycleLR` mal dimensionné | `Tried to step X times...` | `steps_per_epoch=len(loader)` et mêmes `epochs` |
| `pretrained=True` (déprécié) | warning / erreur selon version | `weights="DEFAULT"` |

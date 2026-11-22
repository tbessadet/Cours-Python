# Quiz d'entraînement — Architectures CNN (code à compléter)

Construit à partir de `quizz3.pdf` et des notebooks `2_0` à `2_5`. Essaie chaque question **sans exécuter le code**, puis ouvre la réponse.

**Rappels utiles**

- Taille de sortie d'une conv / d'un pooling : $H_{out} = \left\lfloor \dfrac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1$ (pour $d=1$ : $\lfloor (H+2p-k)/s \rfloor + 1$)
- Pooling : par défaut `stride = kernel_size` (≠ conv où `stride = 1`)
- Paramètres d'une `Conv2d(c_in, c_out, k)` : $c_{in} \cdot c_{out} \cdot k^2 + c_{out}$ ; d'un `Linear(n, m)` : $n \cdot m + m$
- Entrée d'un CNN : **(B, C, H, W)** ; d'un MLP : **(B, d)**

---

## Q1 — Échauffement : dimensions et première valeur (type Q1.1 / Q1.2)

```python
x = torch.tensor([float(i) for i in range(25)]).view(5, 5)
c1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
c2 = nn.Conv2d(1, 1, kernel_size=2, stride=2, bias=False)
c3 = nn.Conv2d(1, 1, kernel_size=2, dilation=2, bias=False)
c4 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
p1 = nn.MaxPool2d(2)
p2 = nn.AvgPool2d(2)
```

a) Que faut-il ajouter à `x` pour pouvoir écrire `c1(x)` ?
b) Donner la taille de sortie de chaque couche.
c) Si tous les poids valent 1, donner la **première** valeur de chaque sortie (et celle de `p1`, `p2`).

<details>
<summary>Réponse</summary>

a) Une `Conv2d` attend (C, H, W) ou (B, C, H, W) : il faut `x.unsqueeze(0)` (→ 1×5×5) ou, mieux, `x.view(1, 1, 5, 5)`.

b) et c) (première valeur = somme des pixels couverts par le filtre en haut à gauche)

| couche | calcul | taille | 1re valeur |
|---|---|---|---|
| `c1` k=3, p=1 | (5+2−3)/1+1 | 5×5 | padding → seuls 0,1,5,6 sont dans l'image : **12** |
| `c2` k=2, s=2 | ⌊(5−2)/2⌋+1 | 2×2 | 0+1+5+6 = **12** |
| `c3` k=2, d=2 | 5−2·1−1+1 | 3×3 | filtre « troué » : x[0,0]+x[0,2]+x[2,0]+x[2,2] = 0+2+10+12 = **24** |
| `c4` k=3, s=2, p=1 | ⌊(5+2−3)/2⌋+1 | 3×3 | **12** (même fenêtre que `c1`) |
| `p1` MaxPool 2 | stride=2 par défaut | 2×2 | max(0,1,5,6) = **6** |
| `p2` AvgPool 2 | idem | 2×2 | 12/4 = **3** |

⚠ Le pooling n'a **aucun paramètre** et agit canal par canal (pas de `in_channels`/`out_channels`). Avec le même hyper-paramétrage qu'une conv, il donne la même taille de sortie — mais attention au `stride` par défaut qui n'est pas le même.
</details>

---

## Q2 — CNN avec `nn.Sequential` (la « dernière question », version CNN)

On veut un CNN pour classer des images MNIST (1×28×28, 10 classes) :
Conv 3×3 à 8 filtres avec `padding=1` → ReLU → MaxPool 2×2 → Conv 3×3 à 16 filtres (sans padding) → ReLU → MaxPool 2×2 → mise à plat → Linear vers 64 → ReLU → Linear de sortie.

Donner le code pour :
- instancier ce réseau avec `torch.nn.Sequential` ;
- créer un **batch de 4 images** aléatoires à la bonne taille ;
- faire passer ce batch dans le réseau, et donner la taille de la sortie.

<details>
<summary>Réponse</summary>

Trace des dimensions : 28 → conv(k=3, p=1) → **28** → pool → **14** → conv(k=3) → **12** → pool → **6**. Donc `in_features = 16 × 6 × 6 = 576`.

```python
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),  # (4, 8, 28, 28)
    nn.ReLU(),
    nn.MaxPool2d(2),                                                     # (4, 8, 14, 14)
    nn.Conv2d(8, 16, kernel_size=3),                                     # (4, 16, 12, 12)
    nn.ReLU(),
    nn.MaxPool2d(2),                                                     # (4, 16, 6, 6)
    nn.Flatten(),                                                        # (4, 576)
    nn.Linear(16 * 6 * 6, 64),
    nn.ReLU(),
    nn.Linear(64, 10)                                                    # (4, 10)
)

x = torch.randn(4, 1, 28, 28)   # (B, C, H, W)
yhat = net(x)
print(yhat.shape)               # torch.Size([4, 10])
```

Paramètres : conv1 = 1·8·9+8 = 80 ; conv2 = 8·16·9+16 = 1 168 ; lin1 = 576·64+64 = 36 928 ; lin2 = 64·10+10 = 650 → **38 826**.

⚠ Pièges fréquents :
- `nn.ReLU()` **avec parenthèses** (on passe une instance, pas la classe) ;
- les couches sont des arguments séparés par des virgules, pas une liste (sinon `nn.Sequential(*liste)`) ;
- ne pas oublier `nn.Flatten()` entre la partie conv et la partie linéaire ;
- pas de Softmax à la fin : `nn.CrossEntropyLoss` l'inclut ;
- le `in_channels` d'une conv = le `out_channels` de la précédente.

🔎 Version « régression » (comme Q1.4 du quiz 3) : même chose avec `nn.Linear(64, 1)` en sortie, sans activation, et la loss `nn.MSELoss()`. Pour le MLP du quiz 3 : `nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 1))` et `x = torch.randn(1, 6)` (ou `(B, 6)`).
</details>

---

## Q3 — Chasse aux erreurs

Ce code doit faire passer une image MNIST dans un petit CNN. Trouver **les 5 erreurs** (4 font planter, 1 est une faute de modélisation).

```python
x = torch.rand(28, 28)                 # une image MNIST
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=5),
    nn.ReLU,
    nn.MaxPool2d(2),
    nn.Linear(16 * 12 * 12, 10),
    nn.Softmax(dim=1)
)
loss = nn.CrossEntropyLoss()
yhat = model(x)
```

<details>
<summary>Réponse</summary>

1. **`x` sans dimensions batch et canal** : il faut `x = torch.rand(1, 1, 28, 28)` (ou `x.view(1, 1, 28, 28)`). Même en (1, 28, 28) « non batché », `Flatten` mettrait à plat la mauvaise dimension → toujours mettre le batch.
2. **`in_channels=3`** alors que MNIST est en niveaux de gris → `nn.Conv2d(1, 16, kernel_size=5)`.
3. **`nn.ReLU` sans parenthèses** → `TypeError` (ce n'est pas un module) → `nn.ReLU()`.
4. **`nn.Flatten()` manquant** avant le `Linear` : le Linear recevrait un tenseur (B, 16, 12, 12).
5. **`nn.Softmax` en sortie avec `CrossEntropyLoss`** : la CE applique déjà un log-softmax → on l'enlève (le réseau renvoie des *logits*).

Vérification de la taille : 28 → conv k=5 → 24 → pool 2 → 12, donc `16 * 12 * 12` est **juste**.
</details>

---

## Q4 — Classe `nn.Module` à compléter (CIFAR, cf. 2_3)

```python
class ConvCIFAR(nn.Module):
    def __init__(self, nb_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=____, out_channels=nb_channels, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=____)
        self.conv2 = nn.Conv2d(____, ____, kernel_size=5)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.features = nn.Sequential(____)
        self.classifier = nn.Sequential(nn.Linear(____, 120), nn.ReLU(),
                                        nn.Linear(120, 80), nn.ReLU(),
                                        nn.Linear(80, ____))

    def forward(self, x):
        ____
        ____
        return x
```

a) Compléter (images 3×32×32, pooling avec un pas de 1, 10 classes).
b) Combien de paramètres dans `conv1` ? dans `conv2` ? dans la première couche linéaire ?
c) Le MLP `Linear(3072,500) → Linear(500,100) → Linear(100,10)` a ~1,59 M paramètres, le CNN ~1,57 M. Pourquoi le CNN fait-il nettement mieux (≈ 66 % contre 51 %) ?

<details>
<summary>Réponse</summary>

a) Trace : 32 → conv k=5 → 28 → pool(3, s=1) → 26 → conv k=5 → 22 → pool(3, s=1) → **20**. Entrée du linéaire : 20 × 20 × 32 = **12 800**.

```python
class ConvCIFAR(nn.Module):
    def __init__(self, nb_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=5)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.features = nn.Sequential(self.conv1, nn.ReLU(), self.pool1,
                                      self.conv2, nn.ReLU(), self.pool2)
        self.classifier = nn.Sequential(nn.Linear(20 * 20 * nb_channels, 120), nn.ReLU(),
                                        nn.Linear(120, 80), nn.ReLU(),
                                        nn.Linear(80, 10))

    def forward(self, x):
        x = self.features(x)                              # (B, 32, 20, 20)
        x = self.classifier(x.view(x.size(0), -1))        # (B, 12800) -> (B, 10)
        return x
```

⚠ `x.view(x.size(0), -1)` garde la dimension batch et met tout le reste à plat (équivalent à `nn.Flatten()`). Écrire `x.view(-1)` mélangerait tout le batch.

b)
- `conv1` : 3·32·5·5 + 32 = **2 432**
- `conv2` : 32·32·5·5 + 32 = **25 632**
- `Linear(12800, 120)` : 12 800·120 + 120 = **1 536 120** → presque tous les paramètres sont dans la première couche dense, pas dans les convolutions.

c) Les convolutions partagent leurs poids sur toute l'image (invariance par translation, connexions locales) : elles extraient des motifs (bords, textures) utiles où qu'ils soient. Le MLP doit apprendre chaque motif à chaque position et ignore la structure spatiale de l'image.
</details>

---

## Q5 — Modifier un réseau existant : transfert et CAM (cf. 2_2 / 2_3)

a) Compléter pour adapter AlexNet pré-entraîné (1000 classes ImageNet) à CIFAR-10, en n'entraînant **que** la nouvelle couche :

```python
from torchvision import models
alexnet = models.alexnet(weights=True)
____                                            # geler les poids
num_ftrs = alexnet.classifier[6].____
alexnet.classifier[6] = ____                    # nouvelle couche de sortie
```

b) Pourquoi la nouvelle couche est-elle entraînable alors qu'on a tout gelé ? Combien de paramètres sont entraînés ?

c) Pour faire une CAM sur `ConvCIFAR` (sortie de `features` : 32×20×20), on remplace le classifieur par : remise en forme → moyenne globale de chaque carte → mise à plat → linéaire. Compléter :

```python
set_parameter_requires_grad(CIFARNet, False)
CIFARNet.classifier = nn.Sequential(nn.Unflatten(1, ____),
                                    nn.AvgPool2d(____, 1),
                                    nn.Flatten(1),
                                    nn.Linear(____, 10))
```
et donner la taille du tenseur après chaque couche pour un batch de B images.

<details>
<summary>Réponse</summary>

a)
```python
def set_parameter_requires_grad(model, b):
    for p in model.parameters():
        p.requires_grad = b

set_parameter_requires_grad(alexnet, False)
num_ftrs = alexnet.classifier[6].in_features          # 4096
alexnet.classifier[6] = nn.Linear(num_ftrs, 10)       # ou len(cifar_trainset.classes)
```
🔎 Les images CIFAR doivent aussi être redimensionnées en 224 et normalisées avec la moyenne/écart-type d'ImageNet (`transforms.Resize(224)`, `transforms.Normalize(mean, std)`), car c'est ce que le réseau a vu en pré-entraînement.

b) On gèle **avant** de créer la nouvelle couche : un `nn.Linear` neuf a `requires_grad=True` par défaut. Paramètres entraînés : 4096·10 + 10 = **40 970**.

c) Le `forward` fait `x.view(B, -1)` → le classifieur reçoit (B, 12 800) :

```python
set_parameter_requires_grad(CIFARNet, False)
CIFARNet.classifier = nn.Sequential(nn.Unflatten(1, (32, 20, 20)),   # (B, 32, 20, 20)
                                    nn.AvgPool2d(20, 1),             # (B, 32, 1, 1)
                                    nn.Flatten(1),                   # (B, 32)
                                    nn.Linear(32, 10))               # (B, 10)
```
Seulement 32·10 + 10 = **330** paramètres à réentraîner. La CAM pour la classe c = somme des 32 cartes de `features` pondérées par `weights[c]` (ligne c du dernier linéaire). Pour MNIST (ConvNet) : `Unflatten(1, (16, 16, 16))`, `AvgPool2d(16, 1)`, `Linear(16, 10)`.
</details>

---

## Q6 — Briques U-Net (cf. 2_5)

a) Pour `x = torch.randn(4, 8, 64, 64)`, donner la taille de :
```python
y1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)(x)
y2 = nn.MaxPool2d(2)(x)
y3 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)(x)
y4 = nn.Upsample(scale_factor=2, mode="nearest")(x)
y5 = torch.cat([x, x], dim=1)
y6 = nn.Conv2d(8, 11, kernel_size=1)(x)
```
b) Nombre de paramètres de `ConvTranspose2d(8, 4, kernel_size=2, stride=2)` ? de `Upsample` ?

c) Compléter un U-Net de profondeur 1 (entrée 1×64×64, 11 classes par pixel) :
```python
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DoubleConv(1, 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(8, 16)
        self.up = nn.ConvTranspose2d(____, ____, 2, stride=2)
        self.dec = DoubleConv(____, 8)
        self.head = nn.Conv2d(8, ____, kernel_size=1)

    def forward(self, x):
        ____
        ____
        ____
        ____
        return ____
```

d) Une image 28×28 passe dans 3 `MaxPool2d(2)` puis 3 `ConvTranspose2d(k=2, s=2)`. Quelle taille obtient-on ? Pourquoi travaille-t-on en 64×64 ?

<details>
<summary>Réponse</summary>

a)
| tenseur | taille | pourquoi |
|---|---|---|
| `y1` | (4, 16, 64, 64) | k=3, p=1 conserve la taille |
| `y2` | (4, 8, 32, 32) | pool 2 divise par 2 |
| `y3` | (4, 4, 128, 128) | conv transposée k=2, s=2 double la taille |
| `y4` | (4, 8, 128, 128) | upsample double, garde les canaux |
| `y5` | (4, 16, 64, 64) | concaténation sur les **canaux** (dim=1) |
| `y6` | (4, 11, 64, 64) | conv 1×1 = linéaire pixel par pixel |

b) `ConvTranspose2d` : poids de forme (c_in, c_out, k, k) = 8·4·2·2 = 128, + 4 biais = **132**. `Upsample` : **0** (interpolation fixe).

c)
```python
class MiniUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DoubleConv(1, 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(8, 16)
        self.up = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec = DoubleConv(2 * 8, 8)          # concat up (8) + skip (8)
        self.head = nn.Conv2d(8, 11, kernel_size=1)

    def forward(self, x):
        s = self.enc(x)                          # (B, 8, 64, 64)  -> skip
        x = self.bottleneck(self.pool(s))        # (B, 16, 32, 32)
        x = self.up(x)                           # (B, 8, 64, 64)
        x = self.dec(torch.cat([x, s], dim=1))   # (B, 16, 64, 64) -> (B, 8, 64, 64)
        return self.head(x)                      # (B, 11, 64, 64)
```
⚠ Le `in_channels` du décodeur est **le double** à cause de la concaténation ; on concatène sur `dim=1` (canaux), jamais sur la dimension 0 (batch).

d) 28 → 14 → 7 → **3** (arrondi inférieur) → 6 → 12 → **24** ≠ 28. Au premier niveau du décodeur, on aurait 6×6 à concaténer avec un skip 7×7 → erreur. On prend 64 = 2⁶ car il est divisible par 2³ : chaque pooling tombe juste et les tailles se retrouvent à la remontée.
</details>

---

## Bonus — Checkpointing pour l'early stopping (Q1.3 du quiz 3, en code)

Compléter la boucle pour garder le meilleur modèle selon la loss de validation.

<details>
<summary>Réponse</summary>

```python
best_loss = float("inf")
for epoch in range(epochs):
    model.train()
    # ... boucle d'apprentissage sur train_loader ...
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss(model(x), y).item() * len(x) for x, y in val_loader) / len(val_loader.dataset)
    if val_loss < best_loss:              # la validation s'améliore -> on sauvegarde
        best_loss = val_loss
        save_model(model, "models/best")
load_model("models/best", model)          # on récupère le modèle optimal
```
La courbe de train décroît toujours ; celle de validation décroît puis remonte (sur-apprentissage). On garde le modèle du minimum de la courbe de validation.
</details>

---

## Pièges à éviter

- ⚠ Oublier la dimension batch (et canal) : un CNN veut `(B, C, H, W)`, un MLP `(B, d)`.
- ⚠ `nn.ReLU` au lieu de `nn.ReLU()` dans un `Sequential`.
- ⚠ Oublier `nn.Flatten()` (ou `x.view(x.size(0), -1)`) avant le premier `Linear`.
- ⚠ Mal calculer `in_features` : faire la trace des tailles couche par couche, et multiplier **canaux × H × W**.
- ⚠ Confondre le `stride` par défaut : 1 pour une conv, `kernel_size` pour un pooling.
- ⚠ Mettre un Softmax en sortie avec `CrossEntropyLoss`, ou une activation en sortie d'une régression.
- ⚠ En U-Net : concaténer sur `dim=1` et doubler le `in_channels` du bloc décodeur.
- ⚠ Pour le transfert : geler **avant** de remplacer la dernière couche.

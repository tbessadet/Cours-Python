# 🔥 Cheat Sheet PyTorch — Tenseurs & Extraction de features (CNN pré-entraînés)

> Basée sur le notebook *Utilisation de CNNs pré-entraînés pour extraire des caractéristiques visuelles*.
> Convention : `import torch`, `import torchvision.models as models`, `from torchvision import transforms`

## Sommaire
1. [Tenseurs : conversion numpy ↔ torch](#1-tenseurs--conversion-numpy--torch)
2. [Charger un modèle pré-entraîné](#2-charger-un-modèle-pré-entraîné)
3. [Prétraitement des images](#3-prétraitement-des-images)
4. [Extraction de features (inférence)](#4-extraction-de-features-inférence)
5. [Encapsulation compatible scikit-learn](#5-encapsulation-compatible-scikit-learn)
6. [Comparer des représentations](#6-comparer-des-représentations)
7. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Tenseurs : conversion numpy ↔ torch

PyTorch travaille avec ses propres tableaux (`torch.Tensor`), très proches des tableaux numpy, mais avec des conventions différentes — en particulier l'**ordre des dimensions des images**.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torch.from_numpy(arr)` | `arr` = tableau numpy | Convertit un tableau numpy en tenseur torch, **sans copier les données** (les deux partagent la même mémoire). |
| `.numpy()` | — | Conversion inverse, tenseur torch → tableau numpy (nécessite d'être sur CPU et hors du graphe de calcul, voir `.detach()` si besoin). |
| `.float()` | — | Convertit vers `float32`. ⚠️ Les modèles PyTorch attendent en général du `float32`, alors que numpy produit souvent du `float64` par défaut (`np.random.randn`) — conversion **indispensable** avant de passer un tenseur dans un modèle. |
| `.permute(dims)` | `dims` = nouvel ordre des axes | Réordonne les dimensions d'un tenseur, **sans copier les données** (change juste la façon de les lire). Essentiel pour les images (voir piège ci-dessous). |
| `.view(shape)` / `.reshape(shape)` | `shape` | Équivalent du `reshape` numpy. `.view` nécessite que les données soient contiguës en mémoire (souvent le cas juste après un chargement, pas forcément après un `.permute`). |

```python
images = np.random.randn(n, h, w, 3)                    # convention numpy/image classique : (N, H, W, C)
images = torch.from_numpy(images).permute(0, 3, 1, 2)    # -> (N, C, H, W), convention PyTorch
images = images.float()                                   # float64 -> float32, obligatoire
```

⚠️ **Convention des images `(N, H, W, C)` en numpy/PIL vs `(N, C, H, W)` en PyTorch** (canaux en 2ème position, pas en dernier) : un oubli de `.permute` fait planter le modèle (dimensions incompatibles) ou pire, produit un résultat silencieusement faux si les dimensions coïncident par hasard.

---

## 2. Charger un modèle pré-entraîné

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `models.squeezenet1_0(pretrained=True)` (ou tout autre modèle de `torchvision.models`) | `pretrained=True` | Charge une architecture CNN **avec des poids déjà appris** sur un grand jeu de données (ImageNet). Beaucoup d'autres architectures disponibles (`resnet18`, `vgg16`, `mobilenet_v2`...). |
| `.eval()` | — | Bascule le modèle en **mode inférence** : désactive le comportement spécifique à l'entraînement de certaines couches (dropout, batchnorm) qui fausserait les résultats si actif. **Indispensable** avant toute utilisation hors apprentissage. |

```python
model = models.squeezenet1_0(pretrained=True)
model = model.float().eval()
```

---

## 3. Prétraitement des images

Un modèle pré-entraîné a été appris avec des images normalisées d'une façon **précise** (mêmes moyenne/écart-type que le jeu d'entraînement d'origine) — il faut reproduire exactement ce prétraitement, sinon les représentations extraites perdent leur sens.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `transforms.Compose([...])` | liste de transformations | Enchaîne plusieurs prétraitements en un seul objet appelable, comme une `Pipeline` scikit-learn. |
| `transforms.Normalize(mean=, std=)` | `mean`, `std` = listes de 3 valeurs (une par canal RGB) | Centre-réduit chaque canal de couleur. Les valeurs `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` sont les statistiques **standard d'ImageNet**, à réutiliser telles quelles pour tout modèle pré-entraîné dessus. |
| `transforms.Resize(taille)` | `taille` | Redimensionne l'image à la taille attendue par le modèle (souvent commenté/optionnel si les images sont déjà à la bonne taille). |

```python
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 4. Extraction de features (inférence)

Plutôt que d'entraîner un réseau de neurones, on utilise ici un CNN pré-entraîné comme un simple **extracteur de descripteurs** : on lui donne une image, il renvoie un vecteur de caractéristiques exploitable ensuite par n'importe quel modèle classique (SVM, régression...).

| Élément | Explication |
|---|---|
| `torch.no_grad():` | Contexte qui désactive le calcul du gradient — **indispensable en inférence** : évite de gaspiller mémoire et temps de calcul à préparer une rétropropagation qui n'aura jamais lieu. |
| `model(img_tensor)` | Passe l'image dans le réseau (équivalent d'un `.predict`, mais on garde ici la sortie brute — un vecteur de features, pas une classe). |

```python
with torch.no_grad():
    features = model(img_tensor)   # img_tensor déjà prétraité, shape (N, C, H, W)
features = features.numpy()         # retour en numpy pour la suite du traitement (sklearn, matplotlib...)
```

---

## 5. Encapsulation compatible scikit-learn

En héritant de `BaseEstimator` et `TransformerMixin`, on peut intégrer un extracteur de features PyTorch directement dans une `Pipeline` scikit-learn, au même titre qu'un `StandardScaler` ou une `PCA`.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CNNFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    def fit(self, X, y=None):
        return self   # rien à apprendre : le modèle est déjà entraîné

    def transform(self, X):
        img_tensor = self.preprocess(X)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.numpy()

cnn_model = CNNFeatureExtractor(model, preprocess)
features = cnn_model.transform(images)   # utilisable ensuite dans n'importe quelle Pipeline sklearn
```

👉 Voir la cheat sheet *Scikit-learn — Prétraitement & pipelines* pour combiner cet extracteur avec un classifieur dans une `Pipeline` complète.

---

## 6. Comparer des représentations

Une fois des images transformées en vecteurs de features, on peut comparer des images entre elles via une **distance** dans cet espace de représentation — souvent bien plus pertinente que comparer les pixels bruts.

```python
from scipy.spatial.distance import cdist

flat_imgs = np.stack([img.reshape(-1) for img in images])   # comparaison "naïve" pixel à pixel
img_dist = cdist(flat_imgs, flat_imgs)

features_dist = cdist(features, features)   # comparaison dans l'espace des représentations CNN

plt.imshow(img_dist);  plt.title("Espace des images")
plt.imshow(features_dist);  plt.title("Espace de représentations")
```

💡 **Intérêt pédagogique** : deux images visuellement proches en pixels (même cadrage, même fond) peuvent être loin en distance de pixels bruts si légèrement décalées, alors que l'espace de représentation d'un CNN capture des **caractéristiques visuelles de plus haut niveau** (formes, textures) invariantes à ce type de décalage.

---

## 📌 Récapitulatif des pièges classiques

1. **Ordre des dimensions image** : numpy/PIL utilisent `(H, W, C)`, PyTorch attend `(C, H, W)` (canaux en 2ème position) — toujours `.permute()` avant de passer une image dans un modèle.
2. **`float64` vs `float32`** : les modèles PyTorch attendent du `float32` ; convertir systématiquement avec `.float()` un tenseur issu de numpy.
3. **Oublier `.eval()`** avant l'inférence : certaines couches (dropout, batchnorm) se comportent différemment en entraînement et en évaluation, faussant silencieusement les résultats si le mode n'est pas basculé.
4. **Oublier `torch.no_grad()`** en inférence : le calcul (inutile) du gradient consomme mémoire et temps pour rien, et peut même faire planter sur de grandes images faute de mémoire.
5. **Ne pas reproduire exactement la normalisation utilisée à l'entraînement du modèle pré-entraîné** (mêmes `mean`/`std`) — sinon les features extraites perdent leur sens, le modèle "voit" des images hors de sa distribution d'entraînement.
6. **Comparer des images en pixels bruts ≠ comparer leurs représentations apprises** : la distance en pixels est très sensible aux décalages/luminosité, la distance dans l'espace des features d'un CNN est bien plus robuste et sémantique.

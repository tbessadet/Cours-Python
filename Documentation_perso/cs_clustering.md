# 🧩 Cheat Sheet Clustering (apprentissage non-supervisé)

> Basée sur le notebook *Clustering* (correction).
> Convention : `import numpy as np`, `from sklearn import cluster, metrics`, `from scipy.spatial.distance import cdist`, `from scipy.cluster.hierarchy import dendrogram`

## Sommaire
1. [Principe et notations](#1-principe-et-notations)
2. [K-means "à la main"](#2-k-means-à-la-main)
3. [K-means avec scikit-learn](#3-k-means-avec-scikit-learn)
4. [Clustering hiérarchique ascendant](#4-clustering-hiérarchique-ascendant)
5. [DBSCAN](#5-dbscan)
6. [Sélection du nombre de clusters](#6-sélection-du-nombre-de-clusters)
7. [Comparaison de partitions](#7-comparaison-de-partitions)
8. [Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Principe et notations

Le clustering cherche une **partition** $P=\{P_1,...,P_K\}$ des individus en groupes homogènes, **sans étiquette** disponible (apprentissage non-supervisé). Chaque groupe $k$ est résumé par son **barycentre** $\mu_k$ (moyenne des individus qu'il contient).

👉 Différence fondamentale avec la classification supervisée : **pas de validation croisée classique possible**, puisqu'il n'y a pas de vérité terrain à comparer — il faut des critères de qualité spécifiques (voir §6).

---

## 2. K-means "à la main"

L'algorithme alterne deux étapes jusqu'à convergence : (1) associer chaque point à son centre le plus proche, (2) recalculer les centres comme barycentres des points qui leur sont associés.

| Étape / fonction | Paramètres essentiels | Explication |
|---|---|---|
| Initialisation des centres | `data`, `K` | Tirer `K` centres aléatoires **dans l'intervalle des données** (`np.random.uniform(low=x_min, high=x_max, size=(K, d))`), pas n'importe où dans l'espace. |
| Mise à jour des responsabilités (affectation) | `data`, `centroids` | Pour chaque point, calculer la distance à tous les centres (`scipy.spatial.distance.cdist(data, centroids)`) et affecter au centre le plus proche (`np.argmin(dist, axis=1)`). |
| Mise à jour des centres | `data`, `responsabilities` | Nouveau centre = **moyenne pondérée** des points affectés à ce cluster (pondération 0/1 selon l'appartenance). |
| Fonction objectif (`km_loss`) | — | Somme des distances au carré entre chaque point et le centre de son cluster — c'est cette quantité que l'algorithme fait décroître à chaque itération. |
| Critère d'arrêt | `eps`, `max_it` | On arrête quand les centres bougent peu (`diff < eps`) ou après un nombre maximal d'itérations. |

```python
from scipy.spatial.distance import cdist

def init_centroids(data, K=3):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    return np.random.uniform(low=x_min, high=x_max, size=(K, len(x_min)))

def update_responsabilities_km(data, centroids):
    dist = cdist(data, centroids)          # distance de chaque point à chaque centre
    classes = np.argmin(dist, axis=1)      # cluster le plus proche pour chaque point
    return classes

def update_centroids(data, responsabilities):
    # moyenne des points de chaque cluster, via une pondération 0/1 (broadcasting)
    prod = data[:, np.newaxis, :] * responsabilities[:, :, np.newaxis]
    centroids = np.sum(prod, axis=0) / np.sum(responsabilities, axis=0).reshape(-1, 1)
    return centroids
```

⚠️ **Sensibilité à l'initialisation** : selon le tirage aléatoire des centres de départ, l'algorithme peut converger vers des partitions différentes (minimum local). En pratique, `sklearn` relance plusieurs initialisations automatiquement (voir §3).

---

## 3. K-means avec scikit-learn

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `cluster.KMeans(n_clusters=)` | `sklearn.cluster` | `n_clusters` = nombre de clusters `K` | Version optimisée (relance plusieurs initialisations automatiquement, converge plus vite). |
| `.fit(X)` | — | — | Calcule la partition. |
| `.labels_` | — | — | Cluster attribué à chaque individu (équivalent de `classes` ci-dessus). |
| `.inertia_` | — | — | Valeur finale de la fonction objectif (somme des distances au carré) — utile pour la sélection du nombre de clusters (§6). |

```python
from sklearn import cluster

clusters = {}
for k in range(2, 12):
    clusters[f'k={k}'] = cluster.KMeans(n_clusters=k).fit(X)

plt.scatter(pca_X[:,0], pca_X[:,1], c=clusters['k=3'].labels_, cmap='plasma')  # visualisation via une PCA 2D
```

---

## 4. Clustering hiérarchique ascendant

Principe : au départ, chaque individu est son propre cluster ; à chaque itération, on **fusionne** les deux clusters les plus proches, jusqu'à n'en avoir plus qu'un seul. Le résultat complet se visualise sous forme d'arbre (**dendrogramme**).

| Fonction / attribut | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `cluster.AgglomerativeClustering(n_clusters=, distance_threshold=, linkage=)` | `sklearn.cluster` | `n_clusters=None` **avec** `distance_threshold=0` pour obtenir l'arbre complet ; `linkage` = `'ward'` (déf., minimise la variance intra-cluster), `'average'`, `'complete'`, `'single'` (critères de distance entre clusters) | `n_clusters` et `distance_threshold` sont **mutuellement exclusifs** : soit on fixe le nombre de clusters voulu, soit on fixe une distance de fusion maximale et le nombre de clusters en découle. |
| `.children_` | — | — | Historique des fusions (quels clusters ont été regroupés à chaque étape). |
| `.distances_` | — | — | Distance à laquelle chaque fusion a eu lieu — **croissante** au fil des itérations (les paires les plus proches fusionnent en premier). |
| `dendrogram(linkage_matrix)` | `scipy.cluster.hierarchy` | nécessite de reconstruire une `linkage_matrix` à partir de `.children_`, `.distances_` et des effectifs de chaque nœud | Affiche l'arbre de fusion, avec la distance de fusion en hauteur. |

```python
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram

# arbre complet, pour observer toutes les distances de fusion
hier_clustering = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=0)
hier_clustering.fit(X)

plt.plot(hier_clustering.distances_)   # évolution de la distance de fusion au fil des itérations
plt.xlabel('Itérations')

# une fois une distance seuil choisie visuellement sur le dendrogramme :
hier_k3 = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=8).fit(X)
print('n_clusters:', len(np.unique(hier_k3.labels_)))
```

💡 **Choisir le nombre de clusters via le dendrogramme** revient exactement à choisir une `distance_threshold` : tracer une ligne horizontale sur le dendrogramme à une hauteur donnée, et compter le nombre de branches coupées.

⚠️ **`linkage` change fortement le résultat** : `'single'` est très sensible au bruit (effet de chaîne), `'ward'` est le choix par défaut le plus robuste dans la majorité des cas.

---

## 5. DBSCAN

Algorithme basé sur la **densité** : un cluster est une zone dense de points, séparée des autres par des zones creuses. Ne nécessite **pas** de fixer le nombre de clusters à l'avance.

| Fonction / attribut | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `cluster.DBSCAN(eps=)` | `sklearn.cluster` | `eps` = rayon de la boule de voisinage explorée autour de chaque point | Si un point n'a aucun voisin dans un rayon `eps`, il est classé comme **bruit** (cluster `-1`), sans appartenir à aucun groupe. |
| `.labels_` | — | — | Cluster de chaque point ; **`-1` = bruit / anomalie**, pas un vrai cluster. |

```python
for e in np.linspace(0.1, 2, 10):
    cl = cluster.DBSCAN(eps=e).fit(X)
```

⚠️ **`eps` trop petit** → presque tous les points classés en bruit (`-1`). **`eps` trop grand** → tous les points fusionnent en un seul cluster. Le bon compromis se trouve en balayant plusieurs valeurs et en observant le nombre de clusters obtenus (histogramme des tailles de cluster).

---

## 6. Sélection du nombre de clusters

Sans étiquettes, impossible d'utiliser une validation croisée classique : on utilise des critères internes, basés uniquement sur la géométrie des points et de leur affectation.

| Métrique | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `.inertia_` (K-means) | — | — | Somme des distances au carré aux centres. **Décroît mécaniquement** avec le nombre de clusters (jamais un bon critère seul — en `K=n`, l'inertie est nulle). Utile surtout pour repérer un "coude" (méthode du coude). |
| `metrics.silhouette_score(X, labels)` | `sklearn.metrics` | — | Mesure si chaque point est bien plus proche de son cluster que des autres (entre -1 et 1, plus c'est haut mieux c'est). Ne fonctionne que si `len(np.unique(labels)) > 1`. |
| `metrics.silhouette_samples(X, labels)` | idem | — | Score silhouette **par point** (au lieu de la moyenne globale) — permet de visualiser l'homogénéité cluster par cluster. |
| `metrics.davies_bouldin_score(X, labels)` | idem | — | Ratio dispersion intra-cluster / distance inter-cluster, **moyenne sur tous les clusters** — plus c'est **bas**, mieux c'est (sens inverse de la silhouette !). |

```python
from sklearn import metrics

silhouette = {}
for k, mod in clusters.items():
    if len(np.unique(mod.labels_)) > 1:      # nécessaire : score non défini pour un cluster unique
        silhouette[k] = metrics.silhouette_score(X, mod.labels_, metric="euclidean")
    else:
        silhouette[k] = np.nan

inertias = {k: mod.inertia_ for k, mod in clusters.items()}
plt.plot(list(inertias.values()))   # méthode du coude
```

⚠️ **Silhouette (plus haut = mieux) et Davies-Bouldin (plus bas = mieux) peuvent être en désaccord** sur le "meilleur" nombre de clusters — c'est attendu, chaque métrique capture une notion légèrement différente de "bonne" partition. Croiser plusieurs critères plutôt que se fier à un seul.

---

## 7. Comparaison de partitions

Utile pour comparer deux algorithmes de clustering entre eux, ou un clustering à une vérité terrain connue (quand elle existe, à titre d'évaluation externe uniquement — jamais utilisée pendant l'apprentissage).

| Fonction | Import | Paramètres essentiels | Explication |
|---|---|---|---|
| `metrics.cluster.contingency_matrix(labels1, labels2)` | `sklearn.metrics` | — | Table croisant les deux partitions : si elles sont similaires, la matrice est quasi diagonale (après réordonnancement). Ne permet pas de sélectionner des hyperparamètres sans vérité terrain, seulement de **comparer deux résultats**. |
| `metrics.rand_score(labels1, labels2)` | idem | — | Résume la table de contingence en un seul score entre 0 et 1 : proportion de paires de points classées de façon cohérente entre les deux partitions (que ce soit "ensemble" ou "séparément" dans les deux cas). |

```python
contingency = metrics.cluster.contingency_matrix(clusters['k=2'].labels_, hier_clustering_k3.labels_)

rand = metrics.rand_score(cluster_labels, y_true.to_numpy().reshape(-1))  # comparaison à une vérité terrain
```

---

## 📌 Récapitulatif des pièges classiques

1. **K-means est sensible à l'initialisation** : deux lancements peuvent donner des partitions différentes (minimum local) — `sklearn` relance plusieurs fois en interne pour limiter ce risque.
2. **`.inertia_` seule n'est jamais un bon critère de sélection du nombre de clusters** : elle décroît mécaniquement avec `K` (nulle quand chaque point est son propre cluster).
3. **`silhouette_score` nécessite au moins 2 clusters distincts** : toujours vérifier `len(np.unique(labels)) > 1` avant de l'appeler, sinon erreur.
4. **Silhouette et Davies-Bouldin n'ont pas le même sens** : silhouette "plus haut = mieux", Davies-Bouldin "plus bas = mieux" — facile à confondre.
5. **DBSCAN : le cluster `-1` est du bruit**, pas un vrai groupe — à ne pas compter comme un cluster "de plus" dans les analyses.
6. **`AgglomerativeClustering` : `n_clusters` et `distance_threshold` sont exclusifs** — n'en fixer qu'un des deux (l'autre doit être laissé à sa valeur par défaut, souvent `None`).
7. **La table de contingence / l'indice de Rand ne servent qu'à comparer deux partitions** (ou une partition à une vérité terrain externe) — ils ne remplacent pas silhouette/Davies-Bouldin pour choisir un nombre de clusters en l'absence totale d'étiquettes.

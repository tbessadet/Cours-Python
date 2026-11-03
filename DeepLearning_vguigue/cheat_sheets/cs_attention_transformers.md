# 🧠 Cheat Sheet Attention & Transformers — PyTorch, Hugging Face, LLM local (Ollama)

> Basée sur les notebooks `5_1-attention`, `5_2-self-attention` (dossier `notebooks`) et `5_0_transf`, `5_1_transf_classif`, `5_2_transf_pretrained`, `5_3_transf_pretrained-kmeans`, `5_4_transf_API` (dossier `newRNN`).
> Conventions : `import torch`, `from torch import nn`, `import torch.nn.functional as F`, `import math`.
> Prérequis : `cs_pytorch_fondamentaux.md`, `cs_rnn_torch.md` (padding, masques, seq2seq).

## Sommaire
1. [Attention : l'idée et les formules](#1-attention--lidée-et-les-formules)
2. [Attention simple : pooling pondéré appris](#2-attention-simple--pooling-pondéré-appris)
3. [Softmax masqué](#3-softmax-masqué)
4. [Self-attention « maison » : Q, K, V](#4-self-attention--maison---q-k-v)
5. [Couche résiduelle, LayerNorm, encodage de position](#5-couche-résiduelle-layernorm-encodage-de-position)
6. [`nn.TransformerEncoder` de PyTorch](#6-nntransformerencoder-de-pytorch)
7. [Classifieur Transformer complet (token `[CLS]`)](#7-classifieur-transformer-complet-token-cls)
8. [Analyser un Transformer : embeddings et matrices d'attention](#8-analyser-un-transformer--embeddings-et-matrices-dattention)
9. [Hugging Face : modèles pré-entraînés](#9-hugging-face--modèles-pré-entraînés)
10. [Fine-tuning avec `Trainer`](#10-fine-tuning-avec-trainer)
11. [Embeddings `[CLS]` + clustering](#11-embeddings-cls--clustering)
12. [LLM local avec Ollama](#12-llm-local-avec-ollama)
13. [📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Attention : l'idée et les formules

L'attention est un **pooling (une moyenne) dont les poids sont appris** et dépendent du contenu : au lieu de moyenner uniformément les éléments, on pondère chacun par un score normalisé par un softmax.

Pour une séquence de vecteurs $h_1,\dots,h_N\in\mathbb R^{d}$ et un score $e_i$ par élément :
$$a_i=\frac{\exp(e_i)}{\sum_{j=1}^{N}\exp(e_j)}\qquad c=\sum_{i=1}^{N}a_i\,h_i$$

- $e_i$ : score (réel) de l'élément $i$ ; $a_i\ge0$, $\sum_i a_i=1$ : poids d'attention ; $c$ : vecteur de contexte (combinaison convexe).
- *Uniforme* : $a_i=1/N$ ⇒ moyenne classique (cas de base du TD).

**Attention par produit scalaire (Q, K, V)** — hypothèses : $Q\in\mathbb R^{n\times d_k}$ (questions), $K\in\mathbb R^{m\times d_k}$ (clés), $V\in\mathbb R^{m\times d_v}$ (valeurs) :
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V$$

- $Q$ : « ce que je cherche » ; $K$ : « ce que je propose » ; $V$ : « ce que je transmets » ; $\sqrt{d_k}$ : facteur de normalisation qui empêche les produits scalaires de devenir trop grands (softmax saturé, gradients nuls). Le softmax s'applique **sur les clés** (chaque ligne de la matrice de scores somme à 1).

Les 4 formes envisagées dans le TD `5_1` (de la plus simple à la plus riche) :

| # | Modèle | Poids d'attention $a_i$ | Remarque |
|---|---|---|---|
| 1 | Uniforme | $1/N$ | Équivalent à un *average pooling* (avec embeddings gelés + `Linear`). |
| 2 | Attention simple | $\mathrm{softmax}(w^\top x_i)$ avec **un vecteur $w$** appris | Un seul « détecteur » de mots importants, commun à tous les textes. |
| 3 | Personnalisée | scores calculés par une petite sous-réseau (MLP) sur chaque mot | Chaque mot définit sa fonction d'attention. |
| 4 | RNN + attention | scores calculés sur les **états cachés** $h_t$ | Prend en compte le contexte. |

💡 **Entropie de l'attention** $H(a)=-\sum_i a_i\log a_i$ : mesure si l'attention est diffuse (≈ $\log N$, uniforme) ou concentrée (≈ 0). Le TD l'ajoute comme pénalité optionnelle : `loss = ce + entropy_pen * H` (pousse l'attention à se concentrer sur peu de mots) ; `+1e-10` dans le `log` évite `log(0)`.

---

## 2. Attention simple : pooling pondéré appris

```python
class ModelAttention1(nn.Module):
    def __init__(self, embeddings, label_count):
        super().__init__()
        self.emb_layer = nn.Embedding.from_pretrained(embeddings)        # GloVe gelé
        self.w = nn.Linear(embeddings.size(1), 1, bias=False)            # vecteur de score appris
        self.linear = nn.Linear(embeddings.size(1), label_count)
    def emb(self, x):  return self.emb_layer(x)                          # (B, N, E)
    def attention(self, x_emb, mask=None):
        a = self.w(x_emb).squeeze(-1)                                    # (B, N) scores
        if mask is not None: a = a.masked_fill(~mask, float('-inf'))
        return a.softmax(1)                                              # (B, N), somme 1 par texte
    def forward(self, x):
        e = self.emb(x); a = self.attention(e)
        return self.linear((a.unsqueeze(-1) * e).sum(1))                 # moyenne pondérée -> classes
```

| Outil | Explication |
|---|---|
| `x.softmax(dim)` / `F.softmax(x, dim)` | Normalise le long de `dim`. Pour des scores `(B, N)` : `dim=1` (sur les mots). Le TD montre `softmax([10,1,.5,.2,0,0]) ≈ [1, 1e-4, …]` : un score élevé **écrase** les autres. |
| `a.unsqueeze(-1) * e` | `(B,N,1) * (B,N,E)` → broadcasting puis `.sum(1)` : somme pondérée sur les mots. Équivalent : `torch.bmm(a.unsqueeze(1), e).squeeze(1)`. |
| `nn.Embedding.from_pretrained(emb)` | Embeddings **gelés** par défaut. |

Visualiser l'attention sur du texte : normaliser les poids dans $[0,1]$ et colorier chaque mot avec `matplotlib.colormaps['Reds']` + `IPython.display.HTML(...)` (une `<span style="background-color: …">` par mot) — voir fonction `colorize` du TD. Pratique pour repérer les mots que le modèle juge importants (ex. « bad », « great »).

---

## 3. Softmax masqué

Un batch de textes de longueurs variables est complété par du **padding** ; il ne doit **jamais recevoir de poids d'attention**. On met ses scores à $-\infty$ **avant** le softmax (donc poids $e^{-\infty}=0$).

```python
def masked_softmax(scores, lens):                 # scores : (B, N) ; lens : (B,) longueurs réelles
    mask = torch.arange(scores.size(1), device=scores.device).unsqueeze(0) < lens.unsqueeze(1)   # (B, N) True = vrai mot
    return scores.masked_fill(~mask, float('-inf')).softmax(1)

# cas self-attention (B, N, N) : masquer les COLONNES (clés) correspondant au padding
scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))     # mask : (B, N) -> (B, 1, N)
a = scores.softmax(-1).nan_to_num(0)          # nan_to_num : une ligne entièrement masquée donnerait des NaN
```

⚠️ **Une ligne entièrement masquée** (séquence vide / pad) donne `softmax([-inf,…]) = NaN` → `nan_to_num(0)` ou garantir au moins un token réel (le `[CLS]` joue ce rôle).

⚠️ Dans le notebook `5_2`, la fonction `masked_softmax` fournie **ignore `lens`** (elle fait seulement `softmax(-1).nan_to_num(0)`) : à vous de construire le masque. Appliquer le masque **sur les scores**, pas sur les probabilités après coup (sinon elles ne somment plus à 1).

---

## 4. Self-attention « maison » : Q, K, V

Chaque élément de la séquence produit sa propre question, clé et valeur par des projections linéaires apprises ; **chaque position regarde toutes les positions** (y compris elle-même) ; les entrées et sorties ont la même forme `(B, L, d)`.

```python
class AttentionBasicLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim); self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim); self.final = nn.Linear(dim, dim)
    def forward(self, x, lens=None):                           # x : (B, L, d)
        Q, K, V = self.query(x), self.key(x), self.value(x)     # (B, L, d) chacun
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)    # (B, L, L) : scores[b,i,j] = Q_i · K_j
        a = masked_softmax(scores, lens)                              # (B, L, L), chaque ligne somme à 1
        out = torch.bmm(a, V)                                         # (B, L, d)
        return F.relu(self.final(out)), a                             # + matrice d'attention pour l'analyse
```

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `torch.bmm(A, B)` | `A (B,n,m)`, `B (B,m,p)` → `(B,n,p)` | Produit matriciel **batché** (strictement 3-D). Pour plus de dimensions : `torch.matmul` ou `@`. |
| `K.transpose(1, 2)` | axes à permuter | `(B,L,d)` → `(B,d,L)` pour calculer $QK^\top$. |
| `math.sqrt(d_k)` | — | Normalisation du produit scalaire. |

Lecture de la **matrice d'attention** `(L, L)` (ligne $i$ = où le token $i$ regarde) : une diagonale forte = chaque mot s'écoute surtout lui-même ; elle n'est **pas forcément symétrique** ($QK^\top\neq KQ^\top$ car $W_Q\ne W_K$). Chaque **ligne** somme à 1 (softmax sur les clés), pas les colonnes.

💡 **Multi-têtes** : $h$ têtes en parallèle avec chacune une dimension $d/h$ (`nn.MultiheadAttention` ou `nhead` dans `TransformerEncoderLayer`), concaténées puis projetées : chaque tête peut se spécialiser (syntaxe, coréférence…). Le TD travaille avec 1 tête pour pouvoir tout calculer à la main.

---

## 5. Couche résiduelle, LayerNorm, encodage de position

### Connexion résiduelle + normalisation
```python
class AttentionResidualLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim); self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim); self.final = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
    def forward(self, x, lens=None):
        x = self.layer_norm(x)                                       # normalisation avant l'attention (pre-norm)
        ...                                                          # Q, K, V, scores, masked_softmax, out = a @ V
        return F.relu(self.final(x + out))                           # connexion RÉSIDUELLE : on ajoute l'entrée x
```
| Composant | Explication |
|---|---|
| **Résiduelle** $x+f(x)$ | Chemin direct du gradient : permet d'**empiler beaucoup de couches** sans dégradation ; la couche apprend une correction. |
| `nn.LayerNorm(normalized_shape)` | Normalise **chaque vecteur** (sur la dimension d'embedding), contrairement à `BatchNorm` (sur le batch) ; indépendant de la taille du batch et de la longueur de séquence → standard pour les séquences. |

### Encodage de position
La self-attention est **invariante à l'ordre** des tokens (sans position, « chien mord homme » = « homme mord chien »). On ajoute donc une information de position.

| Variante | Mise en œuvre | Explication |
|---|---|---|
| **Apprise** | `nn.Embedding(MAX_LENGTH, dim)` ; `x + pos_emb(torch.arange(L))` | Un vecteur appris par position ; limité à `MAX_LENGTH`. |
| **Sinusoïdale** (fixe) | formule ci-dessous | Pas de paramètres ; généralise à des longueurs plus grandes. |

$$PE_{(p,2i)}=\sin\!\Big(\frac{p}{10000^{2i/d}}\Big)\qquad PE_{(p,2i+1)}=\cos\!\Big(\frac{p}{10000^{2i/d}}\Big)$$

- $p$ : position dans la séquence ; $i$ : indice de paire de dimensions ; $d$ : dimension du modèle (`d_model`). Chaque dimension est une sinusoïde de fréquence différente ; le produit scalaire de deux encodages dépend de leur **distance**.

```python
def sinusoidal_embeddings(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)                                         # (L, 1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))   # (d/2,)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe                                                                         # (L, d)

# dans le module : self.register_buffer("pe", sinusoidal_embeddings(maxlen, d).unsqueeze(1))   # (L, 1, d)
# forward : x = emb(tokens) + self.pe[:L]                       # broadcast sur le batch
```

| Élément | Explication |
|---|---|
| `self.register_buffer("pe", tensor)` | Tenseur **non entraînable** mais suivi par le module : déplacé avec `.to(device)` et sauvegardé dans `state_dict`. Un simple attribut tenseur resterait sur CPU. |
| `pe.requires_grad = False` | Encodage fixe : pas de gradient. |

⚠️ Sans encodage de position, le modèle se comporte comme un **sac de mots** ; avec le TD `5_2`, on compare : 1 couche de base → 3 couches résiduelles → 3 couches résiduelles + position (modèles `transfo-base-l1`, `transfo-res-l3`, `transfo-res-pos-l3`).

⚠️ Longueur maximale : `MAX_LENGTH=500` tronque les textes (`item[0][:MAX_LENGTH]`) ; l'attention coûte $O(L^2)$ en temps et mémoire.

⚠️ Pooling final : le TD moyenne les sorties avec `out.sum(1) / lens.view(-1,1)`, ce qui inclut les positions de padding (qui valent des sorties non nulles) → préférer la somme masquée.

---

## 6. `nn.TransformerEncoder` de PyTorch

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, norm_first=False)` | `d_model` = dimension des embeddings (**divisible par `nhead`**) ; `nhead` = nb de têtes ; `dim_feedforward` = taille cachée du MLP interne ; `dropout` ; `activation` ∈ `relu`,`gelu` ; `batch_first` ; `norm_first` = LayerNorm avant (pre-norm) ou après | Une couche = **self-attention multi-têtes → Add & Norm → MLP position-wise (2 `Linear`) → Add & Norm**. |
| `nn.TransformerEncoder(encoder_layer, num_layers, norm=None)` | `num_layers` = nombre de couches empilées (copies indépendantes de la couche) | Empile les couches. Appel : `enc(src, mask=None, src_key_padding_mask=None)`. |
| `nn.TransformerDecoderLayer / nn.TransformerDecoder` | idem + `memory` | Self-attention **masquée causalement** + attention croisée sur l'encodeur (génération, seq2seq). |
| `nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)` | — | Bloc d'attention seul : `attn(q, k, v, key_padding_mask=…)` → `(sortie, poids)`. |

```python
layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=128, activation='relu')   # batch_first=False !
enc = nn.TransformerEncoder(layer, num_layers=4)
x = torch.randn(seq_len, batch, 128)                  # (L, B, d) par défaut
y = enc(x, src_key_padding_mask=padding_mask)         # (L, B, d) même forme
```

**Masques** (deux notions distinctes) :

| Masque | Forme | Rôle |
|---|---|---|
| `src_key_padding_mask` | `(B, L)` **bool**, `True` = position **ignorée** (padding) | Empêche d'accorder de l'attention au padding. |
| `mask` (*attention mask*) | `(L, L)` bool/float, `True`/`-inf` = interdit | Contrôle quel token peut voir quel autre ; **masque causal** `nn.Transformer.generate_square_subsequent_mask(L)` pour la génération (empêche de voir le futur). |

```python
padding_mask = (input_ids == PAD).T            # input_ids : (L, B) -> (B, L), True aux positions de padding
```

⚠️ **`batch_first=False` par défaut** : entrée `(L, B, d)`. Avec `batch_first=True` : `(B, L, d)`. Mélanger les deux = résultat faux sans erreur (si `L == B`) ou erreur de dimensions.

⚠️ `d_model` doit être divisible par `nhead` ; `dim_feedforward` défaut 2048 (≫ `d_model`) — le TD utilise des petites valeurs pour aller vite.

⚠️ Le **masque de padding utilise un booléen True = ignoré**, à l'inverse du masque du §3 (`True` = vrai mot) → bien inverser.

⚠️ Ne pas oublier l'**encodage de position** : `nn.TransformerEncoder` n'en ajoute aucun.

🔎 Le TD utilise la version *post-norm* (`norm_first=False`) ; les modèles modernes préfèrent le *pre-norm* (`norm_first=True`), plus stable à l'entraînement.

---

## 7. Classifieur Transformer complet (token `[CLS]`)

Tâche : classification d'opinion (IMDB). On préfixe chaque texte d'un token spécial **`[CLS]`** : après les couches d'attention, sa sortie résume tout le texte et nourrit le classifieur.

```python
class TransSent(nn.Module):
    def __init__(self, emb_size, voc_size, num_layers, num_heads, hidden_mlp, output_size, maxlen=1000):
        super().__init__()
        self.emb_size = emb_size
        self.emb = nn.Embedding(voc_size, emb_size)
        layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=hidden_mlp, activation='relu')
        self.trans = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer("posemb", sinusoidal_embeddings(maxlen, emb_size).unsqueeze(1))   # (maxlen, 1, d)
        self.h2o = nn.Linear(emb_size, output_size)
    def forward(self, x):                                    # x : (L, B) indices, x[0] = [CLS]
        L = x.size(0)
        padding_mask = (x == PAD).T                          # (B, L)
        h = self.emb(x) * math.sqrt(self.emb_size) + self.posemb[:L]    # (L, B, d)   [échelle optionnelle 🔎]
        h = self.trans(h, src_key_padding_mask=padding_mask)             # (L, B, d)
        return self.h2o(h[0])                                            # sortie du [CLS] (position 0) -> (B, classes)
```

Préparation des données (code du TD) :
```python
tokenizer.train(files, WordPieceTrainer(vocab_size=10000, special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"]))
codes = [torch.tensor(tokenizer.encode("[CLS] " + txt).ids) for txt in txts]       # [CLS] en 1re position
PAD = tokenizer.encode("[PAD]").ids[0]
def collate_fn(batch):
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=False, padding_value=PAD), torch.stack(labels)   # (L, B)
```
Hyperparamètres du TD : `emb_size=128`, `num_layers=4`, `num_heads=4`, `hidden_mlp=128`, Adam `lr=5e-4`, 20 epochs ; modèles fournis `TransSent-1H-1L`, `-4H-4L-25k`, `-4H-4L-25k-pos`…

| Choix | Effet |
|---|---|
| Plus de **têtes** / **couches** | Plus de capacité (et de coût) ; comparer `1H-1L` vs `4H-4L`. |
| Plus de **données** (5k → 25k) | Améliore la généralisation (modèles `…-5k`, `…-25k`). |
| **Position sinusoïdale** | Ajout d'ordre ; modèles `…-pos`. |
| Prédiction sur `[CLS]` vs moyenne des tokens | `[CLS]` : appris pour résumer ; moyenne masquée : alternative robuste. |

⚠️ **Incohérence de padding dans le TD** : `pad_sequence(..., batch_first=False)` sans `padding_value` remplit avec **0** (= `[UNK]`), alors que le masque est construit avec `PAD` (= 3). Résultat : le masque ne coïncide pas avec les vrais paddings. → `padding_value=PAD` dans le `collate_fn`.

⚠️ Position du `[CLS]` : avec `(L, B, d)`, c'est `h[0]` ; avec `batch_first=True`, `h[:, 0]`. Le `[CLS]` doit être **toujours en 1ʳᵉ position** (tronquer le texte, pas le `[CLS]`).

⚠️ Seuls les **modules** sont déplacés par `.to(device)` ; les tenseurs constants doivent passer par `register_buffer` (sinon erreur de device).

---

## 8. Analyser un Transformer : embeddings et matrices d'attention

**Évolution des embeddings** (notebook `5_0`, `d_model=3` pour tracer en 3D) : passer des données aléatoires `(seq, batch, 3)` dans un encodeur, tracer l'entrée vs la sortie d'**un seul** exemple du batch (`data[:, i, :]`), relier les points entrée→sortie.

**Reconstruire la matrice d'attention d'une tête** à partir des poids de PyTorch :
```python
W = encoder_layer.self_attn.in_proj_weight        # (3d, d) : [W_Q ; W_K ; W_V] EMPILÉS dans cet ordre
b = encoder_layer.self_attn.in_proj_bias          # (3d,)
d = W.size(1)
qkv = x_i @ W.T + b                               # x_i : (L, d) pour un exemple i
Q, K = qkv[:, :d], qkv[:, d:2*d]                  # ⚠️ ordre PyTorch : Q d'abord, puis K, puis V
attn = F.softmax(Q @ K.T / math.sqrt(d), dim=1)   # (L, L)
attn.sum(1)                                       # = 1 pour chaque ligne ; attn.sum(0) ≠ 1
```
Cette reconstruction n'est valable **qu'avec une seule tête** et sans le dropout/LayerNorm intermédiaires (en `eval()`).

⚠️ Dans le notebook `5_0`, le code nomme `K = qkv[:, :d]` et `Q = qkv[:, d:2*d]` : c'est l'inverse de l'ordre réel de PyTorch (`in_proj_weight = [Q; K; V]`) ; la matrice obtenue est donc la **transposée** de l'attention réelle. Vérification simple : les **lignes** doivent sommer à 1 avec la bonne convention.

Analyses proposées pour le classifieur (`5_1`) : tester des phrases saisies (tokenisation + `[CLS]`), lister les critiques mal classées, afficher la **ligne du `[CLS]`** de la matrice d'attention (quels mots comptent pour la décision), visualiser les embeddings de mots dans TensorBoard (`writer.add_embedding(model.emb.weight, metadata=…)`).

💡 Pour récupérer les poids d'attention avec `nn.MultiheadAttention` : `attn_out, attn_w = mha(q, k, v, need_weights=True, average_attn_weights=False)` (poids par tête).

---

## 9. Hugging Face : modèles pré-entraînés

La bibliothèque `transformers` fournit tokenizer + modèle pré-entraîné par un **nom** (identifiant du Hub).

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `AutoTokenizer.from_pretrained(name)` | `name` = id du Hub (ex. `"distilbert-base-uncased-finetuned-sst-2-english"`) | Charge le tokenizer **correspondant exactement au modèle** (vocabulaire, tokens spéciaux). |
| `tok(texts, padding=True, truncation=True, max_length=None, return_tensors="pt")` | `padding` ∈ `True`/`"max_length"`/`False` ; `truncation` ; `max_length` ; `return_tensors="pt"` (tenseurs PyTorch) | Renvoie un dict : `input_ids` `(B, L)`, `attention_mask` `(B, L)` (**1 = vrai token, 0 = padding**, convention inverse du `src_key_padding_mask` de PyTorch). Convertit liste de phrases → batch. |
| `AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)` | `num_labels` | Encodeur + **tête de classification**. Sortie : objet avec `.logits` `(B, num_labels)`. Si `num_labels` ≠ celui du checkpoint, la tête est réinitialisée. |
| `AutoModel.from_pretrained(name)` | — | Encodeur **sans tête** : sortie `last_hidden_state` `(B, L, d)` (embeddings contextuels). |
| `model(**inputs)` | `inputs` = sortie du tokenizer | `**` dépaquette le dict en arguments nommés (`input_ids=…, attention_mask=…`). |
| `pipeline("sentiment-analysis", model=model, tokenizer=tok)` | tâche (`"text-classification"`, `"text-generation"`, `"fill-mask"`, `"summarization"`, `"translation"`…), `model`, `tokenizer`, `device=` | Enchaîne tokenisation → modèle → post-traitement. Renvoie `[{'label': 'POSITIVE', 'score': 0.99}, …]`. |
| `model.eval()` | — | Obligatoire pour l'inférence (désactive le dropout). |

```python
name = "distilbert-base-uncased-finetuned-sst-2-english"        # modèle léger déjà fine-tuné sur du sentiment
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()
inputs = tok(["I love this product!", "Worst experience ever."], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits                             # (2, 2)
probs = logits.softmax(-1); pred = probs.argmax(-1)             # model.config.id2label[pred]
clf = pipeline("sentiment-analysis", model=model, tokenizer=tok)
clf(["It's okay, not great but not bad either."])               # texte neutre : score moins tranché
```

⚠️ Toujours utiliser le **tokenizer du même identifiant** que le modèle. Sans `truncation=True`, un texte dépassant 512 tokens (limite BERT/DistilBERT) fait planter le modèle ; un avis IMDB long doit être tronqué (ou découpé).

⚠️ Pas de `torch.no_grad()` ⇒ graphe construit inutilement en inférence. `pipeline` gère le `no_grad` pour vous.

⚠️ Les sorties sont des **logits** : appliquer `softmax` pour obtenir des probabilités. Les labels (`NEGATIVE`/`POSITIVE`) sont dans `model.config.id2label`.

👉 Évaluer le modèle pré-entraîné sur IMDB directement (*zero-shot* vis-à-vis de ce jeu : le modèle a été entraîné sur SST-2, un autre jeu d'avis) avant de décider de fine-tuner.

---

## 10. Fine-tuning avec `Trainer`

Ajuster un modèle pré-entraîné sur son propre jeu de données, avec peu de code.

```python
from datasets import Dataset, DatasetDict
ds = DatasetDict({"train": Dataset.from_dict({"text": tr_texts, "label": tr_labels}),
                  "test":  Dataset.from_dict({"text": te_texts, "label": te_labels})})

def tokenize(batch): return tok(batch["text"], padding="max_length", truncation=True)
tokenized = ds.map(tokenize, batched=True)                       # applique par paquets

args = TrainingArguments(output_dir="./results", eval_strategy="epoch", learning_rate=2e-5,
                         per_device_train_batch_size=16, per_device_eval_batch_size=16,
                         num_train_epochs=1, weight_decay=0.01, logging_steps=10, logging_dir="./logs")
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
                  tokenizer=tok, compute_metrics=compute_metrics)
trainer.train(); trainer.evaluate()
```

| Paramètre (`TrainingArguments`) | Valeur usuelle | Explication |
|---|---|---|
| `output_dir` | `"./results"` | Dossier de sortie (checkpoints). |
| `learning_rate` | `2e-5` (1e-5 à 5e-5) | **Petit** `lr` pour ne pas détruire le pré-entraînement. |
| `per_device_train_batch_size` / `per_device_eval_batch_size` | 16–32 | Taille de batch par GPU. |
| `num_train_epochs` | 1–4 | Le fine-tuning converge en peu d'epochs. |
| `weight_decay` | `0.01` | Régularisation L2 (AdamW). |
| `eval_strategy` | `"epoch"` / `"steps"` / `"no"` | Quand évaluer. ⚠️ Dans les versions < 4.41 le nom est `evaluation_strategy`. |
| `logging_steps`, `logging_dir` | — | Fréquence/dossier des logs (TensorBoard). |
| `save_strategy`, `load_best_model_at_end`, `metric_for_best_model` | — | 🔎 Sauvegarde et restauration du meilleur checkpoint. |
| `fp16=True` | — | 🔎 Précision mixte sur GPU : plus rapide, moins de mémoire. |

```python
def compute_metrics(eval_pred):                     # sinon evaluate() ne renvoie que la loss
    logits, labels = eval_pred
    return {"accuracy": (logits.argmax(-1) == labels).mean()}
```

| Outil | Explication |
|---|---|
| `Dataset.from_dict({...})`, `DatasetDict` | Format `datasets` (colonnes en Arrow). |
| `ds.map(fn, batched=True)` | Applique `fn` par lots (bien plus rapide que ligne à ligne). |
| `DataCollatorWithPadding(tokenizer)` 🔎 | Padding **dynamique** au plus long du batch (au lieu de `padding="max_length"`, qui complète tout à 512 → lent). À passer à `Trainer(data_collator=…)` avec `padding=False` dans `tokenize`. |
| `trainer.train()` / `trainer.evaluate()` / `trainer.predict(ds)` | Entraîne / évalue / prédit. |

⚠️ Sans `compute_metrics`, `trainer.evaluate()` ne donne que la **loss** (pas d'accuracy).

⚠️ `Trainer(tokenizer=…)` est renommé `processing_class=…` dans les versions récentes.

⚠️ Colonnes : le dataset fourni au `Trainer` doit contenir `label` (ou `labels`) + `input_ids`/`attention_mask` ; les colonnes de texte brut sont ignorées.

⚠️ `padding="max_length"` + dataset de 25 000 avis = entraînement très long ; prévoir un sous-échantillon pour tester (`ds["train"].select(range(1000))`) ou le padding dynamique.

---

## 11. Embeddings `[CLS]` + clustering

Utiliser un Transformer pré-entraîné **comme extracteur de features** (voir aussi `cs_pytorch.md`), puis un algorithme classique.

```python
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").eval()

def get_cls_embeddings(texts):
    embs = []
    with torch.no_grad():
        for t in texts:
            inp = tok(t, return_tensors="pt", truncation=True, padding=True, max_length=128)
            out = model(**inp)
            embs.append(out.last_hidden_state[:, 0, :].squeeze(0).numpy())   # token [CLS] = indice 0
    return np.array(embs)                                                     # (n_textes, 768)

emb = get_cls_embeddings(texts)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
clusters = kmeans.fit_predict(emb)
```

| Élément | Paramètres | Explication |
|---|---|---|
| `outputs.last_hidden_state` | forme `(B, L, d)` | Représentation contextuelle de chaque token. `[:, 0, :]` = `[CLS]`. |
| `sklearn.cluster.KMeans(n_clusters, n_init, random_state)` | `n_clusters` = nombre de groupes | `fit_predict(X)` → étiquette de cluster par texte. |
| Pooling alternatif 🔎 | `(h * mask).sum(1) / mask.sum(1)` avec `attention_mask` | La **moyenne masquée** des tokens est souvent meilleure que `[CLS]` pour un modèle non fine-tuné (le `[CLS]` brut de BERT n'est pas entraîné pour la similarité). |

⚠️ `torch.Tensor.numpy()` nécessite CPU et pas de gradient (→ `torch.no_grad()` + `.cpu()`). KMeans exige un `ndarray` 2-D `(n_samples, n_features)`.

⚠️ Sur de très peu de textes (le TD en utilise 6), les clusters (ici « ML » vs « nature ») sont indicatifs ; normaliser les embeddings (cosinus) et fixer `random_state` pour la reproductibilité.

---

## 12. LLM local avec Ollama

Ollama exécute des LLM en local (modèle `phi3:latest` dans le TD) avec une API Python.

| Fonction | Paramètres essentiels | Explication |
|---|---|---|
| `ollama.chat(model, messages, options=None, format=None)` | `messages=[{"role": "user" \| "system" \| "assistant", "content": "…"}]` | Conversation multi-tours. Réponse : `response.message.content` (ou `response['message']['content']`). |
| `ollama.generate(model, prompt, options=None, format=None)` | `prompt` | Complétion simple à partir d'un prompt. Réponse : `.response`. |
| `ollama.embed(model, input)` | `input` = texte ou liste | Renvoie des embeddings (`.embeddings`). |
| `options={"temperature": 0, "num_predict": 20, "seed": 0}` | — | `temperature=0` : sortie quasi déterministe (essentiel pour de la classification) ; `num_predict` limite la longueur. |
| `format="json"` | — | Force une sortie JSON valide. |

```python
import ollama
def classification_zero_shot(texte):
    prompt = f"Give the sentiment of the following text: 'positive' or 'negative' :\n\n{texte}"
    return ollama.generate(model="phi3:latest", prompt=prompt, options={"temperature": 0}).response
```
- **Zero-shot** : consigne sans exemple ; **few-shot** : on ajoute quelques exemples dans le prompt.
- Pour comparer à des modèles supervisés : convertir la réponse en label (`"positive" in rep.lower()`) et mesurer l'accuracy sur un échantillon (`NB_DOC_MAX=1000`).

⚠️ Le modèle n'est **pas obligé de respecter** le format demandé (il peut répondre par une phrase) : parser de façon robuste, contraindre la sortie (`format="json"`, instructions strictes « Answer with one word ») et prévoir un cas « indéterminé ».

⚠️ Le serveur Ollama doit tourner (`ollama serve`) et le modèle être téléchargé (`ollama pull phi3`). La première requête est lente (chargement).

⚠️ Coût : un LLM de plusieurs milliards de paramètres est beaucoup plus lent qu'un petit Transformer fine-tuné pour de la classification massive ; tronquer les textes (`MAX_CHAR_SIZE=1000`).

---

## 📌 Récapitulatif des pièges classiques

| # | Piège | Réflexe |
|---|---|---|
| 1 | Softmax sur la mauvaise dimension | Softmax sur l'axe des **clés** (`dim=-1` pour `(B,L,L)`, `dim=1` pour `(B,N)`). |
| 2 | Oublier le masque de padding dans l'attention | `masked_fill(~mask, -inf)` avant le softmax ; `nan_to_num` si ligne vide. |
| 3 | Masque booléen inversé (`True`=garder vs `True`=ignorer) | PyTorch `src_key_padding_mask` : `True` = **ignoré** ; Hugging Face `attention_mask` : `1` = **gardé**. |
| 4 | `padding_value` différent de l'indice `PAD` | `pad_sequence(..., padding_value=PAD)`. |
| 5 | `batch_first=False` par défaut → `(L,B,d)` | Vérifier les formes ; `[CLS]` = `h[0]` ou `h[:,0]` selon la convention. |
| 6 | Oublier l'encodage de position | `nn.TransformerEncoder` n'en ajoute pas : sinusoïdal ou appris. |
| 7 | Oublier `/√d_k` | Sans normalisation, softmax saturé et gradients faibles. |
| 8 | Constante non enregistrée (position) | `register_buffer` pour qu'elle suive `.to(device)`. |
| 9 | `d_model` non divisible par `nhead` | Choisir `d_model = k × nhead`. |
| 10 | Texte > 512 tokens avec un modèle BERT | `truncation=True, max_length=512`. |
| 11 | Mauvais tokenizer pour le modèle | `AutoTokenizer.from_pretrained(<même nom que le modèle>)`. |
| 12 | `model(inputs)` au lieu de `model(**inputs)` | Dépaquetage du dict de tokenisation. |
| 13 | Inférence sans `eval()` / `no_grad()` | Les deux (dropout actif + graphe inutile sinon). |
| 14 | `Trainer` sans `compute_metrics` | Seule la loss est rapportée ; ajouter l'accuracy. |
| 15 | `lr` trop fort au fine-tuning | `2e-5` environ, peu d'epochs. |
| 16 | Reconstruction manuelle des poids d'attention : ordre Q/K/V et tête unique | `in_proj_weight = [Q; K; V]` ; valable seulement à 1 tête, en `eval()`. |
| 17 | Réponse d'un LLM mal formatée | `temperature=0`, instructions strictes, parsing tolérant. |

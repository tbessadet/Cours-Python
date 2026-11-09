# ⚡ Cheat Sheet PyTorch — Transformer « à la main » (`nn.TransformerEncoder`)

> Basée sur les notebooks `5_0_transf` et `5_1_transf_classif` (classification de sentiment IMDB).
> Convention : `import torch`, `from torch import nn`, `import torch.nn.functional as F`, `import math`

## Sommaire

1. [`nn.TransformerEncoderLayer` / `nn.TransformerEncoder`](#1-nntransformerencoderlayer--nntransformerencoder)
2. [Self-attention à la main et lecture des poids](#2-self-attention-à-la-main-et-lecture-des-poids)
3. [`nn.MultiheadAttention`](#3-nnmultiheadattention)
4. [Encodage de position (et `register_buffer`)](#4-encodage-de-position-et-register_buffer)
5. [Masques : padding et causal](#5-masques--padding-et-causal)
6. [Classifier avec un token CLS](#6-classifier-avec-un-token-cls)
7. [Préparer les données (tokenizer, CLS, padding)](#7-préparer-les-données-tokenizer-cls-padding)
8. [Régler l'architecture](#8-régler-larchitecture)
9. [Analyser le modèle](#9-analyser-le-modèle)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. `nn.TransformerEncoderLayer` / `nn.TransformerEncoder`

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False, norm_first=False)` | `d_model` = dimension des embeddings ; `nhead` (doit **diviser** `d_model`) ; `dim_feedforward` = couche cachée du FFN | Un bloc : self-attention multi-têtes + Add & Norm + FFN + Add & Norm. `norm_first=True` = Pre-LN (plus stable pour des réseaux profonds). `activation='gelu'` courant. |
| `nn.TransformerEncoder(encoder_layer, num_layers, norm=None)` | `num_layers` | Empile `num_layers` **copies** du bloc (poids indépendants). |
| `encoder(src, mask=None, src_key_padding_mask=None, is_causal=False)` | voir §5 | Sortie de **même forme** que l'entrée. |
| `nn.TransformerDecoderLayer` / `nn.Transformer` | 🔎 | Décodeur (attention causale + croisée) ; modèle encodeur-décodeur complet. |

| Tenseur | `batch_first=False` (défaut) | `batch_first=True` |
|---|---|---|
| `src` | `(L, B, d_model)` | `(B, L, d_model)` |
| `src_key_padding_mask` | **toujours** `(B, L)`, `True` = position à **ignorer** | idem |
| `mask` (attention) | `(L, L)` | `(L, L)` |

```python
encoder_layer = nn.TransformerEncoderLayer(d_model=3, nhead=1, dim_feedforward=6, activation='relu')
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
data = torch.randn(10, 8, 3)                  # (L=10, B=8, d=3)
out = transformer_encoder(data)               # (10, 8, 3) : nouvelles représentations contextuelles
```

> 💡 Chaque token **se déplace** dans l'espace des embeddings (tracer entrée vs sortie en 3D pour d=3) : c'est la « diffusion » de l'information entre tokens.

---

## 2. Self-attention à la main et lecture des poids

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```python
def self_attention(X, Wq, Wk, Wv, mask=None):          # X : (L, d)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / math.sqrt(K.size(-1))            # (L, L)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    alpha = F.softmax(scores, dim=-1)                   # chaque LIGNE (requête) somme à 1
    return alpha @ V, alpha
```

Lire les poids d'une couche `nn.TransformerEncoderLayer` (1 tête) :

| Attribut | Forme | Contenu |
|---|---|---|
| `layer.self_attn.in_proj_weight` | `(3·d, d)` | $W^Q$, $W^K$, $W^V$ **empilés dans cet ordre** (Q, K, V) |
| `layer.self_attn.in_proj_bias` | `(3·d,)` | biais correspondants |
| `layer.self_attn.out_proj` | `Linear(d, d)` | $W^O$ |

```python
W, b = layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
d = W.size(1)
QKV = data[:, i, :] @ W.T + b                     # séquence i du batch : (L, 3d)
Q, K = QKV[:, :d], QKV[:, d:2*d]                  # ⚠️ ordre Q puis K (le TP inverse les noms)
attn = F.softmax(Q @ K.T / math.sqrt(d), dim=1)   # (L, L)
attn.sum(1)                                       # = 1 sur chaque ligne (pas sur les colonnes)
plt.imshow(attn.detach().numpy()); plt.colorbar()
```

> ⚠️ Recalculé ainsi, la matrice **ignore les LayerNorm/dropout** et ne vaut que pour **une tête** ; pour plusieurs têtes, découper Q et K en `nhead` blocs de `d/nhead` colonnes. Plus simple : §3 (`need_weights=True`).

---

## 3. `nn.MultiheadAttention`

| Classe / appel | Paramètres essentiels | Explication |
|---|---|---|
| `nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=False)` | `embed_dim % num_heads == 0` | Bloc d'attention seul (sans FFN ni norm). |
| `attn(query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True, average_attn_weights=True)` | self-attention : `attn(x, x, x)` ; cross-attention : `attn(q_dec, k_enc, v_enc)` | Renvoie `(sortie, poids)` ; poids `(B, L_q, L_k)` (moyenne des têtes) ou `(B, h, L_q, L_k)` si `average_attn_weights=False`. |
| `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)` | 🔎 | Version fonctionnelle optimisée (Flash attention). |

```python
mha = nn.MultiheadAttention(embed_dim=128, num_heads=4)
out, w = mha(x, x, x, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=False)
# w : (B, 4, L, L) → visualiser la ligne du CLS : w[b, h, 0, :]
```

---

## 4. Encodage de position (et `register_buffer`)

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

```python
def generate_sinusoidal_embeddings(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)                                   # (L, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe                                                                       # (L, d)
```

| Choix | Code | Remarque |
|---|---|---|
| Sinusoïdal **fixe** | `self.register_buffer("posemb", pe.unsqueeze(1))` → `(maxlen, 1, d)` | **Pas de gradient**. `register_buffer` : le tenseur suit le modèle (`.to(device)`) et est sauvé dans le `state_dict`, sans être un paramètre. |
| **Appris** (BERT, ViT) | `self.pos = nn.Embedding(maxlen, d)` → `self.pos(torch.arange(L))` | Paramètres entraînés. |
| Ajout | `x = self.emb(input) + self.posemb[:L]` | **Somme** (pas concaténation) ; `[:L]` = longueur du batch. 🔎 Souvent `emb * sqrt(d)` avant d'ajouter. |

> ⚠️ Un simple attribut `self.pe = tensor` **ne suit pas** `model.to(device)` → erreur de device. Toujours `register_buffer` (ou `nn.Parameter` si appris).
> ⚠️ `maxlen` du buffer ≥ longueur max des séquences (sinon `posemb[:L]` est trop court).

---

## 5. Masques : padding et causal

| Masque | Construction | Argument |
|---|---|---|
| **Padding** | `pad_mask = (x == PAD).T` pour `x` `(L, B)` → `(B, L)` ; `True` = à ignorer | `src_key_padding_mask=pad_mask` |
| **Causal** (génération) | `nn.Transformer.generate_square_subsequent_mask(L)` (−inf au-dessus de la diagonale) | `mask=causal`, ou `is_causal=True` |
| Masque booléen manuel | `torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)` | `True` = interdit |

```python
padding_mask = (input == PAD).T                       # (B, L)
encoded = self.trans(xemb, src_key_padding_mask=padding_mask)
```

> ⚠️ Convention **inverse** de beaucoup de libs (Hugging Face `attention_mask` : 1 = à garder). En PyTorch : `True` = **masqué**.
> ⚠️ Le padding doit être masqué **aussi** dans tout pooling ultérieur (moyenne des tokens) ; avec le CLS en position 0, on lit simplement la position 0.

---

## 6. Classifier avec un token CLS

```python
class TransSent(nn.Module):
    def __init__(self, emb_size, voc_size, num_layers, num_heads, hidden_size_mlp, output_size, maxlen=1000):
        super().__init__()
        self.emb = nn.Embedding(voc_size, emb_size)
        layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads,
                                           dim_feedforward=hidden_size_mlp, activation='relu')
        self.trans = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.register_buffer("posemb", generate_sinusoidal_embeddings(maxlen, emb_size).unsqueeze(1))
        self.h2o = nn.Linear(emb_size, output_size)

    def forward(self, input, lengths=None):                       # input : (L, B), CLS en position 0
        L = input.size(0)
        padding_mask = (input == PAD).T                           # (B, L)
        xemb = self.emb(input) + self.posemb[:L]                  # (L, B, d)
        encoded = self.trans(xemb, src_key_padding_mask=padding_mask)
        return self.h2o(encoded[0])                               # représentation du CLS → (B, n_classes)

transf = TransSent(emb_size=128, voc_size=10000, num_layers=4, num_heads=4,
                   hidden_size_mlp=128, output_size=2)
optim = torch.optim.Adam(transf.parameters(), lr=5e-4)
```

| Agrégation | Code | Remarque |
|---|---|---|
| **CLS** | `encoded[0]` | Le CLS attend tous les tokens à chaque couche : il « résume » la séquence. |
| Moyenne masquée 🔎 | `(encoded * ~mask.T.unsqueeze(-1)).sum(0) / lengths.unsqueeze(-1)` | Souvent aussi bon, sans token spécial. |

---

## 7. Préparer les données (tokenizer, CLS, padding)

```python
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(vocab_size=10000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files, trainer)                      # liste de fichiers texte
PAD = tokenizer.token_to_id("[PAD]")

allcodes = [torch.tensor(tokenizer.encode("[CLS] " + t).ids) for t in txts]   # astuce : CLS en tête

def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=False, padding_value=PAD), lengths, torch.tensor(labels)
```

> ⚠️ `pad_sequence` remplit par **0** par défaut : si `[PAD]` n'a pas l'indice 0 (ici 3), passer **`padding_value=PAD`**, sinon `input == PAD` ne détecte rien et le masque est vide.
> 👉 Tronquer les textes (`MAX_CHAR_SIZE = 1000` caractères) : le coût de l'attention est en $O(L^2)$.

---

## 8. Régler l'architecture

| Hyper-paramètre | Valeurs du TP | Conseil |
|---|---|---|
| `emb_size` (`d_model`) | 128 | 128–512 ; divisible par `nhead` |
| `num_heads` | 1 → 4 | 4–8 ; plus de têtes = relations variées (dim par tête `d/h`) |
| `num_layers` | 1 → 4 | 2–6 pour un petit corpus |
| `dim_feedforward` | 128 | souvent 4 × `d_model` |
| `dropout` | 0.1 (défaut) | 0.1–0.3 |
| `lr` (Adam) | 5e-4 | 1e-4–5e-4 ; 🔎 **warmup** + décroissance (Transformers instables au début) |
| Vocabulaire | 10 000 | |
| Positions | sinusoïdales | comparer avec / sans (`TransSent-…-pos`) : sans position = sac de mots |

> 💡 Comparer les runs nommés `TransSent-1H-1L`, `-4H-4L`, `-25k-pos` dans TensorBoard pour mesurer l'effet des têtes, couches, taille de données et positions.
> 👉 `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` stabilise l'entraînement.

---

## 9. Analyser le modèle

| Analyse | Comment |
|---|---|
| Prédire un texte | `x = torch.tensor(tokenizer.encode("[CLS] " + txt).ids).unsqueeze(1)` → `model(x).softmax(-1)` |
| Erreurs | parcourir le test, afficher les textes où `argmax ≠ label` |
| Attention du CLS | `nn.MultiheadAttention(..., need_weights=True)` sur la dernière couche, ou recalcul §2 → ligne 0 de la matrice → mots qui pèsent sur la décision |
| Embeddings de mots | `writer.add_embedding(model.emb.weight, metadata=[tokenizer.id_to_token(i) for i in range(V)])` |

> 🔎 `nn.TransformerEncoderLayer` ne renvoie pas ses poids d'attention : passer par un *forward hook* sur `layer.self_attn` (avec `need_weights=True`) ou recalculer à la main.

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| `d_model % nhead != 0` | `AssertionError: embed_dim must be divisible by num_heads` | choisir des dimensions compatibles |
| `(B, L, d)` avec `batch_first=False` | attention calculée sur le batch, résultats absurdes | `(L, B, d)` ou `batch_first=True` |
| `src_key_padding_mask` en `(L, B)` | erreur de forme | `(B, L)` → `(x == PAD).T` |
| Masque avec `True` = garder | tokens utiles masqués | PyTorch : `True` = **ignorer** |
| `padding_value` ≠ indice `[PAD]` | masque vide, pad pris en compte | `pad_sequence(..., padding_value=PAD)` |
| Pas d'encodage de position | modèle insensible à l'ordre | ajouter `posemb[:L]` |
| PE stocké en simple attribut | `Expected all tensors to be on the same device` | `register_buffer` |
| `posemb` trop court | erreur de broadcast pour longues séquences | `maxlen` ≥ longueur max (ou tronquer) |
| Lire `encoded[:, 0]` avec `batch_first=False` | lit le 1er exemple au lieu du CLS | `encoded[0]` (position 0) |
| Ordre Q/K dans `in_proj_weight` | matrice transposée | ordre **Q, K, V** |
| Softmax par colonne | poids qui ne somment pas à 1 par requête | `dim=-1` sur `QKᵀ` |
| Séquences très longues | mémoire GPU saturée ($L^2$) | tronquer / batch plus petit |
| lr trop grand sans warmup | loss qui diverge / plateau | lr 1e-4–5e-4, warmup, clipping |

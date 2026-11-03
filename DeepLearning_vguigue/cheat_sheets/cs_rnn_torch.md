# 🔁 Cheat Sheet Réseaux récurrents (RNN / LSTM / GRU) en PyTorch — classification, génération, seq2seq, séries temporelles

> Basée sur les notebooks `4_1-rnn-gen`, `4_2-seq2seq-attention` (dossier `notebooks`) et `4_1_RNN-metro`, `4_2_char_rnn`, `4_3_rnn_classif_attn`, `4_4_token-generative` + `utils.py` (dossier `newRNN`).
> Conventions : `import torch`, `from torch import nn`, `from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence`, `from torch.utils.data import Dataset, DataLoader`.
> Prérequis : `cs_pytorch_fondamentaux.md`. Attention/Transformers : `cs_attention_transformers.md`.

## Sommaire
1. [Les modules récurrents : `nn.RNN`, `nn.LSTM`, `nn.GRU`](#1-les-modules-récurrents--nnrnn-nnlstm-nngru)
2. [Formes d'entrée/sortie et les 3 grands schémas (many-to-one, many-to-many, forecasting)](#2-formes-dentréesortie-et-les-3-grands-schémas)
3. [Données séquentielles : encodage, padding, `collate_fn`, pondération des classes](#3-données-séquentielles--encodage-padding-collate_fn-pondération-des-classes)
4. [RNN « from scratch » (cellule maison)](#4-rnn-from-scratch-cellule-maison)
5. [Many-to-one : classification (noms, sentiment)](#5-many-to-one--classification-noms-sentiment)
6. [Many-to-many : génération de texte et *teacher forcing*](#6-many-to-many--génération-de-texte-et-teacher-forcing)
7. [Tokenisation en sous-mots (BPE, SentencePiece, WordPiece)](#7-tokenisation-en-sous-mots-bpe-sentencepiece-wordpiece)
8. [Génération : glouton, échantillonnage, beam search, nucleus](#8-génération--glouton-échantillonnage-beam-search-nucleus)
9. [Seq2seq et attention encodeur-décodeur](#9-seq2seq-et-attention-encodeur-décodeur)
10. [Séries temporelles : classification et prévision (métro)](#10-séries-temporelles--classification-et-prévision-métro)
11. [Évaluation : matrice de confusion, ROUGE](#11-évaluation--matrice-de-confusion-rouge)
12. [📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

## 1. Les modules récurrents : `nn.RNN`, `nn.LSTM`, `nn.GRU`

Récurrence : l'état caché $h_t$ résume le passé et est mis à jour à chaque pas de temps avec **les mêmes poids**.

$$h_t=\tanh\!\big(W_x\,x_t+W_h\,h_{t-1}+b\big)\qquad(\text{RNN simple})$$

- $x_t\in\mathbb R^{d_{in}}$ : entrée au temps $t$ ; $h_t\in\mathbb R^{d_h}$ : état caché ; $W_x\in\mathbb R^{d_h\times d_{in}}$, $W_h\in\mathbb R^{d_h\times d_h}$ : matrices partagées sur tous les pas.

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False)` | `input_size` = dim de $x_t$ ; `hidden_size` = dim de $h_t$ ; `num_layers` = couches empilées ; `nonlinearity` ∈ `tanh`,`relu` ; `batch_first` ; `dropout` = entre couches (si `num_layers>1`) ; `bidirectional` = lit dans les 2 sens | RNN simple. Souffre de la **disparition/explosion du gradient** sur de longues séquences. |
| `nn.LSTM(…même paramètres, proj_size=0)` | idem | Ajoute une **cellule mémoire** $c_t$ et des **portes** (oubli, entrée, sortie) : retient l'information à long terme. Renvoie `(output, (h_n, c_n))`. |
| `nn.GRU(…même paramètres)` | idem | Version allégée du LSTM (2 portes : mise à jour et réinitialisation, pas de $c_t$). Souvent aussi bon, plus simple/rapide. Renvoie `(output, h_n)` comme le RNN. |

**Comment choisir** : RNN simple → pédagogie ou très courtes séquences ; **GRU** → bon compromis par défaut ; **LSTM** → longues dépendances, standard historique ; `bidirectional=True` pour la **classification** quand toute la séquence est disponible (jamais pour de la génération, qui ne doit pas voir le futur).

```python
rnn = nn.RNN(input_size=100, hidden_size=50, num_layers=1, nonlinearity='tanh')
x = torch.rand(23, 10, 100)          # (T, B, d_in) : temps D'ABORD (batch_first=False par défaut)
seq_h, last_h = rnn(x)               # seq_h : (23, 10, 50)  tous les états ; last_h : (1, 10, 50) dernier état
# LSTM : seq_h, (last_h, last_c) = lstm(x)
```

| Sortie | Forme | Contenu |
|---|---|---|
| `output` (= `seq_h`) | `(T, B, D·H)` (ou `(B, T, D·H)` si `batch_first=True`) | État caché **de la dernière couche à chaque pas** ($D=2$ si bidirectionnel). |
| `h_n` | `(D·num_layers, B, H)` | État caché **final de chaque couche** (et de chaque direction). Pour la dernière couche : `h_n[-1]`. |
| `c_n` (LSTM) | `(D·num_layers, B, H)` | État de la cellule mémoire final. |

⚠️ **`h_n` a une dimension de couche en plus** : `(1, B, H)` pour une couche → `h_n[-1]` ou `.squeeze(0)` avant un `Linear`. `output[-1]` n'est le dernier état **réel** que s'il n'y a pas de padding.

⚠️ **Par défaut le temps est la 1ʳᵉ dimension** `(T, B, d)` (contrairement aux CNN/MLP). Soit on respecte cette convention (c'est le choix du cours : `pad_sequence` sans `batch_first`), soit on passe `batch_first=True` dans le RNN **et** dans `pad_sequence`.

💡 L'état initial est nul par défaut ; on peut passer `rnn(x, h_0)` avec `h_0` de forme `(D·num_layers, B, H)` (utile pour **poursuivre une génération** ou initialiser un décodeur avec l'état de l'encodeur).

🔎 Pour de longues séquences, ajouter le **gradient clipping** : `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` entre `backward()` et `step()` (évite l'explosion du gradient).

---

## 2. Formes d'entrée/sortie et les 3 grands schémas

| Schéma | Entrée → sortie | Supervision | Exemples du cours | Perte |
|---|---|---|---|---|
| **Many-to-one** | séquence → 1 sortie | **dernier** pas de temps | nom → langue ; critique → sentiment ; série → station | `CrossEntropyLoss` sur `(B, classes)` |
| **Many-to-many (génération)** | séquence → séquence décalée | **à chaque** pas ($x_t\to x_{t+1}$) | génération de texte, modèle de langue | `CrossEntropyLoss(ignore_index=PAD)` sur `(T·B, V)` |
| **Many-to-many (prévision)** | série → série décalée | **à chaque** pas | flux de métro à $t+1$ | `MSELoss` sur `(T, B, d)` |

La rétropropagation (*BPTT*, dans le temps) se fait **après** avoir parcouru la séquence entière : à l'instant $t$, le gradient cumule l'erreur de $t$ et celles de tous les pas suivants.

Étapes d'un modèle récurrent (vision cours) : **Embedding** (indices → vecteurs) → **module récurrent** → **décodeur** (`nn.Linear(hidden, nb_sorties)`) appliqué soit au dernier état (many-to-one), soit à **tous** les états (many-to-many).

---

## 3. Données séquentielles : encodage, padding, `collate_fn`, pondération des classes

### Encodage caractères / indices
```python
all_letters = string.ascii_letters + " .,';_%"          # alphabet ; on réserve des caractères spéciaux
PAD = len(all_letters) - 1                               # indice du padding
def lineToTensor(line):  return torch.tensor([all_letters.find(c) for c in line], dtype=torch.long)   # (L,)
```
- Les séquences d'**indices entiers** (`int64`) sont le format universel : un `nn.Embedding` s'occupe de la représentation vectorielle (même pour un one-hot : initialiser `emb.weight` en identité et `requires_grad_(False)`).
- Nettoyage typique : `unicodedata.normalize('NFD', s)` + retirer les accents (catégorie `Mn`) pour réduire l'alphabet.

### Padding et `collate_fn` (temps d'abord)
```python
def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])                    # longueurs RÉELLES (avant padding)
    padded = pad_sequence(seqs, batch_first=False, padding_value=PAD) # (T_max, B)
    return padded, lengths, torch.tensor(labels)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
```

⚠️ **`padding_value` doit être l'indice de `<pad>`** : le `pad_sequence` de `4_3` et `5_1` n'a pas de `padding_value` (donc 0) alors que le code suppose `PAD ≠ 0` (indice `[PAD]`=3 dans le tokenizer WordPiece, 0 = `[UNK]` ou un vrai mot comme « the ») → incohérent. Toujours le passer explicitement.

### Pondération des classes déséquilibrées (noms : 9 000 russes vs 65 portugais…)
Deux options **complémentaires** :

| Outil | Paramètres | Explication |
|---|---|---|
| `WeightedRandomSampler(weights, num_samples, replacement=True)` (`torch.utils.data`) | `weights` = poids **par exemple** (taille = nb d'exemples) | Rééchantillonne les exemples : classes rares tirées plus souvent. Passer `sampler=` (et **pas** `shuffle=True`) au `DataLoader`. |
| `nn.CrossEntropyLoss(weight=cl_weight)` | `weight` = tenseur de taille `n_classes` (sur le bon device) | Pénalise davantage les erreurs sur les classes rares, sans changer les données. |

```python
counts = torch.bincount(torch.tensor(train_labels))              # nb d'exemples par classe
class_w = 1.0 / counts.float()                                    # inverse de la fréquence
sample_w = class_w[torch.tensor(train_labels)]                    # poids de chaque exemple
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, collate_fn=collate_fn)
```
⚠️ Sampler **et** `weight=` ensemble : double correction → classes rares sur-pondérées. Choisir l'un ou l'autre (ou doser).

⚠️ Sur un jeu déséquilibré, l'**accuracy** est trompeuse ; utiliser la **matrice de confusion** normalisée par ligne (§11) ou le F1 macro.

---

## 4. RNN « from scratch » (cellule maison)

Le fichier `utils.py` implémente la récurrence à la main (utile pour comprendre ; dans la pratique, `nn.RNN` est bien plus rapide car optimisé).

```python
class RNN(nn.Module):
    def __init__(self, inputdim, latentdim):
        super().__init__()
        self.inputdim, self.latentdim = inputdim, latentdim
        self.encoder = nn.Linear(inputdim, latentdim)       # W_x x + b
        self.latent  = nn.Linear(latentdim, latentdim)      # W_h h
    def one_step(self, x, h):
        return torch.tanh(self.encoder(x.view(-1, self.inputdim)) + self.latent(h))   # h_t
    def hzero(self, batch_size):  return torch.zeros(batch_size, self.latentdim)
    def forward(self, x, h=None):                           # x : (T, B, d)
        if h is None: h = self.hzero(x.shape[1]).to(x.device)
        hseq = []
        for t in range(x.shape[0]):                         # boucle sur le TEMPS
            h = self.one_step(x[t], h); hseq.append(h)
        return torch.stack(hseq)                            # (T, B, latent) : tous les états
```
⚠️ L'état initial `hzero` doit être créé **sur le même device** que `x` (`.to(x.device)`), sinon erreur CPU/GPU.

---

## 5. Many-to-one : classification (noms, sentiment)

Principe : on prend l'état caché du **dernier mot réel** de chaque séquence et on le décode.

```python
class RNNSent(nn.Module):
    def __init__(self, hidden_size, output_size, embeddings):       # embeddings : matrice numpy GloVe
        super().__init__()
        self.emb = nn.Embedding(len(embeddings), len(embeddings[0]))
        self.emb.weight.data.copy_(torch.tensor(embeddings, dtype=torch.float))
        self.emb.weight.requires_grad = False                       # embeddings figés
        self.rec = nn.LSTM(len(embeddings[0]), hidden_size)         # ou nn.RNN(..., nonlinearity='tanh')
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):                                  # x : (T, B)
        out, _ = self.rec(self.emb(x))                              # (T, B, H)
        last = out[lengths - 1, torch.arange(x.size(1))]            # dernier état RÉEL de chaque séquence -> (B, H)
        return self.h2o(last), out
```

| Méthode pour gérer le padding | Explication |
|---|---|
| **Indexer le bon pas** : `out[lengths-1, arange(B)]` | Simple : on lit l'état au dernier mot réel (le TD utilise `hidden[lengths[i]-1, i]`). `lengths` doit être un tenseur d'entiers. |
| **`pack_padded_sequence`** | La méthode propre : le RNN ignore complètement le padding. |
| **Masque** | Obligatoire avec attention : on met `-inf` sur les scores du padding (voir ci-dessous). |

```python
packed = pack_padded_sequence(emb_x, lengths.cpu(), enforce_sorted=False)   # lengths : CPU, int64
out_packed, (h_n, c_n) = lstm(packed)
out, _ = pad_packed_sequence(out_packed)                                     # retour à (T, B, H)
last = h_n[-1]                                                                # (B, H) : dernier état réel automatiquement
```

### Ajouter de l'attention (pooling pondéré appris)
Au lieu du dernier état, on calcule une **somme pondérée** de tous les états cachés :
$$a_t=\mathrm{softmax}_t\big(v^\top h_t\big),\qquad c=\sum_t a_t\,h_t$$

- $h_t$ : états cachés ; $v$ : vecteur appris (`nn.Linear(H, 1)`) ; $a_t$ : poids d'attention (somme = 1) ; $c$ : vecteur de contexte envoyé au décodeur.

```python
scores = self.att(out).squeeze(-1)                                    # (T, B)
mask = torch.arange(T).unsqueeze(1).expand(T, B) < lengths.unsqueeze(0)    # True = vrai mot, (T, B)
scores = scores.masked_fill(~mask, float('-inf'))                     # le padding reçoit un poids nul
a = torch.softmax(scores, dim=0)                                      # softmax sur le TEMPS (dim 0)
context = (a.unsqueeze(-1) * out).sum(0)                              # (B, H)
```

⚠️ **Masque indispensable avec attention** : sans lui, le padding reçoit du poids. `masked_fill(~mask, -inf)` **avant** le softmax. Softmax sur la bonne dimension (`dim=0` si `(T,B)`).

⚠️ Pour un `nn.LSTM`, le dernier état est `h_n[-1]` ; ne pas utiliser `out[-1]` si les séquences sont paddées. `nn.RNN` renvoie `(out, h_n)`, `nn.LSTM` `(out, (h_n, c_n))` : adapter le dépaquetage.

⚠️ Dans la boucle : `model(x, lengths)`, `lengths` reste sur CPU si `pack_padded_sequence`. Sortie `(B, classes)` → `CrossEntropyLoss` (pas de `squeeze` fantaisiste).

---

## 6. Many-to-many : génération de texte et *teacher forcing*

**Modèle de langue** : à chaque pas, prédire le token suivant. L'entrée est la séquence décalée de la cible :

```
séquence :  <bos>  w1   w2   w3   <eos>
entrée x :  <bos>  w1   w2   w3          x = seq[:-1]
cible  y :         w1   w2   w3   <eos>  y = seq[1:]
```
En **teacher forcing**, on fournit au modèle le *vrai* token précédent (et non sa propre prédiction) pendant l'entraînement.

```python
class RNNGen(nn.Module):
    def __init__(self, embeddings, hidden_dim):
        super().__init__()
        self.embeddings = embeddings                                   # nn.Embedding(V, d)
        self.rnn = nn.GRU(embeddings.embedding_dim, hidden_dim)        # ou LSTM
        self.fc = nn.Linear(hidden_dim, embeddings.num_embeddings)     # décodeur : H -> taille du vocabulaire V
    def forward(self, x, h_0=None):                                    # x : (T, B)
        out, h = self.rnn(self.embeddings(x), h_0)                     # TOUS les états
        return self.fc(out), h                                         # (T, B, V), dernier état (pour poursuivre)

# entraînement
x, y = seq[:-1], seq[1:]
yhat, _ = model(x)
loss = nn.CrossEntropyLoss(ignore_index=PAD)(yhat.view(-1, yhat.size(2)), y.reshape(-1))
```

| Élément | Explication |
|---|---|
| `yhat.view(-1, V)` + `y.reshape(-1)` | Aplatit `(T, B, V)` → `(T·B, V)` et `(T, B)` → `(T·B,)` : format attendu de `CrossEntropyLoss` (classe en 2ᵉ axe, cible en indices). |
| `ignore_index=PAD` | Les positions où la cible est `PAD` ne contribuent **ni à la perte ni au gradient**. |
| Perte masquée « à la main » | `mask = (y != PAD); loss = (l_par_token * mask).sum() / mask.sum()` avec `CrossEntropyLoss(reduction='none')`. |

⚠️ **Piège du TD `4_4`** : `(mask * CrossEntropyLoss()(out, y)).sum() / mask.sum()` ne masque rien : `CrossEntropyLoss()` renvoie déjà un **scalaire** (moyenne). Il faut `reduction='none'` pour obtenir une perte par token, puis multiplier par le masque. Plus simple : `ignore_index`.

⚠️ Le décodeur donne des **logits sur tout le vocabulaire** (`dim = V`) ; pas de softmax avant la loss. Un grand vocabulaire = couche de sortie énorme (d'où les tokens en sous-mots, §7).

⚠️ Dans le notebook `4_1`, la boucle de validation utilise `batch["summary"]` du *train* au lieu de `x` : toujours évaluer sur le `test_loader`.

💡 Réduire le coût : séquences courtes, vocabulaire réduit, batchs élevés (le RNN est séquentiel en $T$).

---

## 7. Tokenisation en sous-mots (BPE, SentencePiece, WordPiece)

Le **mot** pose problème (vocabulaire énorme, mots inconnus) ; le **caractère** donne des séquences très longues. Compromis : **sous-mots** (*subword units*), vocabulaire de taille fixe couvrant au mieux le corpus (`You shoulda got David Carr` → `_You`, `_should`, `a`, `_got`, `_D`, `av`, `id`, …). Les mots fréquents restent entiers, les rares sont découpés.

| Bibliothèque | Appel | Explication |
|---|---|---|
| **SentencePiece** | `spm.SentencePieceTrainer.train(sentence_iterator=…, model_prefix=…, vocab_size=1000, pad_id=0, unk_id=1, bos_id=2, eos_id=3)` puis `sp = spm.SentencePieceProcessor(); sp.load("….model")` | Entraîne un modèle de segmentation. `sp.EncodeAsIds(s)`, `sp.encode_as_pieces(s)`, `sp.Decode(ids)`, `sp.vocab_size()`. `sp.SetEncodeExtraOptions("bos:eos")` ajoute `<s>`/`</s>` automatiquement. |
| **HuggingFace `tokenizers`** (WordPiece) | `Tokenizer(WordPiece(unk_token="[UNK]"))`, `tokenizer.pre_tokenizer = Whitespace()`, `WordPieceTrainer(vocab_size=10000, special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"])`, `tokenizer.train([fichier], trainer)` | `tokenizer.encode(s).ids / .tokens`, `tokenizer.decode(ids)`, `tokenizer.get_vocab_size()`. Les suites de sous-mots sont préfixées par `##` (retirés au décodage). |
| **`BertTokenizerFast.from_pretrained('bert-base-uncased', bos_token="<bos>", eos_token="<eos>")`** | `tok(batch, truncation=True, padding=True, max_length=512, add_special_tokens=False, return_tensors="pt")` | Vocabulaire BERT pré-entraîné (30 000) ; renvoie un dict avec `input_ids`. |

**Tokens spéciaux** : `<pad>` (remplissage), `<unk>`/`<oov>` (inconnu), `<bos>`/`<s>` (début), `<eos>`/`</s>` (fin → arrêt de la génération), `[CLS]` (agrégat de la séquence pour la classification), `[SEP]`, `[MASK]`.

```python
ids = tokenizer.encode("[CLS] " + texte).ids              # astuce : préfixer [CLS] à la main
PAD = tokenizer.encode("[PAD]").ids[0]; EOS = tokenizer.encode(".").ids[0]   # indices des spéciaux (ici '.' sert d'EOS)
batch_tensor = tokenizer_hf(batch, return_tensors="pt", padding=True)["input_ids"].T.contiguous()   # (B,T) -> (T,B)
```

⚠️ **Format temps-d'abord** : les tokenizers HuggingFace renvoient `(B, T)` ; pour un RNN par défaut il faut `.T.contiguous()` (le `contiguous` évite des erreurs de `view`).

⚠️ La tokenisation reflète le corpus d'entraînement : un texte hors domaine (ex. du français avec un tokenizer anglais) est fragmenté en sous-mots (« Exemple » → `Ex`, `##emp`, `##le`) → séquences plus longues et moins bonnes performances. Les **inégalités entre langues** et le vocabulaire technique en découlent.

⚠️ Ne pas oublier `<bos>`/`<eos>` autour des phrases (indispensables pour générer et savoir quand s'arrêter). Le tokenizer doit être **sauvegardé avec le modèle** (`tokenizer.save("tok.json")`), sinon les indices ne correspondent plus.

---

## 8. Génération : glouton, échantillonnage, beam search, nucleus

Une fois entraîné, on génère **pas à pas** : état initial = traitement du début de phrase (*prompt*), puis à chaque pas on choisit un token, on le réinjecte en entrée avec l'état précédent.

```python
@torch.no_grad()
def generate(model, tokenizer, start="it is", maxlen=50, eos=EOS):
    x = torch.tensor(tokenizer.encode(start).ids).unsqueeze(1)          # (T, 1) : batch de 1
    y, h = model(x)                                                      # on traite tout le prompt -> état h
    out = x.squeeze(1).tolist()
    for _ in range(maxlen):
        logits = y[-1, 0]                                                # distribution du prochain token
        w = torch.distributions.Categorical(logits=logits).sample()     # ou logits.argmax()
        if w.item() == eos: break
        out.append(w.item())
        y, h = model(w.view(1, 1), h)                                    # réinjecter UN token + l'état h (unsqueeze !)
    return tokenizer.decode(out)
```

| Stratégie | Principe | Pour / contre |
|---|---|---|
| **Glouton** (`argmax`) | Token le plus probable à chaque pas | Déterministe mais répétitif, rarement la meilleure séquence globale. |
| **Échantillonnage** (`Categorical(logits=…).sample()` ou `torch.multinomial(softmax(y), 1)`) | Tirage selon la distribution | Diversité ; mais peut piocher des tokens très improbables. |
| **Température** 🔎 `softmax(logits / T)` | $T<1$ : plus conservateur ; $T>1$ : plus créatif | Réglage simple de la diversité. |
| **Beam search** | Garde les $K$ meilleures séquences partielles (log-probabilités) ; à chaque pas, chacune est prolongée par ses $K$ meilleurs tokens, puis on garde les $K$ meilleures parmi les $K^2$ | Approche l'argmax **de la séquence entière** ; bien meilleur que le glouton. |
| **Nucleus (top-p)** | Échantillonne seulement dans le plus petit ensemble de tokens dont la masse de probabilité $\ge\alpha$ (ex. 0,95), renormalisé | Évite la queue de tokens absurdes tout en gardant de la diversité. |

**Nucleus sampling** — hypothèse : $\alpha\in(0,1]$ ; $I_\alpha(p,s)$ est le plus petit ensemble de symboles tel que $\sum_{y\in I_\alpha}p(y\mid s)\ge\alpha$.
$$p_{\text{nucleus}}(y\mid s_t)=\mathbf 1_{y\in I_\alpha(p,s_t)}\;\frac{p(y\mid s_t)}{\sum_{y'\in I_\alpha(p,s_t)}p(y'\mid s_t)}$$

- $s_t$ : état/contexte au pas $t$ ; $p(y\mid s_t)$ : probabilité du token $y$ ; $\mathbf 1$ : indicatrice (1 si $y$ est dans le noyau, 0 sinon).

**Normalisation de longueur** (beam search) — le score $\sum\log p$ pénalise les longues phrases ; on divise par :
$$lp(L)=\left(\frac{\text{base}+L}{\text{base}+1}\right)^{\alpha}\quad(\text{base}=5,\ \alpha=0{,}7)\qquad\text{score}=\frac{\sum_t\log p(w_t)}{lp(L)}$$
(écrit dans le TD `math.exp(alpha*(log(base+L) - log(base+1)))`, identique).

```python
def generate_beam(model, tokenizer, eos, k, start="", maxlen=50):
    x = torch.tensor(tokenizer.encode(start).ids).unsqueeze(1)
    y, h = model(x)
    beams = [(h, x, y[-1], 0.0, False)]                              # (état, tokens, logits, logprob, terminé)
    for _ in range(maxlen):
        cand = []
        for h, x, y, lp, done in beams:
            if done: cand.append((h, x, y, lp, True, None)); continue
            vals, idx = torch.log_softmax(y, dim=-1).topk(k)         # k meilleurs tokens (log-probas)
            for v, c in zip(vals.view(-1), idx.view(-1)):
                cand.append((h, x, y, lp + v.item(), False, c))
        cand.sort(key=lambda t: -t[3]); beams = []
        for h, x, y, lp, done, c in cand[:k]:                         # on garde les k meilleurs
            if c is None: beams.append((h, x, y, lp, True)); continue
            x2 = torch.cat([x, c.view(1, 1)], 0)
            y2, h2 = model(c.view(1, 1), h)                           # prolonger d'UN token avec l'état précédent
            beams.append((h2, x2, y2[-1], lp, c.item() == eos))
    return tokenizer.decode(beams[0][1].squeeze(1).tolist())
```

⚠️ Pendant la génération : `torch.no_grad()` + `model.eval()`, et **toujours `unsqueeze`** le token pour avoir `(T=1, B=1)`. Penser à réinjecter l'état `h` (sinon le modèle « oublie » le contexte).

⚠️ Un token `eos` déjà généré doit **geler** la séquence dans le beam (sinon elle continue). Comparer des séquences de longueurs différentes sans normalisation favorise les phrases trop courtes.

💡 Beam search ≈ meilleure **qualité/fidélité** (traduction, résumé) ; échantillonnage/nucleus ≈ meilleure **diversité/créativité** (génération libre). Pour chaque `k`, plus il est grand plus c'est coûteux (×k).

---

## 9. Seq2seq et attention encodeur-décodeur

**Seq2seq** : un **encodeur** résume la séquence source en un état ; un **décodeur** (paramètres distincts) génère la cible à partir de cet état (traduction, résumé, légendage). Le tout est entraîné en bout-en-bout en teacher forcing.

```python
class Seq2Seq(nn.Module):
    def __init__(self, name, encoder, decoder, classifier):
        super().__init__(); self.name, self.encoder, self.decoder, self.classifier = name, encoder, decoder, classifier
    def forward(self, src_ids, tgt_ids):
        enc_emb, enc_out, hidden = self.encoder(src_ids)                          # état final de l'encodeur
        _, out, hidden = self.decoder(tgt_ids, hidden,                            # = h_0 du décodeur
                                      encoder_embeddings=enc_emb, encoder_outputs=enc_out)
        return self.classifier(out), hidden, enc_emb, enc_out                     # logits (T, B, V)

loss = nn.CrossEntropyLoss(ignore_index=pad_id)
yhat, *_ = model(src, tgt[:-1])                                                   # décodeur reçoit tgt décalé
l = loss(yhat.view(-1, yhat.size(2)), tgt[1:].reshape(-1))
```
Encodeur et décodeur partagent la même interface `forward(x, h_0, *, encoder_outputs, encoder_embeddings)` → renvoie `(embeddings, sorties, état final)` (classe de base `RNNBase`).

### Attention sur l'encodeur
Sorties de l'encodeur $o_1..o_N$ (temps × batch × latent) et sorties du décodeur $z_1..z_M$. Hypothèse : clés et questions de même dimension $d$.
1. Clés : $k_i=W_k\,o_i$ (`nn.Linear`) ; questions : $q_m=W_q\,z_m$.
2. Poids : $p(i\mid m)=\dfrac{\exp(k_i\cdot q_m)}{\sum_{i'}\exp(k_{i'}\cdot q_m)}$ (softmax sur les positions sources $i$).
3. Sortie enrichie : $z'_m=z_m+\sum_{i=1}^{N}p(i\mid m)\,v(x_i)$, où $v$ = transformation (identité si dimensions compatibles) et $x_i$ = embeddings d'entrée de l'encodeur.

- $o_i$ : sortie de l'encodeur au pas $i$ ; $z_m$ : sortie du décodeur au pas $m$ ; $p(i\mid m)$ : « le token cible $m$ regarde le token source $i$ » ; $\sum_i p(i\mid m)=1$ (**combinaison convexe**).

⚠️ Ajouter aux sorties du décodeur une combinaison des **embeddings sources** permet de recopier facilement des mots du texte source (utile en résumé).

⚠️ L'attention d'un décodeur sur la source ne doit pas être masquée sur le futur de la **cible** (c'est la source entière qui est visible), mais le padding de la **source** doit être masqué (softmax avec `-inf`).

### Évaluation du résumé : ROUGE
Compare texte généré et référence par recouvrement : ROUGE-1 (unigrammes), ROUGE-2 (bigrammes), ROUGE-L (plus longue sous-séquence commune). `rouge.add_batch(predictions=…, references=…)` puis `rouge.compute()` (bibliothèque `evaluate`/`datasets.load_metric("rouge")`, ou `rouge-score`). 🔎 `load_metric` est retiré des versions récentes de `datasets` → `evaluate.load("rouge")`.

💡 Dans `4_2`, le modèle sans attention produit un texte quasi aléatoire au départ ; avec attention, la perte train/val diminue et la qualité s'améliore.

---

## 10. Séries temporelles : classification et prévision (métro)

Données Hangzhou : tenseur `(jours, créneaux=73, stations=80, 2)` (flux entrant/sortant par quart d'heure, 5h30–23h30). `SampleMetroDataset` extrait des fenêtres de `length` pas ; **normalisation** par le max de chaque station (`self.data / self.stations_max`), avec les stats **du train réutilisées sur le test** (`stations_max=ds_train.stations_max`).

### Classification : « de quelle station vient cette série ? » (many-to-one)
```python
rnn = RNN(dim_input=2, latent=10)                    # RNN maison de utils.py : renvoie (T, B, latent)
decoder = nn.Linear(10, n_stations)
optim = torch.optim.Adam(chain(rnn.parameters(), decoder.parameters()), lr=1e-4)   # 2 modules -> chain
h = rnn(x)[-1]                                       # dernier état (B, latent)  [x : (T,B,2) -> utiliser x.transpose(0,1) si besoin]
loss = nn.CrossEntropyLoss()(decoder(h), y)
```
- `itertools.chain(a.parameters(), b.parameters())` : un seul optimiseur pour plusieurs modules.
- Choisir **où brancher le décodeur** : sur le dernier état (many-to-one).

### Prévision (*forecasting*) : prédire $t+1$, $t+2$, … (many-to-many, régression)
- Dataset `ForecastMetroDataset` : renvoie `(x[t : t+L-1], x[t+1 : t+L])` → cible = entrée **décalée d'un pas**.
- Décodeur : `nn.Linear(latent, dim_input)` appliqué à **tous** les états ; perte **MSE** (régression, pas de cross-entropie, sortie réelle).
- **Prévision multi-pas** : on réinjecte la sortie prédite comme prochaine entrée (*autorégressif*).

```python
def forecast(rnn, decoder, x, h=None, length=10):
    with torch.no_grad():
        h = rnn.hzero(x.size(1)).to(x.device) if h is None else h
        h = rnn(x, h)[-1]                                  # résumé de la série observée
        x = decoder(h); preds = [x]
        for _ in range(length - 1):
            h = rnn.one_step(x, h)                         # un pas avec la prédiction précédente
            x = decoder(h); preds.append(x)
    return torch.stack(preds)                              # (length, B, dim)
```

⚠️ Fuite de données : normaliser avec les statistiques du **train** (pas du test). Les erreurs s'accumulent avec l'horizon : la MSE croît quand on prédit plus loin ; comparer à la baseline « copier la dernière valeur » ou « valeur du créneau précédent ».

⚠️ Dimensions `(T, B, d)` vs `(B, T, d)` : le `DataLoader` renvoie `(B, T, d)` ; le RNN par défaut attend `(T, B, d)` → `x.transpose(0, 1)` (ou `batch_first=True`).

---

## 11. Évaluation : matrice de confusion, ROUGE

```python
confusion = torch.zeros(n_classes, n_classes)
with torch.no_grad():
    for x, l, y in loader:
        guess = model(x.to(device), l)[0].argmax(1).cpu()
        for t, g in zip(y, guess): confusion[t, g] += 1               # ligne = vraie classe, colonne = prédite
confusion = confusion / confusion.sum(1, keepdim=True)                # normalisation par LIGNE (taux de rappel)
plt.matshow(confusion.numpy())
```
- Cases hors diagonale = confusions (ex. coréen ↔ chinois, italien ↔ espagnol). 🔎 `sklearn.metrics.confusion_matrix` / `classification_report` donnent aussi précision, rappel, F1.
- **Top-k** pour les noms : `yhat.topk(5).indices` (langues les plus probables).
- Évaluer **sur le test** (pas sur le train comme dans la cellule du TD qui boucle sur `train_loader`).

---

## 📌 Récapitulatif des pièges classiques

| # | Piège | Réflexe |
|---|---|---|
| 1 | Dimension temporelle : `(T,B,d)` par défaut ≠ `(B,T,d)` | Choisir une convention ; `batch_first=True` partout ou `.transpose(0,1)`. |
| 2 | `h_n` a une dimension de couche : `(layers·D, B, H)` | `h_n[-1]` pour le dernier état de la dernière couche. |
| 3 | Dernier état pris sur du padding (`out[-1]`) | Indexer avec `lengths-1` ou `pack_padded_sequence`. |
| 4 | `pad_sequence` sans `padding_value` correct | `padding_value=PAD` explicitement. |
| 5 | Perte qui compte le padding | `CrossEntropyLoss(ignore_index=PAD)` ou `reduction='none'` + masque (pas de `×masque` sur un scalaire). |
| 6 | `(T,B,V)` mal aplati pour la perte | `view(-1, V)` pour les logits, `reshape(-1)` pour les cibles. |
| 7 | Softmax dans le modèle avec `CrossEntropyLoss` | Logits bruts ; softmax seulement à la génération. |
| 8 | Attention sans masque | `masked_fill(~mask, -inf)` avant le softmax, softmax sur l'axe du temps. |
| 9 | Génération : oublier `no_grad`, `eval`, `unsqueeze`, ou de réinjecter l'état `h` | Un token `(1,1)` + l'état à chaque pas. |
| 10 | Beam search sans normalisation de longueur / séquence terminée non gelée | Diviser par $lp(L)$ ; marquer `eos` comme terminé. |
| 11 | `bidirectional=True` pour de la génération | Réservé à la classification/encodage (il voit le futur). |
| 12 | Classes déséquilibrées + accuracy | `WeightedRandomSampler` **ou** `weight=` ; matrice de confusion. |
| 13 | Deux modules (RNN + décodeur) avec un seul `parameters()` | `chain(rnn.parameters(), dec.parameters())`. |
| 14 | `h_0` créé sur CPU alors que le modèle est sur GPU | `.to(x.device)`. |
| 15 | Tokenizer non sauvegardé avec le modèle / `<bos>`,`<eos>` oubliés | Sauvegarder les deux ensemble ; ajouter les tokens de début/fin. |
| 16 | Gradients qui explosent (RNN/LSTM) | `clip_grad_norm_(…, 1.0)`, GRU/LSTM, `lr` plus faible. |
| 17 | Évaluer sur le train par erreur (boucles de validation recopiées) | Vérifier le `loader` utilisé dans le bloc `eval`. |

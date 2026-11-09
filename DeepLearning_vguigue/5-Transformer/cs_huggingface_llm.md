# 🤗 Cheat Sheet — Modèles pré-entraînés : Hugging Face & LLM locaux (Ollama)

> Basée sur les notebooks `5_2_transf_pretrained`, `5_3_transf_pretrained-kmeans`, `5_4_transf_API`.
> Convention : `from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline, Trainer, TrainingArguments`, `from datasets import Dataset, DatasetDict, load_dataset`

## Sommaire

**I. Hugging Face `transformers`**
1. [Charger un modèle et son tokenizer](#1-charger-un-modèle-et-son-tokenizer)
2. [Tokeniser](#2-tokeniser)
3. [Inférence : sorties du modèle](#3-inférence--sorties-du-modèle)
4. [`pipeline` : l'usage en une ligne](#4-pipeline--lusage-en-une-ligne)
5. [Données : `datasets`](#5-données--datasets)
6. [Fine-tuning avec `Trainer`](#6-fine-tuning-avec-trainer)
7. [Extraire des embeddings (CLS) + clustering](#7-extraire-des-embeddings-cls--clustering)

**II. LLM locaux avec Ollama**

8. [Interroger un LLM : `chat`, `generate`, `embed`](#8-interroger-un-llm--chat-generate-embed)
9. [Classification zero-shot par prompt](#9-classification-zero-shot-par-prompt)

[10. Choisir la bonne approche](#10-choisir-la-bonne-approche)

[📌 Récapitulatif des pièges classiques](#-récapitulatif-des-pièges-classiques)

---

# I. Hugging Face `transformers`

## 1. Charger un modèle et son tokenizer

| Classe | Paramètres essentiels | Explication |
|---|---|---|
| `AutoTokenizer.from_pretrained(nom)` | nom du *Hub* ou dossier local | Le tokenizer **exact** du modèle (même vocabulaire, mêmes tokens spéciaux). |
| `AutoModel.from_pretrained(nom)` | | Le **corps** du modèle (encodeur) : renvoie les états cachés, pas de tête de tâche. Pour des **embeddings**. |
| `AutoModelForSequenceClassification.from_pretrained(nom, num_labels=2)` | `num_labels`, `id2label`, `label2id` | Corps + **tête de classification** sur le CLS. Si le modèle n'a pas été entraîné pour cette tâche, la tête est **initialisée aléatoirement** (warning) → à fine-tuner. |
| `AutoModelForTokenClassification`, `…ForQuestionAnswering`, `…ForCausalLM`, `…ForSeq2SeqLM` | | Têtes pour NER, QA extractive, génération (GPT), seq2seq (T5). |
| `model.save_pretrained(dir)` / `tokenizer.save_pretrained(dir)` | | Sauver **les deux** ensemble. |

| Modèle (TP) | Rôle |
|---|---|
| `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT **déjà fine-tuné** en sentiment (SST-2) : utilisable directement. |
| `distilbert-base-uncased` | DistilBERT **générique** (pré-entraîné seulement) : embeddings ou base de fine-tuning. |

> 💡 *uncased* = texte mis en minuscules ; *distil* = version compressée (≈ 40 % plus petite que BERT, ≈ 97 % des performances).

---

## 2. Tokeniser

| Appel | Paramètres essentiels | Explication |
|---|---|---|
| `tokenizer(textes, padding=True, truncation=True, max_length=512, return_tensors="pt")` | `padding=True` (au plus long du batch) ou `"max_length"` ; `truncation=True` (**indispensable** au-delà de 512 tokens) ; `return_tensors="pt"` (tenseurs PyTorch) | Renvoie un dict : `input_ids`, `attention_mask` (**1 = vrai token, 0 = padding**), parfois `token_type_ids`. Ajoute automatiquement `[CLS]` … `[SEP]`. |
| `tokenizer.tokenize(txt)` / `convert_ids_to_tokens(ids)` | | Voir le découpage en sous-mots (`##`). |
| `tokenizer.decode(ids, skip_special_tokens=True)` | | Ids → texte. |
| `tokenizer(a, b)` | paire de textes | `[CLS] a [SEP] b [SEP]` (similarité, QA, NLI). |

```python
inputs = tokenizer(["I love this product!", "Worst experience ever."],
                   padding=True, truncation=True, return_tensors="pt")
inputs["input_ids"].shape          # (2, L_max)
```

> ⚠️ `padding="max_length"` sans `max_length` pad à 512 : lent. Préférer le **padding dynamique** (`DataCollatorWithPadding`, §6).

---

## 3. Inférence : sorties du modèle

```python
model.eval()
with torch.no_grad():
    outputs = model(**inputs)                      # ** : déballe input_ids, attention_mask...
logits = outputs.logits                            # (B, num_labels) : scores bruts
probas = torch.softmax(logits, dim=-1)
preds = probas.argmax(-1)
labels = [model.config.id2label[p.item()] for p in preds]   # 'POSITIVE' / 'NEGATIVE'
```

| Sortie | Modèle | Forme |
|---|---|---|
| `outputs.logits` | `…ForSequenceClassification` | `(B, num_labels)` |
| `outputs.last_hidden_state` | `AutoModel` | `(B, L, hidden)` (**batch first**) — 768 pour BERT/DistilBERT |
| `outputs.hidden_states` | avec `output_hidden_states=True` | tuple : une sortie par couche |
| `outputs.attentions` | avec `output_attentions=True` | tuple `(B, h, L, L)` par couche → visualisation (bertviz) |

> 👉 Sur GPU : `model.to(device)` et `{k: v.to(device) for k, v in inputs.items()}`. Pour beaucoup de textes, traiter **par batchs** (pas tout d'un coup, pas un par un).

---

## 4. `pipeline` : l'usage en une ligne

| Appel | Paramètres essentiels | Explication |
|---|---|---|
| `pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)` | tâche ; `model` (nom ou objet) ; `device` (-1 CPU, 0 GPU) | Tokenisation + modèle + post-traitement. Renvoie `[{'label': 'POSITIVE', 'score': 0.99}, ...]`. |
| Autres tâches | `"text-classification"`, `"zero-shot-classification"`, `"ner"`, `"question-answering"`, `"summarization"`, `"translation_en_to_fr"`, `"text-generation"`, `"feature-extraction"`, `"fill-mask"` | 🔎 |
| Appel sur une liste | `classifier(textes, batch_size=32, truncation=True)` | Traitement par batch, troncature des longs textes. |

```python
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier(txts_test[:200], truncation=True, batch_size=32)
preds = [1 if r["label"] == "POSITIVE" else 0 for r in results]
acc = np.mean(np.array(preds) == np.array(filelabels_test[:200]))   # évaluer sur IMDB
```

> 💡 `zero-shot-classification` (modèle NLI) : classer dans des **étiquettes libres** sans entraînement : `pipeline("zero-shot-classification")(txt, candidate_labels=["sport", "politique"])`.

---

## 5. Données : `datasets`

| Outil | Paramètres essentiels | Explication |
|---|---|---|
| `Dataset.from_dict({"text": [...], "label": [...]})` | colonnes | Dataset à partir de listes (ex. fichiers IMDB `pos/` `neg/`). |
| `load_dataset("imdb")` | 🔎 nom du Hub | Téléchargement direct (splits `train`/`test`). |
| `DatasetDict({"train": ds_tr, "test": ds_te})` | | Regroupe les splits. |
| `ds.map(fonction, batched=True)` | `batched=True` (beaucoup plus rapide) ; `remove_columns=["text"]` | Applique la tokenisation à tout le dataset (résultat mis en cache). |
| `ds.train_test_split(test_size=0.1, seed=0)` / `ds.shuffle(seed=0).select(range(n))` | | Validation / sous-échantillon pour tester vite. |

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)      # padding laissé au collator
tokenized = datasets.map(tokenize_function, batched=True)
```

> ⚠️ La colonne des étiquettes doit s'appeler **`label`** (ou `labels`) pour que le `Trainer` la trouve.

---

## 6. Fine-tuning avec `Trainer`

| Paramètre de `TrainingArguments` | Valeur typique | Effet |
|---|---|---|
| `output_dir` | `"./results"` | checkpoints |
| `eval_strategy` (anciennement `evaluation_strategy`) | `"epoch"` / `"steps"` | quand évaluer |
| `learning_rate` | **2e-5** (1e-5 à 5e-5) | ⚠️ petit : on ajuste un modèle déjà bon |
| `per_device_train_batch_size` / `…eval…` | 16 / 32 | mémoire GPU |
| `num_train_epochs` | 1–3 | quelques epochs suffisent |
| `weight_decay` | 0.01 | régularisation (AdamW) |
| `warmup_ratio` | 🔎 0.1 | montée progressive du lr |
| `fp16=True` | 🔎 | demi-précision sur GPU (2× plus rapide) |
| `save_strategy`, `load_best_model_at_end=True`, `metric_for_best_model="accuracy"` | 🔎 | garder le meilleur checkpoint |
| `logging_dir`, `logging_steps` | `"./logs"`, 10 | logs TensorBoard |

```python
import numpy as np
from transformers import DataCollatorWithPadding

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (np.argmax(logits, -1) == labels).mean()}

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
                  tokenizer=tokenizer,                               # (processing_class= dans les versions récentes)
                  data_collator=DataCollatorWithPadding(tokenizer),  # padding dynamique par batch
                  compute_metrics=compute_metrics)
trainer.train()
trainer.evaluate()
```

> ⚠️ IMDB complet (25k textes × 512 tokens) est long sur CPU : tester sur un sous-échantillon (`select(range(2000))`) d'abord.
> 💡 🔎 Fine-tuning économe : geler le corps (`for p in model.base_model.parameters(): p.requires_grad = False`) et n'entraîner que la tête, ou utiliser **LoRA** via la librairie `peft`.

---

## 7. Extraire des embeddings (CLS) + clustering

```python
from sklearn.cluster import KMeans
model = AutoModel.from_pretrained("distilbert-base-uncased").eval()

def get_cls_embeddings(texts, batch_size=32):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i+batch_size], return_tensors="pt",
                               truncation=True, padding=True, max_length=128)
            out = model(**inputs)
            embs.append(out.last_hidden_state[:, 0, :])        # token CLS = position 0 → (B, 768)
    return torch.cat(embs).numpy()

X = get_cls_embeddings(texts)
clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
```

| Choix de représentation | Code | Remarque |
|---|---|---|
| CLS | `last_hidden_state[:, 0]` | Simple ; pour un modèle **non fine-tuné**, le CLS est un résumé médiocre. |
| **Moyenne masquée** (*mean pooling*) | `(h * m).sum(1) / m.sum(1)` avec `m = inputs["attention_mask"].unsqueeze(-1)` | Souvent meilleure ; ⚠️ exclure le padding. |
| Modèles dédiés 🔎 | `sentence-transformers` (`all-MiniLM-L6-v2`) | Embeddings de phrases entraînés pour la similarité cosinus. |

> 👉 Normaliser les vecteurs (`X / ‖X‖`) avant KMeans pour raisonner en **cosinus** ; visualiser avec PCA / t-SNE.

---

# II. LLM locaux avec Ollama

## 8. Interroger un LLM : `chat`, `generate`, `embed`

Prérequis : application Ollama installée et lancée, modèle téléchargé (`ollama pull phi3`), `pip install ollama`.

| Fonction | Paramètres essentiels | Sortie |
|---|---|---|
| `ollama.chat(model, messages=[{"role": "user", "content": ...}], options={...})` | rôles `system` / `user` / `assistant` (historique) | `response.message.content` (ou `response['message']['content']`) |
| `ollama.generate(model, prompt, options={...}, format=...)` | `prompt` brut ; `format="json"` ou un schéma JSON | `response.response` |
| `ollama.embed(model, input)` | texte ou liste | `response.embeddings` (liste de vecteurs) |
| `options` | `{"temperature": 0, "num_predict": 10, "seed": 0}` | `temperature=0` : réponses (quasi) déterministes |

```python
import ollama
response = ollama.chat(model='phi3:latest', messages=[
    {'role': 'system', 'content': 'You are a concise assistant.'},   # consignes générales
    {'role': 'user',   'content': 'Why is the sky blue?'}])
print(response.message.content)
```

> ⚠️ La clé `'preprompt'` du TP n'existe pas dans l'API : les consignes passent par un message de rôle **`system`**.

---

## 9. Classification zero-shot par prompt

```python
def classification_zero_shot(text):
    prompt = ("Give the sentiment of the following movie review. "
              "Answer with exactly one word: 'positive' or 'negative'.\n\n" + text)
    rep = ollama.generate(model='phi3:latest', prompt=prompt,
                          options={"temperature": 0, "num_predict": 5})
    ans = rep.response.strip().lower()
    return 1 if "positive" in ans else 0 if "negative" in ans else -1   # -1 = réponse non conforme

preds = [classification_zero_shot(t) for t in txts_test[:100]]
```

| Technique | Effet |
|---|---|
| Contraindre le format (« exactly one word », liste des réponses permises) | réponses parsables |
| `format="json"` / schéma JSON 🔎 | sortie structurée garantie |
| `temperature=0` | reproductible |
| **Few-shot** : 2–3 exemples annotés dans le prompt | souvent +5 à +15 points |
| Compter les réponses non conformes (−1) | mesurer la fiabilité du format |

> 💡 Alternative : `ollama.embed` sur chaque texte puis une **régression logistique** (sklearn) entraînée sur quelques centaines d'exemples : souvent plus rapide et plus fiable qu'un prompt.

---

## 10. Choisir la bonne approche

| Situation | Approche | Outil |
|---|---|---|
| Tâche standard, modèle déjà fine-tuné existe | utiliser tel quel | `pipeline("…", model=...)` |
| Données annotées (milliers) | **fine-tuning** | `AutoModelForSequenceClassification` + `Trainer` |
| Peu d'annotations | embeddings gelés + classifieur simple | `AutoModel` (CLS / mean pooling) + sklearn |
| Pas d'annotation, regrouper | embeddings + clustering | KMeans / HDBSCAN |
| Pas d'annotation, classer | zero-shot / LLM par prompt | `zero-shot-classification`, Ollama |
| Données confidentielles | modèle **local** | Ollama, modèles HF téléchargés |

---

## 📌 Récapitulatif des pièges classiques

| Piège | Symptôme | Correction |
|---|---|---|
| Tokenizer d'un autre modèle | prédictions absurdes | `AutoTokenizer.from_pretrained(même nom)` |
| Pas de `truncation=True` | erreur au-delà de 512 tokens | `truncation=True` |
| `model(inputs)` au lieu de `model(**inputs)` | `TypeError` | déballer le dict |
| Oublier `model.eval()` + `no_grad` en inférence | dropout actif, mémoire | `eval()` + `torch.no_grad()` |
| Lire `logits` comme des probabilités | scores hors [0,1] | `softmax(-1)` |
| `last_hidden_state[0]` pour le CLS | prend le 1er **exemple** | `[:, 0, :]` (batch first) |
| Moyenne des tokens sans masque | embeddings biaisés par le padding | pondérer par `attention_mask` |
| `num_labels` non précisé pour un modèle générique | tête mal dimensionnée | `num_labels=k` |
| lr de 1e-3 en fine-tuning | modèle détruit (*catastrophic forgetting*) | 1e-5 à 5e-5 |
| `padding="max_length"` partout | entraînement très lent | `DataCollatorWithPadding` |
| Colonne d'étiquettes mal nommée | `Trainer` sans loss | renommer en `label` |
| `evaluation_strategy` vs `eval_strategy` | erreur selon la version | utiliser le nom de votre version |
| Réponse LLM non contrainte | parsing impossible | consignes strictes, `format`, `temperature=0` |
| Clé `preprompt` dans `chat` | ignorée | message `{'role': 'system', ...}` |
| Ollama non lancé | `ConnectionError` | démarrer l'application / `ollama serve` |

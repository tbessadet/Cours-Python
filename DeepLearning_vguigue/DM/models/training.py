import copy
import inspect
import pathlib
import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def probas(model, x):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model.forward(x[i:i + model.batch_size]) for i in range(0, len(x), model.batch_size)])
    return torch.softmax(logits.double(), dim=1).cpu().numpy()   # double : sklearn vérifie que chaque ligne somme à 1

def probas_torch(model, x, batch_size=128):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    return torch.softmax(logits.double(), dim=1).cpu().numpy()

def stats_par_classe(proba, y_true, classes_data):
    '''DataFrame : une ligne par classe + une ligne "moyenne" ; colonnes : loss, accuracy, F1, ROC AUC, average precision'''
    pred = proba.argmax(axis=1)
    y_bin = np.eye(len(classes_data))[y_true]          # (N, n_classes) : one-hot, pour les métriques "une classe contre les autres"
    lignes = []
    for k in range(len(classes_data)):
        masque = y_true == k
        lignes.append({
            "loss":              -np.log(np.clip(proba[masque, k], 1e-12, None)).mean(),   # -log(proba de la vraie classe), images de la classe k
            "accuracy":          (pred[masque] == k).mean(),                               # = rappel de la classe k
            "F1":                f1_score(y_bin[:, k], pred == k),
            "ROC AUC":           roc_auc_score(y_bin[:, k], proba[:, k]),
            "average precision": average_precision_score(y_bin[:, k], proba[:, k]),
        })
    df = pd.DataFrame(lignes, index=classes_data)
    df.loc["moyenne (macro)"] = df.mean()
    return df

def log_confusion_probas(proba, y_true, classes_data):
    '''Dans le run MLflow actif : matrice de confusion (image) et stats par classe (table) en artefacts'''
    pred = proba.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_true, pred), display_labels=classes_data).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("Matrice de confusion (validation)")
    fig.tight_layout()
    mlflow.log_figure(fig, "matrice_confusion_val.png")
    plt.close(fig)       # évite l'affichage de la figure dans la cellule
    df = stats_par_classe(proba, y_true, classes_data).round(4).rename_axis("classe").reset_index()
    mlflow.log_table(df, "stats_par_classe_val.json")

def augmenter_torch(x):
    '''Parmi les 8 symétries du carré (4 rotations de 90° x retournement), tirée au hasard pour chaque image. x : (N, C, H, W) avec H = W'''
    N = x.shape[0]
    k = torch.randint(0, 4, (N,), device=x.device)
    x = torch.stack([torch.rot90(x, j, dims=(2, 3)) for j in range(4)])    # (4, N, C, H, W)
    x = x[k, torch.arange(N, device=x.device)]                              # (N, C, H, W)
    flip = torch.rand(N, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)

def evaluer_torch(model, x, y, batch_size=128):
    model.eval()
    critere = nn.CrossEntropyLoss(reduction='sum')
    total_loss, correct = 0., 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            yhat = model(x[i:i+batch_size])
            total_loss += critere(yhat, y[i:i+batch_size]).item()
            correct += (yhat.argmax(dim=1) == y[i:i+batch_size]).sum().item()
    return total_loss / len(x), correct / len(x)   # loss, accuracy

def predire_torch(model, x, batch_size=128):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i+batch_size]).argmax(dim=1) for i in range(0, len(x), batch_size)])

def fit_torch(model, x, y, x_val, y_val, EPOCH=60, lr=0.05, batch_size=128, patience=8, eps=1e-4,
              momentum=0.0, lr_decay=1.0, augment=False, freeze="aucun"):
    optimiseur = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiseur, gamma=lr_decay)   # lr de l'epoch e = lr * lr_decay**e
    critere = nn.CrossEntropyLoss()
    perf_fit, min_loss, best_state, wait = [], float("inf"), None, 0

    barre_epoch = tqdm(range(EPOCH), desc="Epochs")
    for epoch in barre_epoch:
        model.train()

        if freeze == 'rn18':
            for module in model.modules():               # Faire en sorte que les moyennes mobiles ne bougent pas
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

        somme_loss, correct = 0, 0
        perm = torch.randperm(len(x), device=x.device)                      # Remplace le DataLoader(...)
        for i in range(0, len(x) - len(x) % batch_size, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x[idx], y[idx]

            if augment:
                xb = augmenter_torch(xb)
            yhat = model(xb)
            loss = critere(yhat, yb)
            optimiseur.zero_grad()
            loss.backward()
            optimiseur.step()

            somme_loss += loss.detach() * len(idx)          # remplace batch_loss.append(loss.item())
            correct += (yhat.argmax(dim=1) == yb).sum()     # remplace batch_accuracy.append((yhat.argmax(dim=1) == yb).float().mean().item())
        
        scheduler.step()    # fin d'epoch : mise à jour du lr

        batch_loss = [(somme_loss / len(x)).item()]
        batch_accuracy = [correct.item() / len(x)]
        val_loss, val_accuracy = evaluer_torch(model, x_val, y_val, batch_size)

        perf_fit.append([np.mean(batch_loss), np.mean(batch_accuracy), val_loss, val_accuracy])
        if mlflow.active_run():
            mlflow.log_metrics({"train_loss": round(float(np.mean(batch_loss)), 4), "train_acc": round(float(np.mean(batch_accuracy)), 4),
                                "val_loss": round(float(val_loss), 4), "val_acc": round(float(val_accuracy), 4)}, step=epoch)

        barre_epoch.set_postfix(loss=f"{np.mean(batch_loss):.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_accuracy:.4f}")

        if val_loss < min_loss - eps:
            wait, min_loss = 0, val_loss
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                tqdm.write(f"Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Stop apres {len(perf_fit)} epochs : val_loss={min_loss:.4f}")
    return perf_fit

def mlflow_load(model, name, X_train_ml, y_train_ml, X_val_ml, y_val_ml, classes_data, strategie="", impl="", fit=fit_torch, evaluer=evaluer_torch, probabilites=probas_torch, archi=None, cellule=None, **params):
    print(f"=== {name} ===")
    torch.manual_seed(42)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {n_params:,} (entraînables : {n_train:,})")

    defauts_torch = {k: v.default for k, v in inspect.signature(fit).parameters.items()
                 if v.default is not inspect.Parameter.empty}
    hyper = {**defauts_torch, **params}

    with mlflow.start_run(run_name=f"{impl} {name}"):
        # --- Contexte : paramètres, hyperparamètres, tags
        mlflow.log_params({**hyper, "modele": name, "implementation": impl, "seed": 42,
                           "n_params": n_params, "n_params_entrainables": n_train,
                           "n_train": len(X_train_ml), "n_val": len(X_val_ml)})
        mlflow.set_tags({"strategie": strategie, "device": str(device),
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})

        # --- Code : fichier source du modèle, architecture, fonction d'entraînement
        try:
            fichier_modele = inspect.getsourcefile(type(model))
            mlflow.log_text(pathlib.Path(fichier_modele).read_text(encoding="utf-8"),
                            pathlib.Path(fichier_modele).name)   # ex. CNN_Pytorch.py, Transformers_LTAE_like.py
        except (TypeError, OSError):                             # modèle défini dans le notebook
            mlflow.log_text(cellule if cellule is not None else str(model), "cellule_modeles_torch.py")
            # _ih[-1] n'existe pas ici : _ih est propre au noyau IPython, mlflow_load est importée depuis un module
        mlflow.log_text(archi if archi is not None else str(model), "architecture.py")
        mlflow.log_text(inspect.getsource(fit), "fit_torch.py")

        # --- Entraînement
        t0 = time.time()
        perf_fit = fit(model, X_train_ml, y_train_ml, X_val_ml, y_val_ml, **params)
        temps = time.time() - t0

        # --- Métriques sur la validation (modèle = meilleurs poids restaurés)
        y_np = y_val_ml.cpu().numpy()
        proba = probabilites(model, X_val_ml)
        pred = proba.argmax(axis=1)
        val_loss, val_acc = evaluer(model, X_val_ml, y_val_ml)

        perf = np.array(perf_fit)                  # colonnes : train_loss, train_acc, val_loss, val_acc
        best = int(perf[:, 2].argmin())

        mlflow.log_metrics({
            "best_val_loss": round(val_loss, 4),
            "val_OA": round(val_acc, 4),                                         # overall accuracy
            "val_f1_macro": round(f1_score(y_np, pred, average="macro"), 4),
            "train_loss_at_best": round(perf[best, 0], 4),
            "train_acc_at_best": round(perf[best, 1], 4),
            "gap_acc_train_val": round(perf[best, 1] - perf[best, 3], 4),       # écart de sur-apprentissage
            "best_epoch": best + 1,
            "epochs": len(perf_fit),
            "early_stopped": int(len(perf_fit) < hyper["EPOCH"]),
            "temps_min": round(temps / 60, 2),
            "temps_par_epoch_s": round(temps / len(perf_fit), 2),
            "val_roc_auc": round(roc_auc_score(y_np, proba, multi_class="ovr"), 4),
            "val_avg_precision": round(average_precision_score(np.eye(proba.shape[1])[y_np], proba, average="macro"), 4),
        })

        # --- Artefacts : matrice de confusion et stats par classe
        log_confusion_probas(proba, y_np, classes_data)
    
    return perf_fit, temps


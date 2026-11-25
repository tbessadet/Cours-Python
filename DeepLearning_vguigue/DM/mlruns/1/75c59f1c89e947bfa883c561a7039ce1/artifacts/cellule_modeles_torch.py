configs_torch = {
    "v2":      (creer_modele_torch_v2, dict(lr=0.1, patience=10)),
    "v2 tout": (creer_modele_torch_v2, dict(lr=0.02, patience=10, momentum=0.9, lr_decay=0.95, augment=True)),
}

modeles_t, perfs_t, temps_t = {}, {}, {}
defauts_torch = {k: v.default for k, v in inspect.signature(fit_torch).parameters.items()
                 if v.default is not inspect.Parameter.empty}
for nom, (creer, params) in configs_torch.items():
    print(f"=== {nom} ===")
    torch.manual_seed(42)
    modeles_t[nom] = creer()
    n_params = sum(p.numel() for p in modeles_t[nom].parameters())
    print(f"Paramètres : {n_params:,}")
    with mlflow.start_run(run_name=f"torch {nom}"):
        mlflow.log_params({**defauts_torch, **params, "modele": nom, "implementation": "pytorch", "seed": 42, "n_params": n_params})
        mlflow.log_text(_ih[-1], "cellule_modeles_torch.py")
        mlflow.log_text(inspect.getsource(creer), "architecture.py")
        t0 = time.time()
        perfs_t[nom] = fit_torch(modeles_t[nom], X_train, y_train, X_val, y_val, **params)
        temps_t[nom] = time.time() - t0
        val_loss, val_acc = evaluer_torch(modeles_t[nom], X_val, y_val)
        f1 = f1_score(y_val.cpu().numpy(), predire_torch(modeles_t[nom], X_val).cpu().numpy(), average="macro")
        mlflow.log_metrics({"best_val_loss": val_loss, "best_val_acc": val_acc, "val_f1_macro": f1,
                            "epochs": len(perfs_t[nom]), "temps_min": temps_t[nom] / 60})
        log_confusion_probas(probas_torch(modeles_t[nom], X_val), y_val.cpu().numpy())
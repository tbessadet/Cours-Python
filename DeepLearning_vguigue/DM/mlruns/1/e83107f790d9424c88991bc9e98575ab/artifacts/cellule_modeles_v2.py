def creer_modele_v2(input_shape=(3, 64, 64), EPOCH=60, lr=0.1, batch_size=128, patience=10, eps=1e-4, **kwargs):
    mp, r = Pool(2, 2, 'max'), ReLu()     # sans état : réutilisables
    features = [Conv(3, 32, 3, padding='same'), r, mp,                                          # (B, 32, 32, 32)
                Conv(32, 64, 3, padding='same'), r, Conv(64, 64, 3, padding='same'), r, mp,     # (B, 64, 16, 16)
                Conv(64, 128, 3, padding='same'), r, Conv(128, 128, 3, padding='same'), r, mp,  # (B, 128, 8, 8)
                Conv(128, 256, 3, padding='same'), r, mp,                                       # (B, 256, 4, 4)
                DropOut(p=0.3)]
    classifier = [Flatten(), Dense(n_out=128), r, DropOut(p=0.3), Dense(n_out=10)]              # 4096 -> 128 -> 10
    return Modele(features, classifier, input_shape=input_shape, EPOCH=EPOCH, lr=lr, batch_size=batch_size, patience=patience, eps=eps, **kwargs)

configs = {
    "base":             dict(),
    "augmentation":     dict(augment=True),
    "momentum + decay": dict(lr=0.02, momentum=0.9, lr_decay=0.95),
    "tout":             dict(lr=0.02, momentum=0.9, lr_decay=0.95, augment=True),
}

modeles_v2, temps_v2 = {}, {}
defauts = {k: v.default for k, v in inspect.signature(creer_modele_v2).parameters.items()
           if v.default is not inspect.Parameter.empty}
for nom, params in configs.items():
    print(f"=== {nom} ===")
    torch.manual_seed(42)
    modeles_v2[nom] = creer_modele_v2(**params)
    with mlflow.start_run(run_name=f"maison {nom}"):
        mlflow.log_params({**defauts, **params, "modele": nom, "implementation": "maison", "seed": 42})
        mlflow.log_text(_ih[-1], "cellule_modeles_v2.py")
        t0 = time.time()
        modeles_v2[nom].fit(X_train, y_train, X_val, y_val)
        temps_v2[nom] = time.time() - t0
        for epoch, (tl, ta, vl, va) in enumerate(modeles_v2[nom].perf_fit):
            mlflow.log_metrics({"train_loss": float(tl), "train_acc": float(ta),
                                "val_loss": float(vl), "val_acc": float(va)}, step=epoch)
        mlflow.log_metric("temps_min", temps_v2[nom] / 60)

for nom, m in modeles_v2.items():
    loss, acc = m.evaluate(X_val, y_val)
    f1 = f1_score(y_val.cpu().numpy(), m.predict(X_val).cpu().numpy(), average="macro")
    print(f"{nom:<18} val_loss={loss:.4f}  acc={acc * 100:.2f} %  F1={f1:.4f}  epochs={len(m.perf_fit)}  {temps_v2[nom] / 60:.1f} min")
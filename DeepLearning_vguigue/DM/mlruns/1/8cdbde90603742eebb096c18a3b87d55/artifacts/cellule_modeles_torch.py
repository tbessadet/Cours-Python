configs = {
    "base":             (creer_modele_v2, {}),
    "augmentation":     (creer_modele_v2, {'augment': True}),
    "momentum + decay": (creer_modele_v2, {"lr": 0.02, "momentum": 0.9, "lr_decay": 0.95}),
    "tout":             (creer_modele_v2, {"lr": 0.02, "momentum": 0.9, "lr_decay": 0.95, "augment": True}),
}

modeles_v2, perfs_v2, temps_v2 = {}, {}, {}
defauts = {k: v.default for k, v in inspect.signature(creer_modele_v2).parameters.items()
           if v.default is not inspect.Parameter.empty}

for nom, (creer, params) in configs.items():
    torch.manual_seed(42)
    modeles_v2[nom] = creer(**params)
    perfs_v2[nom], temps_v2[nom] = mlflow_load(modeles_v2[nom], nom, 
                                               X_train, y_train, X_val, y_val,
                                               strategie='maison',
                                               fit=fit_maison, evaluer=evaluer_maison, probabilites=probas,
                                               archi=inspect.getsource(creer_modele_v2), 
                                               **{**defauts, **params})
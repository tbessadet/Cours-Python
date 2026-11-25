epoch = 5
configs = {
    "base":             dict(EPOCH = epoch),
    "augmentation":     dict(augment=True, EPOCH = epoch),
    "momentum + decay": dict(lr=0.02, momentum=0.9, lr_decay=0.95, EPOCH = epoch),
    "tout":             dict(lr=0.02, momentum=0.9, lr_decay=0.95, augment=True, EPOCH = epoch),
}

modeles_v2, perfs_v2, temps_v2 = {}, {}, {}
defauts = {k: v.default for k, v in inspect.signature(creer_modele_v2).parameters.items()
           if v.default is not inspect.Parameter.empty}
for nom, params in configs.items():
    torch.manual_seed(42)
    modeles_v2[nom] = creer_modele_v2(**params)
    perfs_v2[nom], temps_v2[nom] = mlflow_load(modeles_v2[nom], nom, strategie='maison',
                                               fit=fit_maison, evaluer=evaluer_maison, probabilites=probas,
                                               **{**defauts, **params})
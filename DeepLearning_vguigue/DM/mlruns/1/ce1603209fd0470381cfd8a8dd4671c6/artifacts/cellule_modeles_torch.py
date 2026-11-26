configs_torch = {
    "v2":      (creer_modele_torch_v2, {"lr": 0.1, "patience": 10}),
    "v2 tout": (creer_modele_torch_v2, {"lr": 0.02, "patience": 10, "momentum": 0.9, "lr_decay": 0.95, "augment": True}),
}

modeles_t, perfs_t, temps_t = {}, {}, {}
for nom, (creer, params) in configs_torch.items():
    torch.manual_seed(42)
    modeles_t[nom] = creer()
    perfs_t[nom], temps_t[nom] = mlflow_load(modeles_t[nom], nom, strategie='full training', **params)
    print(f"{int(temps_t[nom])} secondes")
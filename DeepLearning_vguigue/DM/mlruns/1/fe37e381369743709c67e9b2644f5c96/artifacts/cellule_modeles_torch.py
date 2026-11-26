params = {"EPOCH": 1, "lr": 0.02, "patience": 10, "momentum": 0.9, "lr_decay": 0.95, "augment": True}
perf_fit, temps = mlflow_load(specLTAE, X_train, y_train, X_val, y_val, "L-TAE", **params)
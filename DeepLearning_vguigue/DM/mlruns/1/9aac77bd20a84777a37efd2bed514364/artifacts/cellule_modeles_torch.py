print(X_train_ms.shape, X_val_ms.shape, y_train_ms.shape, y_val_ms.shape)
params = {"EPOCH": 1, "lr": 0.02, "patience": 10, "momentum": 0.9, "lr_decay": 0.95, "augment": False}
perf_fit, temps = mlflow_load(specLTAE, "nome du modele !!!!!!!!!!!!!!", X_train_ms[:2000], y_train_ms[:2000], X_val_ms[:200], y_val_ms[:200], "L-TAE", **params)
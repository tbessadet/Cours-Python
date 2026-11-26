print(X_train_ms.shape, X_val_ms.shape, y_train_ms.shape, y_val_ms.shape)
params = {"EPOCH": 20, "lr": 0.02, "patience": 10, "momentum": 0.9, "lr_decay": 0.95, "augment": False}
perf_fit, temps = mlflow_load(specLTAE, "nom du modele !!!!!!!!!!!!!!", X_train_ms, y_train_ms, X_val_ms, y_val_ms, classes_ms, "L-TAE", **params)
specLTAE_Band = SpectralLTAE(BandEncoder).to(device)
print(sum(p.numel() for p in specLTAE_Band.parameters()))
specLTAE_Canal = SpectralLTAE(CanalEncoder).to(device)
print(sum(p.numel() for p in specLTAE_Canal.parameters()))

print(X_train_ms.shape, X_val_ms.shape, y_train_ms.shape, y_val_ms.shape)
params = {"EPOCH": 30, "batch_size": 128, "lr": 0.02, "patience": 10, "momentum": 0.9, "lr_decay": 0.95, "augment": False}
perf_fit, temps = mlflow_load(specLTAE_Band, "L-TAE Like Band", X_train_ms, y_train_ms, X_val_ms, y_val_ms, classes_ms, "L-TAE", **params)
perf_fit, temps = mlflow_load(specLTAE_Canal, "L-TAE Like Canal", X_train_ms, y_train_ms, X_val_ms, y_val_ms, classes_ms, "L-TAE", **params)
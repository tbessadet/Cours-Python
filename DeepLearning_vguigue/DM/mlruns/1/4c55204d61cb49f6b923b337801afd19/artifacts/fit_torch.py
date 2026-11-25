def fit_maison(model, x, y, x_val, y_val, **_):      # **_ : ignore les hyperparamètres (déjà dans le modèle)
    model.fit(x, y, x_val, y_val)
    for epoch, (tl, ta, vl, va) in enumerate(model.perf_fit):    # fit maison ne loggue pas par epoch
        mlflow.log_metrics({"train_loss": float(tl), "train_acc": float(ta),
                            "val_loss": float(vl), "val_acc": float(va)}, step=epoch)
    return model.perf_fit

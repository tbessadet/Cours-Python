def fit_torch(model, x, y, x_val, y_val, EPOCH=60, lr=0.05, batch_size=128, patience=8, eps=1e-4,
              momentum=0.0, lr_decay=1.0, augment=False, freeze="aucun"):
    optimiseur = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiseur, gamma=lr_decay)   # lr de l'epoch e = lr * lr_decay**e
    critere = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)
    perf_fit, min_loss, best_state, wait = [], float("inf"), None, 0

    barre_epoch = tqdm(range(EPOCH), desc="Epochs")
    for epoch in barre_epoch:
        model.train()

        if freeze == 'rn18':
            for module in model.modules():               # Faire en sorte que moyenne, sigma ne bougent pas
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()

        batch_loss, batch_accuracy = [], []
        for xb, yb in tqdm(loader, desc="Batchs", leave=False):
            if augment:
                xb = augmenter_torch(xb)
            yhat = model(xb)
            loss = critere(yhat, yb)
            optimiseur.zero_grad()
            loss.backward()
            optimiseur.step()
            batch_loss.append(loss.item())
            batch_accuracy.append((yhat.argmax(dim=1) == yb).float().mean().item())
        scheduler.step()    # fin d'epoch : mise à jour du lr

        val_loss, val_accuracy = evaluer_torch(model, x_val, y_val, batch_size)

        perf_fit.append([np.mean(batch_loss), np.mean(batch_accuracy), val_loss, val_accuracy])
        if mlflow.active_run():
            mlflow.log_metrics({"train_loss": round(float(np.mean(batch_loss)), 4), "train_acc": round(float(np.mean(batch_accuracy)), 4),
                                "val_loss": round(float(val_loss), 4), "val_acc": round(float(val_accuracy), 4)}, step=epoch)

        barre_epoch.set_postfix(loss=f"{np.mean(batch_loss):.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_accuracy:.4f}")

        if val_loss < min_loss - eps:
            wait, min_loss = 0, val_loss
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                tqdm.write(f"Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"Stop apres {len(perf_fit)} epochs : val_loss={min_loss:.4f}")
    return perf_fit

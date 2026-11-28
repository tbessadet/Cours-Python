def CNN_scratch(input_shape=(3, 64, 64), EPOCH=60, lr=0.1, batch_size=128, patience=10, eps=1e-4, **kwargs):
    mp, r = Pool(2, 2, 'max'), ReLu()     # sans état : réutilisables                           # (B, C, H, W) = (B, 3, 64, 64)
    features = [Conv(input_shape[0], 32, 3, padding='same'), r, mp,                             # (B, 32, 32, 32)
                Conv(32, 64, 3, padding='same'), r, Conv(64, 64, 3, padding='same'), r, mp,     # (B, 64, 16, 16)
                Conv(64, 128, 3, padding='same'), r, Conv(128, 128, 3, padding='same'), r, mp,  # (B, 128, 8, 8)
                Conv(128, 256, 3, padding='same'), r, mp,                                       # (B, 256, 4, 4)
                DropOut(p=0.3)]
    classifier = [Flatten(), Dense(n_out=128), r, DropOut(p=0.3), Dense(n_out=10)]              # 4096 -> 128 -> 10
    return Modele(features, classifier, input_shape=input_shape, EPOCH=EPOCH, lr=lr, batch_size=batch_size, patience=patience, eps=eps, **kwargs)

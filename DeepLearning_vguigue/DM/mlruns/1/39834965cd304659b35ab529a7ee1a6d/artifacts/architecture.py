def creer_modele_torch_v2():
    r, mp = nn.ReLU(), nn.MaxPool2d(2, 2)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding='same'), r, mp,                                                # (B, 32, 32, 32)
        nn.Conv2d(32, 64, 3, padding='same'), r, nn.Conv2d(64, 64, 3, padding='same'), r, mp,      # (B, 64, 16, 16)
        nn.Conv2d(64, 128, 3, padding='same'), r, nn.Conv2d(128, 128, 3, padding='same'), r, mp,   # (B, 128, 8, 8)
        nn.Conv2d(128, 256, 3, padding='same'), r, mp,                                             # (B, 256, 4, 4)
        nn.Dropout(0.3),
        nn.Flatten(), nn.Linear(256 * 4 * 4, 128), r, nn.Dropout(0.3), nn.Linear(128, 10),         # 4096 -> 128 -> 10
    ).to(device)

## Transformers
import math

import torch
from torch import nn

WAVELENGTHS_NM = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]

class BandEncoder(nn.Module):
    """(B,13,H,W) -> (B,13,n) ; CNN partagé entre les bandes."""
    def __init__(self, n=128):
        super().__init__()
        def block(dim_in, dim_out):
            return nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(dim_out), nn.ReLU(inplace=True))
        self.net = nn.Sequential(                                # (B*C, 1, 64, 64)
            block(1, 32), nn.MaxPool2d(2),                       # (B*C, 32, 32, 32)
            block(32, 64), nn.MaxPool2d(2),                      # (B*C, 64, 16, 16)
            block(64, n), nn.AdaptiveAvgPool2d(1), nn.Flatten()) # (B*C, n, 1, 1) -> (B*C, n)

    def forward(self, x):
        B, C, H, W = x.shape                    # (B, C, H, W)
        z = self.net(x.reshape(B * C, 1, H, W)) # (B*C, 1, H, W)
        return z.view(B, C, -1)                 # (B, C, n)

class CanalEncoder(nn.Module):
    """initiallement (B,13,H,W) -> entrée (B,1,H,W) -> (B, n) ; 13 CNN encodant une longueur spectrale"""
    def __init__(self, n=128, C=13):
        super().__init__()
        self.C = C
        def block(dim_in, dim_out):
            return nn.Sequential(
                nn.Conv2d(C * dim_in, C * dim_out, 3, padding=1, groups=C, bias=False),
                nn.BatchNorm2d(C * dim_out), nn.ReLU(inplace=True))
        self.net = nn.Sequential(                                    # (B, 1, 64, 64)
                block(1, 32), nn.MaxPool2d(2),                       # (B, 32, 32, 32)
                block(32, 64), nn.MaxPool2d(2),                      # (B, 64, 16, 16)
                block(64, n), nn.AdaptiveAvgPool2d(1), nn.Flatten()) # (B, n, 1, 1) => (B, C*n)
        
    def forward(self, x):   # (B, C, H, W)
        return self.net(x).view(x.size(0), self.C, -1)    # (B, C*n) => (B, C, n)

class LTAE(nn.Module):
    """(B,S,n) + positions (B,S) -> (B,n)"""
    def __init__(self, n=128, n_head=16, d_k=8, dropout=0.2, T=1000):
        super().__init__()
        assert n % n_head == 0
        self.n, self.h, self.d_k, self.T = n, n_head, d_k, T
        self.norm_in = nn.LayerNorm(n)
        self.Q = nn.Parameter(torch.randn(n_head, d_k) * math.sqrt(2.0 / d_k))
        self.fc_k = nn.Linear(n, n_head * d_k)
        self.drop = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(n)

    def pos_enc(self, pos):
        i = torch.arange(self.n, device=pos.device)
        div = torch.pow(self.T, 2 * (i // 2) / self.n)
        pe = pos[..., None] / div
        pe = torch.stack([torch.sin(pe[..., 0::2]), torch.cos(pe[..., 1::2])], -1)
        return pe.flatten(-2)                           # (B,S,n)

    def forward(self, x, pos, return_attn=False):
        B, S, _ = x.shape
        x = self.norm_in(x) + self.pos_enc(pos)
        k = self.fc_k(x).view(B, S, self.h, self.d_k)
        attn = torch.einsum('hd,bshd->bhs', self.Q, k) / math.sqrt(self.d_k)
        attn = self.drop(attn.softmax(-1))              # softmax sur les bandes
        v = x.view(B, S, self.h, self.n // self.h)      # valeurs = chunks de x
        out = torch.einsum('bhs,bshc->bhc', attn, v).reshape(B, self.n)
        out = self.norm_out(out)
        return (out, attn) if return_attn else out

class SpectralLTAE(nn.Module):
    def __init__(self, encoder="band", n=128, n_classes=10, dropout=0.3):
        super().__init__()
        E = {"band": BandEncoder, "canal": CanalEncoder}
        encoder = E[encoder]
        self.enc = encoder(n)
        self.ltae = LTAE(n)
        self.head = nn.Sequential(
            nn.Linear(n, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, n_classes))
        self.register_buffer('pos', torch.tensor(WAVELENGTHS_NM, dtype=torch.float32) / 10)

    def forward(self, x, return_attn=False):
        f = self.enc(x)                                  # (B,13,n)
        pos = self.pos.expand(x.size(0), -1)
        if return_attn:
            z, attn = self.ltae(f, pos, return_attn=True)
            return self.head(z), attn
        return self.head(self.ltae(f, pos))
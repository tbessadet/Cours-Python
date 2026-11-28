"""
CNN "fait maison" (repris de CNN.ipynb) : couches, modèle et outils de visualisation.

Utilisation depuis un notebook du même dossier :
    from CNN import Conv, Pool, ReLu, Flatten, DropOut, Dense, Modele, device
"""
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Conv:
    def __init__(self, C_in=1, C_out=6, kernel=3, stride=1, padding="valid"):
        '''
        C_in : nbr de canaux d'entrée - 3 pour RGB ...
        C_out : nombre de filtre et donc nombre de canaux en sortie de convolution
        kernel : taille k du noyau (k x k)
        stride : pas de déplacement de la fenêtre d'application du noyau
        padding : 'valid' (P=0), 'same' (P=(k-1)//2) ou entier >= 0
        '''
        if kernel <= 0 or stride <= 0 or C_in <= 0 or C_out <= 0:
            raise ValueError(f"kernel {kernel}, stride {stride}, C_in {C_in} et C_out {C_out} doivent être positifs")
        if padding not in ['valid', 'same'] and (not isinstance(padding, int) or padding < 0):
            raise ValueError(f"padding {padding} doit être 'valid', 'same' ou entier >= 0")

        self.C_in, self.C_out = C_in, C_out
        self.k, self.stride = kernel, stride
        self.padding = padding

        # Traduction du padding str -> int
        pad_dict = {'valid': 0, 'same': (self.k - 1)//2}
        if padding in pad_dict:
            self.P = pad_dict[padding]
        elif isinstance(padding, int):
            self.P = padding
        else:
            raise KeyError("Definir padding pour 'valid', 'same' ou entier")

        # Initialisation des poids (C_out, C_in, k, k)
        self.noyau = (torch.randn(C_out, C_in, kernel, kernel, device=device) / (C_in * kernel**2) ** 0.5).requires_grad_()
        self.biais = torch.zeros(C_out, device=device, requires_grad=True)

    def parameters(self):
        return [self.noyau, self.biais]

    def forward(self, x):                   # x : (N, C_in, H, W)
        if x.shape[1] != self.C_in:
            raise ValueError(f"Conv attend {self.C_in} canaux en entrée, recu {x.shape[1]}")
        if self.P > 0:
            x = torch.nn.functional.pad(x, (self.P, self.P, self.P, self.P))
        H_out = (x.shape[-2] - self.k) // self.stride + 1
        W_out = (x.shape[-1] - self.k) // self.stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Image trop petite ({x.shape[-2]}x{x.shape[-1]}) pour un noyau {self.k}x{self.k}")

        fenetre = x.unfold(2, self.k, self.stride).unfold(3, self.k, self.stride)   # (N, C_in, H_out, W_out, k, k)
        # avec self.noyau de shape (C_out, C_in, k, k) et self.biais de shape (C_out,)
        return torch.einsum("nchwab,ocab->nohw", fenetre, self.noyau) + self.biais[None, :, None, None]   # (N, C_out, H_out, W_out)

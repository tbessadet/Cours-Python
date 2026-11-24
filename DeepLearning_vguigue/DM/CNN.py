"""
CNN "fait maison" (repris de CNN.ipynb) : couches, modèle et outils de visualisation.

Utilisation depuis un notebook du même dossier :
    from CNN import Conv, Pool, ReLu, Flatten, DropOut, Dense, Modele, device
"""
import matplotlib.pyplot as plt
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


class Pool:
    def __init__(self, kernel=2, stride=2, type='max'):
        '''
        kernel : taille k du noyau (k x k)
        stride : pas de déplacement de la fenêtre d'application du noyau
        type : type de pooling ('max' ou 'mean')
        '''
        if kernel <= 0 or stride <= 0:
            raise ValueError("kernel et stride doivent être positifs")
        if type not in ('max', 'mean'):
            raise ValueError("pool doit être du type 'max' ou 'mean'")

        self.k = kernel
        self.stride = stride
        self.type = type

    def parameters(self):
        return []

    def forward(self, x):   # x : (N, C, H, W), pas de padding possible
        H_out = (x.shape[-2] - self.k) // self.stride + 1
        W_out = (x.shape[-1] - self.k) // self.stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(f"Image trop petite ({x.shape[-2]}x{x.shape[-1]}) pour un noyau {self.k}x{self.k}")
        fenetre = x.unfold(2, self.k, self.stride).unfold(3, self.k, self.stride)   # (N, C, H_out, W_out, k, k)
        if self.type == 'max':
            return torch.amax(fenetre, dim=(-2, -1))    # (N, C, H_out, W_out)
        elif self.type == 'mean':
            return torch.mean(fenetre, dim=(-2, -1))    # (N, C, H_out, W_out)
        else:
            raise KeyError(f"problème de définition de self.type : {self.type}")


class ReLu:
    def parameters(self):
        return []

    def forward(self, x):
        return torch.relu(x)    # max(0,x)


class Flatten:
    def parameters(self):
        return []

    def forward(self, x):
        return x.flatten(1)     # (N, C, H, W) -> (N, C*H*W)


class DropOut:
    '''
    A placer avant un flatten, pour "tuer des neuronnes" aléatoirement
    '''
    def __init__(self, p=0.5):
        if 0 <= p < 1:
            self.p = p
        else:
            raise ValueError(f"p doit etre entre [0,1[, {p} invalide.")
        self.training = True

    def parameters(self):
        return []

    def forward(self, x):
        if not self.training:
            return x
        masque = (torch.rand_like(x) >= self.p).float()
        return x * masque / (1 - self.p)


class Dense:
    def __init__(self, n_out):
        self.n_in = None    # Connu seulement après le 1er passage
        self.n_out = n_out
        self.W = None
        self.b = None

    def parameters(self):
        if self.W is None:
            raise RuntimeError("Dense encore non initialisé : faire tourner le modèle (premier passage)")
        return [self.W, self.b]

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"Dense attend une entrée (N, n_in), recu {x.shape} : oubli de Flatten ?")
        if self.W is None:
            self.n_in = x.shape[1]
            self.W = (torch.randn(self.n_in, self.n_out, device=x.device) / self.n_in ** 0.5).requires_grad_()
            self.b = torch.zeros(self.n_out, device=x.device, requires_grad=True)
        elif x.shape[1] != self.n_in:
            raise ValueError(f"Dense attend {self.n_in} entrées, recu {x.shape[1]}")
        return x @ self.W + self.b      # logits (softmax dans la cross-entropy)


class Modele:
    def __init__(self, features, classifier, input_shape=(1, 28, 28), EPOCH=100, lr=0.1, batch_size=128, patience=5, eps=1e-4,
                 momentum=0.0, lr_decay=1.0, augment=False):
        '''
        features : liste de couches d'extraction (Conv, ReLu, Pool, DropOut ...)
        classifier : liste de couches de classification (Flatten, Dense ...)
        input_shape : (C, H, W) d'une image, pour le passage à blanc qui fixe la taille des Dense
        momentum : 0 = SGD simple
        lr_decay : lr de l'epoch e = lr * lr_decay**e (1 = constant)
        augment : retournements/rotations aléatoires, à l'entraînement uniquement (images carrées)
        '''
        self.features = list(features)
        self.classifier = list(classifier)
        self.couches = self.features + self.classifier
        self.input_shape = input_shape
        self.EPOCH = EPOCH
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.eps = eps
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.augment = augment

        # Passage à blanc : initialise les Dense (taille d'entrée connue)
        with torch.no_grad():
            self.forward(torch.zeros(1, *input_shape, device=device))

    def parameters(self):
        p = []
        for c in self.couches:
            p.extend(param for param in c.parameters() if param.requires_grad)
        return p

    def train(self):
        for c in self.couches:
            if hasattr(c, "training"):
                c.training = True

    def eval(self):
        for c in self.couches:
            if hasattr(c, "training"):
                c.training = False

    def forward(self, x):
        for c in self.couches:
            x = c.forward(x)
        return x

    def evaluate(self, x, y):
        self.eval()     # Desactiver toutes les couches de training (DropOut ...)
        total_loss, correct = 0., 0
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                xb, yb = x[i:i+self.batch_size], y[i:i+self.batch_size]
                yhat = self.forward(xb)
                total_loss += -torch.log_softmax(yhat, dim=1)[torch.arange(len(yb), device=yhat.device), yb].sum().item()
                correct += (torch.argmax(yhat, dim=1) == yb).sum().item()
        return total_loss / len(x), correct / len(x)    # loss, accuracy

    def augmenter(self, x):
        '''Parmi les 8 symétries du carré (4 rotations de 90° x retournement), tirée au hasard pour chaque image du batch. x : (N, C, H, W) avec H = W'''
        N = x.shape[0]
        k = torch.randint(0, 4, (N,), device=x.device)
        x = torch.stack([torch.rot90(x, j, dims=(2, 3)) for j in range(4)])    # (4, N, C, H, W)
        x = x[k, torch.arange(N, device=x.device)]                              # (N, C, H, W)
        flip = torch.rand(N, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x)

    def fit(self, x, y, x_val, y_val):
        self.perf_fit = []
        self.min_loss = float("inf")
        self.best_params = None
        wait = 0

        loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True, drop_last=True)

        vitesse = [torch.zeros_like(param) for param in self.parameters()]     # momentum : une vitesse par paramètre

        barre_epoch = tqdm(range(self.EPOCH), desc="Epochs")
        for epoch in barre_epoch:
            self.train()    # activer toutes les couches permettant d'encadrer l'entrainement (DropOut ...)
            lr_epoch = self.lr * self.lr_decay ** epoch

            batch_loss = []
            batch_accuracy = []

            for xb, yb in tqdm(loader, desc="Batchs", leave=False):
                if self.augment:
                    xb = self.augmenter(xb)
                yhat = self.forward(xb)
                p = self.parameters()

                loss = -torch.log_softmax(yhat, dim=1)[torch.arange(yhat.shape[0], device=yhat.device), yb].mean()

                for param in p:
                    param.grad = None

                loss.backward()
                with torch.no_grad():
                    for param, v in zip(p, vitesse):
                        v.mul_(self.momentum).add_(param.grad)
                        param -= lr_epoch * v       # back propagation du gradient (in place fonctionne dans la boucle)

                batch_loss.append(loss.item())
                batch_accuracy.append((torch.argmax(yhat, dim=1) == yb).float().mean().item())

            val_loss, val_accuracy = self.evaluate(x_val, y_val)
            self.perf_fit.append([np.mean(batch_loss), np.mean(batch_accuracy), val_loss, val_accuracy])
            barre_epoch.set_postfix(loss=f"{np.mean(batch_loss):.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_accuracy:.4f}")

            if val_loss < self.min_loss - self.eps:
                wait = 0
                self.min_loss = val_loss
                self.best_params = [param.detach().clone() for param in self.parameters()]
            else:
                wait += 1
                if wait >= self.patience:
                    tqdm.write(f"Early stop at epoch {epoch + 1}")
                    break

        if self.best_params is None:
            raise RuntimeError("Aucune amélioration de la val_loss (divergence ? lr trop grand ?)")

        with torch.no_grad():
            for param, best in zip(self.parameters(), self.best_params):
                param.copy_(best)

        self.eval()     # Desactiver toutes les couches de training (DropOut ...) : modele pret pour predict pour la suite

        print(f"Stop apres {len(self.perf_fit)} epochs : val_loss={self.min_loss:.4f}")

    def predict(self, x):
        self.eval()     # Desactiver toutes les couches de training (DropOut ...)
        with torch.no_grad():
            preds = [torch.argmax(self.forward(x[i:i+self.batch_size]), dim=1) for i in range(0, len(x), self.batch_size)]
        return torch.cat(preds)


# ---------------------------------------------------------------------------
# Visualisation : filtres / cartes, saliency, CAM, Grad-CAM
# ---------------------------------------------------------------------------

def _image_affichable(img):
    '''(C, H, W) -> (H, W) en gris ou (H, W, 3) en RGB, sur CPU, ramenée dans [0, 1]'''
    img = img.detach().cpu()
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.squeeze(0)


def analyse_conv(model, img, nb_filtre_max=16):
    model.eval()            # passer en mode evaluation (normalement on y est deja)
    x = img.unsqueeze(0)
    lignes = []             # (nom, liste d'images à afficher : filtre/sortie)

    with torch.no_grad():
        for couche in model.features:   # uniquement les couches qui nous interesse
            x = couche.forward(x)
            sortie = x.squeeze(0)
            nb = min(nb_filtre_max, sortie.shape[0])    # on represente maximum `nb_filtre_max` sinon ca peut etre trop petit a l'ecran

            if isinstance(couche, Conv):
                noyau = couche.noyau.detach()
                lignes.append(("Filtre", [noyau[j, 0] for j in range(nb)]))
                lignes.append(("Conv", [sortie[j] for j in range(nb)]))

            elif isinstance(couche, Pool):
                lignes.append((couche.type + "Pool", [sortie[j] for j in range(nb)]))

    if not lignes:
        raise ValueError("Aucune couche Conv/Pool exploitable dans le modèle")

    nb_col = max(len(images) for _, images in lignes)
    nb_lignes = 1 + len(lignes)     # +1 pour l'image en entete

    fig, axs = plt.subplots(nb_lignes, nb_col, figsize=(2*nb_col, 2*nb_lignes), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")

    axs[0, 0].imshow(_image_affichable(img), cmap="gray")
    axs[0, 0].set_title("Image d'origine")

    for i, (nom, images) in enumerate(lignes, start=1):
        for j, im in enumerate(images):
            axs[i, j].imshow(im.cpu(), cmap="gray")
            axs[i, j].set_title(f"{nom} {j + 1}")

    fig.tight_layout()
    plt.show()


def carte_saliency(model, img, label):
    model.eval()
    x = img.clone().unsqueeze(0).requires_grad_()       # (1, C, H, W)
    score = model.forward(x)[0, label]
    score.backward()
    return x.grad.abs().amax(dim=1).squeeze(0).cpu()    # max sur les canaux -> (H, W)


def carte_cam(model, img, label):   # Verifier la taille du kernel du Pool, doit être la taille de la sortie de la derniere conv
    model.eval()                    # ici ca sert à désactiver le dropout par ex
    with torch.no_grad():
        x = img.clone().unsqueeze(0)
        for c in model.features:
            x = c.forward(x)
        w = model.classifier[-1].W[:, label]
        return torch.einsum('nfhw,f->hw', x, w).cpu()


def carte_gradcam(model, img, label):
    model.eval()
    x = img.clone().unsqueeze(0).requires_grad_()       # (1, C, H, W)
    for c in model.features:
        x = c.forward(x)
    a = x                                               # (1, K, H, W) : sortie de la dernière couche de features
    sortie = a
    for c in model.classifier:
        sortie = c.forward(sortie)
    grad = torch.autograd.grad(sortie[0, label], a)[0]  # (1, K, H, W) : d(score) / d(cartes)
    w = grad.mean(dim=(2, 3))                           # (1, K) : poids de chaque filtre = moyenne spatiale du gradient
    return torch.relu(torch.einsum('nf,nfhw->hw', w, a)).detach().cpu()


def carte_guided_gradcam(model, img, label):
    g = carte_gradcam(model, img, label)
    g = torch.nn.functional.interpolate(g[None, None], size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    g = g / (g.max() + 1e-8)                            # normalisé entre 0 et 1
    return g * carte_saliency(model, img, label)

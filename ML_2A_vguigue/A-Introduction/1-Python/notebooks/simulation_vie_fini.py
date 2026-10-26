#%%
import random as rd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


#%%
class plante:
    def __init__(self, pos):
        self.pos = pos
        self.vivant = True

    def mort(self):
        self.vivant = False

#%%
class herbivore:
    def __init__(self, pos):
        self.pos = pos
        self.nourriture = 5
        self.age = 1
        self.vivant = True

    def deplacement(self, xp, yp):
        x = self.pos[0]
        y = self.pos[1]

        if xp and yp:   # si on a des plantes
            dist = [((xp[i]-x)**2 + (yp[i]-y)**2, i) for i in range(len(xp))]
            nearest = min(dist) 

            if np.sqrt(nearest[0]) <= 10:
                dx = np.sign(xp[nearest[1]] - x)
                dy = np.sign(yp[nearest[1]] - y)
                x += dx
                y += dy

            else:
                x += rd.randint(-1, 1)
                y += rd.randint(-1,1)

        else:
            x += rd.randint(-1, 1)
            y += rd.randint(-1,1)
            
        if x < 0:
            x = Taille_carte - 1
        elif x >= Taille_carte:
            x = 0
        if y < 0:
            y = Taille_carte - 1
        elif y >= Taille_carte:
            y = 0
        
        self.pos = (x,y)
        self.nourriture -= 0.25

    def manger(self, plante):
        if plante.vivant:
            plante.mort()
            self.nourriture += 5

    def repro(self):
        if self.nourriture >= 10:
            self.nourriture -= 5
            return herbivore(self.pos)
        return None

    def mort(self):
        self.vivant = False

    def action(self, plante_list, xp, yp):
        if self.nourriture == 0 or self.age > 10:
            self.mort()
            return None
                
        self.deplacement(xp, yp) # mouvement visant une plante range 3

        for plante in plante_list:
            if plante.vivant and plante.pos == self.pos:
                self.manger(plante)
                break

        self.age += 1    
        return self.repro()

class carnivore:
    def __init__(self, pos):
        self.pos = pos
        self.nourriture = 5
        self.age = 1
        self.vivant = True

    def deplacement(self, xh, yh):
            x = self.pos[0]
            y = self.pos[1]
    
            if xh and yh:   # si on a des plantes
                dist = [((xh[i]-x)**2 + (yh[i]-y)**2, i) for i in range(len(xh))]
                nearest = min(dist) 
    
                if np.sqrt(nearest[0]) <= 10:
                    dx = np.sign(xh[nearest[1]] - x)
                    dy = np.sign(yh[nearest[1]] - y)
                    x += dx
                    y += dy

                else:
                    x += rd.randint(-1, 1)
                    y += rd.randint(-1,1)
    
            else:
                x += rd.randint(-1, 1)
                y += rd.randint(-1,1)
                
            if x < 0:
                x = Taille_carte - 1
            elif x >= Taille_carte:
                x = 0
            if y < 0:
                y = Taille_carte - 1
            elif y >= Taille_carte:
                y = 0

            self.pos = (x,y)
            self.nourriture -= 0.5

    def manger(self, herbivore):
        if herbivore.vivant:
            herbivore.mort()
            self.nourriture += 10

    def repro(self):
        if self.nourriture >= 20:
            self.nourriture -= 10
            return carnivore(self.pos)
        return None
    
    def mort(self):
        self.vivant = False

    def action(self, herbivore_list, xh, yh):
        if self.nourriture == 0 or self.age > 50:
            self.mort()
            return None
        
        self.deplacement(xh,yh) # mouvement aléatoire

        for herbivore in herbivore_list:
            if herbivore.vivant and herbivore.pos == self.pos:
                self.manger(herbivore)
                break

        self.age += 1    
        return self.repro()

#%% Activiter sur la grille
def initialisation(nplante, nherbivore, ncarnivore):
    plante_list =       [plante((rd.randint(0,Taille_carte - 1), rd.randint(0,Taille_carte - 1))) for i in range(nplante)]
    herbivore_list =    [herbivore((rd.randint(0,Taille_carte - 1), rd.randint(0,Taille_carte - 1))) for i in range(nherbivore)]
    carnivore_list =    [carnivore((rd.randint(0,Taille_carte - 1), rd.randint(0,Taille_carte - 1))) for i in range(ncarnivore)]
    return plante_list, herbivore_list, carnivore_list


def Simulation(plante_list, herbivore_list, carnivore_list):
    add_herbivore = []
    for herbivore in herbivore_list: #action des herbivores
        if herbivore.vivant:
            xp, yp = get_positions(plante_list)
            enfant = herbivore.action(plante_list, xp, yp)
            if enfant is not None:
                add_herbivore.append(enfant)

    add_carnivore = []
    for carnivore in carnivore_list: #action des carnivores
        if carnivore.vivant:
            xh, yh = get_positions(herbivore_list)
            enfant = carnivore.action(herbivore_list, xh, yh)
            if enfant is not None:
                add_carnivore.append(enfant)

    plante_list[:] = [p for p in plante_list if p.vivant]
    herbivore_list[:] = [h for h in herbivore_list if h.vivant] + add_herbivore 
    carnivore_list[:] = [c for c in carnivore_list if c.vivant] + add_carnivore


#%% Affichage graphique
Taille_carte = 100
NB_FRAMES = 200          # nombre de jours simulés
INTERVALLE_MS = 150      # vitesse d'animation (ms entre 2 frames)


def get_positions(liste_individus):
    """Extrait les positions (x, y) des individus encore vivants."""
    vivants = [ind for ind in liste_individus if ind.vivant]
    if not vivants:
        return [], []
    xs = [ind.pos[0] for ind in vivants]
    ys = [ind.pos[1] for ind in vivants]
    return xs, ys


def creer_animation(plante_list, herbivore_list, carnivore_list):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, Taille_carte)
    ax.set_ylim(0, Taille_carte)
    ax.set_title("Jour 0")
    ax.set_aspect("equal")

    # un scatter plot par classe, pour avoir une couleur/légende distincte
    scat_plantes = ax.scatter([], [], c="forestgreen", s=15, label="Plantes", marker="s")
    scat_herbivores = ax.scatter([], [], c="royalblue", s=30, label="Herbivores")
    scat_carnivores = ax.scatter([], [], c="firebrick", s=40, label="Carnivores")

    ax.legend(loc="upper right")

    jour_compteur = {"jour": 0}
    
    def update(frame):
        # fait avancer la simulation d'un jour
        Simulation(plante_list, herbivore_list, carnivore_list)
        jour_compteur["jour"] += 1

        xp, yp = get_positions(plante_list)
        xh, yh = get_positions(herbivore_list)
        xc, yc = get_positions(carnivore_list)

        # set_offsets attend un tableau de forme (N, 2), même vide
        scat_plantes.set_offsets(np.column_stack((xp, yp)) if xp else np.empty((0, 2)))
        scat_herbivores.set_offsets(np.column_stack((xh, yh)) if xh else np.empty((0, 2)))
        scat_carnivores.set_offsets(np.column_stack((xc, yc)) if xc else np.empty((0, 2)))

        ax.set_title(
            f"Jour {jour_compteur['jour']} — "
            f"{len(xp)} plantes, {len(xh)} herbivores, {len(xc)} carnivores"
        )

        return scat_plantes, scat_herbivores, scat_carnivores

    anim = animation.FuncAnimation(
        fig, update, frames=NB_FRAMES, interval=INTERVALLE_MS, blit=False, repeat=False
    )
    plt.show()
    return anim

#%%
plante_list, herbivore_list, carnivore_list = initialisation(500, 40, 20)
anim = creer_animation(plante_list, herbivore_list, carnivore_list)

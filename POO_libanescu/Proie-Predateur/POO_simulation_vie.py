import random as rd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


#%%
class plante:
    NOURRITURE = 12

    def __init__(self, pos):
        self.pos = pos
        self.nourriture = plante.NOURRITURE
        self.vivant = True

    def mort(self):
        self.vivant = False

#%% Definition des objets
class animale:
    # ----- Parametres redefinis dans chaque espece (voir plus bas) -----
    COUT_DEP     = 0.015   # cout d'un pas = dep_age * rmax * COUT_DEP
    AGE_MATURE   = 6       # avant : deplacement moins efficace | repro impossible
    AGE_VIEU     = 12      # apres : la senescence commence | repro impossible
    ESPERANCE    = 28      # age ou le cout du deplacement est multiplie par (1 + K)
    SENES_K      = 9.0     # amplitude de la senescence
    SENES_P      = 3.0     # brutalite de la senescence (plus grand = plus tardif mais plus violent)
    COUT_JEUNE   = 1.3     # surcout a la naissance
    REPRO_AGE    = 6       # age minimum pour se reproduire
    REPRO_SEUIL  = 0.8     # fraction de rmax necessaire
    REPRO_GARDE  = 0.4     # fraction des reserves que le parent CONSERVE
    REPRO_PART   = 0.75    # part de l'energie cedee qui arrive reellement a l'enfant
    REPRO_PAUSE  = 2       # jours minimum entre deux portees
    SUCCES_CHASSE = 1.0    # probabilite qu'une attaque aboutisse
    PART_CORPS   = 0.0     # valeur du corps de la proie, en fraction de son rmax

    def __init__(self, pos, espece, vitesse, vision, rmax, nourriture):
        self.pos = pos
        self.espece = espece
        self.vitesse = vitesse
        self.vision = vision
        self.rmax = rmax
        self.nourriture = nourriture
        self.trepro = 10
        self.age = 1
        self.vivant = True

    def facteur_age(self):
        return dep_age(self.age, self.AGE_MATURE, self.AGE_VIEU, self.ESPERANCE, self.SENES_K, self.SENES_P, self.COUT_JEUNE)

    def deplacement(self, cibles = None):
        for _ in range(self.vitesse):
            x = self.pos[0]
            y = self.pos[1]
            if cibles:                                          # Si on au moins une cible
                nearest = min( [(abs(c.pos[0]-x) + abs(c.pos[1]-y), c) for c in cibles], key=lambda a: a[0] ) # recherche de la cible la plus proche | Distance de Manhattan plutot qu'euclidienne
                
                if nearest[0] <= self.vision:                   # Si la plus proche est <= portée detection
                    dx = nearest[1].pos[0] - x                  # On va se diriger vers elle (en réduisant la plus grande distance x ou y)
                    dy = nearest[1].pos[1] - y
                    if abs(dx) >= abs(dy):
                        x += np.sign(dx)
                    else:
                        y += np.sign(dy)

                    if nearest[0] <= 1:              # On pourait self.pos = (x,y), puis nearest[1].pos == self.pos, mais ca oblige a le remettre plusierus fois en dessous
                        self.manger(nearest[1]) 
                        cibles.remove(nearest[1])               # Actualisation des cibles vivantes
                else:                                           # Sinon mouvement aléatoire
                    if rd.random() > 0.5:
                        x += rd.choice([-1,1])
                    else:
                        y += rd.choice([-1,1])

            else:
                if rd.random() > 0.5:
                    x += rd.choice([-1,1])
                else:
                    y += rd.choice([-1,1])
            
            if x < 0:                                           # Tests pour boucler sur les bords de la map si on en sort
                x = Taille_carte - 1
            elif x >= Taille_carte:
                x = 0
            if y < 0:
                y = Taille_carte - 1
            elif y >= Taille_carte:
                y = 0

            cout = self.facteur_age() * self.rmax * self.COUT_DEP
            self.nourriture = max(0, self.nourriture - cout)
            self.pos = (x,y)

    def manger(self, cible):
        if rd.random() > self.SUCCES_CHASSE:
            return
        if cible.vivant and self.nourriture < self.rmax*0.95:
            corps = self.PART_CORPS * getattr(cible, 'rmax', 0)
            gain = proportion_reserve_proie(corps + cible.nourriture) / self.facteur_age() + 8
            self.nourriture = min(self.rmax, self.nourriture + gain)
            cible.mort()

    def mort(self):
        self.vivant = False

    def repro(self):
        if self.vivant == False:
            return None
        if (self.nourriture < self.rmax * self.REPRO_SEUIL): 
            return None
        if self.age < self.AGE_MATURE or self.age > self.AGE_VIEU:
            return None
        if self.trepro > self.REPRO_PAUSE:
            cede = self.nourriture * (1 - self.REPRO_GARDE) * 0.5  # On transmet 50% de l'energie perdu pour la repro a l'enfant
            self.nourriture *= self.REPRO_GARDE
            self.trepro = 0
            dx, dy = rd.choice([(0,1),(1,0),(-1,0),(0,-1)])
            pos_enf = ( (self.pos[0] + dx) % Taille_carte, (self.pos[1] + dy) % Taille_carte)          
            return type(self)(pos_enf, self.espece, self.vitesse, self.vision, self.rmax, cede) # On transmet toutes les caracs à l'enfant et 20% des reserves initiales

        return None

class herbivore(animale):       # classe heritiere de animale
    COUT_DEP     = 0.015    # cout metabolique : ~1 plante tous les 5-6 jours
    AGE_MATURE   = 4
    AGE_VIEU     = 12
    ESPERANCE    = 18       # -> plus aucun lapin ne depasse ~20 ans
    SENES_K      = 8.0
    SENES_P      = 2.7
    REPRO_AGE    = 6
    REPRO_SEUIL  = 0.8
    REPRO_GARDE  = 0.5
    REPRO_PAUSE  = 2
    NOURRITURE_INIT = 0.4   # fraction de rmax a la naissance (fondateurs seulement)

    def __init__(self, pos, espece, vitesse, vision, rmax, nourriture=None):
        n = rmax*self.NOURRITURE_INIT if nourriture == None else nourriture
        super().__init__(pos, espece, vitesse, vision, rmax, n)

    def action(self, plante_list):
        if self.nourriture <= 0:   # Si affame ou trop vieux = mort
            self.mort()
            return None
        
        self.deplacement(plante_list)      # deplacement de n cases visant une plante (s'il y en a a porte)

        self.age += 1
        self.trepro += 1    
        return self.repro()

class carnivore(animale):       # classe heritiere de animale
    COUT_DEP      = 0.015   # cout metabolique : ~1 proie tous les 4-5 jours
    AGE_MATURE    = 8
    AGE_VIEU      = 30
    ESPERANCE     = 42      # -> plus aucun renard ne depasse ~45 ans
    SENES_K       = 3.25
    SENES_P       = 4.0
    REPRO_AGE     = 8      # maturite tardive : c'est ce qui empeche l'explosion
    REPRO_SEUIL   = 0.8
    REPRO_GARDE   = 0.6
    REPRO_PAUSE   = 5       # une portee tous les 5 jours maximum
    SUCCES_CHASSE = 0.5     # 20% des attaques echouent (refuge pour les proies)
    PART_CORPS    = 0.5    # une proie vaut la moitie de son rmax + ses reserves
    NOURRITURE_INIT = 0.8   # fraction de rmax a la naissance (fondateurs seulement)

    def __init__(self, pos, espece, vitesse, vision, rmax, nourriture=None):
        n = rmax*self.NOURRITURE_INIT if nourriture == None else nourriture
        super().__init__(pos, espece, vitesse, vision, rmax, n)

    def action(self, herbivore_list):
        if self.nourriture <= 0:   # Si affame ou trop vieux = mort
            self.mort()
            return None

        self.deplacement(herbivore_list)      # deplacement de n cases visant une plante (s'il y en a a porte)

        self.age += 1
        self.trepro += 1
        return self.repro()

#%% Initialisation et fonctions
def initialisation(nplante, nherbivore, ncarnivore):
    plante_list =       [plante((rd.randint(0,Taille_carte - 1), rd.randint(0,Taille_carte - 1))) for _ in range(nplante)]
    herbivore_list =    [herbivore((rd.randint(0,Taille_carte - 1), rd.randint(0,Taille_carte - 1)), 'lapin', 3, 9, 20) for _ in range(nherbivore)]
    carnivore_list =    [carnivore((rd.randint(0,Taille_carte - 1), rd.randint(0,Taille_carte - 1)), 'renard', 3, 12, 30) for _ in range(ncarnivore)]
    return plante_list, herbivore_list, carnivore_list

def Simulation(plante_list, herbivore_list, carnivore_list):
    add_herbivore = []
    for herbivore in herbivore_list: #action des herbivores
        if herbivore.vivant:
            enfant = herbivore.action(plante_list)
            if enfant is not None:
                add_herbivore.append(enfant)

    add_carnivore = []
    for carnivore in carnivore_list: #action des carnivores
        if carnivore.vivant:
            enfant = carnivore.action(herbivore_list)
            if enfant is not None:
                add_carnivore.append(enfant)

    plante_list[:] = [p for p in plante_list if p.vivant]
    herbivore_list[:] = [h for h in herbivore_list if h.vivant] + add_herbivore 
    carnivore_list[:] = [c for c in carnivore_list if c.vivant] + add_carnivore

def dep_age(age, age_mature, age_vieu, esperance, K, p, cout_jeune): 
    """ 
    dependance du cout deplacement à l'age
    moins efficace jeune
    normal adulte
    et on redeviens moins efficace quand on est vieu 
    """

    if age < age_mature:
        return cout_jeune - (cout_jeune - 1) * (age / age_mature)
    elif age <= age_vieu:
        return 1
    else:
        prog = (age - age_vieu) / (esperance - age_vieu)
        return 1 + K * prog ** p


def proportion_reserve_proie(r_proie, r = 0.6):
    """
    energie transmise lors de la digestion (bcp de perte chaleur, non digere ...)
    """
    return r_proie * r


def repousse_plantes(plante_list, plante_max, taux_repousse):
    """
    réapparition de plantes chaque jour
    """
    if taux_repousse <= 0:
        return
    manque = plante_max - len(plante_list)
    if manque > 0:
        nb_nouvelles = int(manque * taux_repousse)
        for _ in range(nb_nouvelles):
            plante_list.append(plante((rd.randint(0, Taille_carte - 1), rd.randint(0, Taille_carte - 1))))


#%% Interface graphique par IA
def run_simulation(nplante, nherbivore, ncarnivore, NB_JOURS,
                    plante_max=None, taux_repousse=0.0):
    plante_list, herbivore_list, carnivore_list = initialisation(nplante, nherbivore, ncarnivore)
    if plante_max is None:
        plante_max = nplante
 
    def snapshot(jour):
        return {
            "jour": jour,
            "plantes_pos": [p.pos for p in plante_list],
            "herbivores": [(h.pos, h.age, h.nourriture) for h in herbivore_list],
            "carnivores": [(c.pos, c.age, c.nourriture) for c in carnivore_list],
        }
 
    history = [snapshot(0)]
 
    for jour in range(1, NB_JOURS + 1):
        Simulation(plante_list, herbivore_list, carnivore_list)
        repousse_plantes(plante_list, plante_max, taux_repousse)
        history.append(snapshot(jour))
 
        if not herbivore_list and not carnivore_list:
            break  # extinction totale, inutile de continuer
 
    return history
 
 
# ============================================================
#                    DASHBOARD INTERACTIF
# ============================================================
def lancer_dashboard(history):
    NB_JOURS = len(history) - 1
 
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], height_ratios=[2, 1, 1],
                           left=0.06, right=0.97, top=0.94, bottom=0.16,
                           hspace=0.55, wspace=0.3)
 
    ax_map = fig.add_subplot(gs[:, 0])
    ax_pop = fig.add_subplot(gs[0, 1])
    ax_age = fig.add_subplot(gs[1, 1])
    ax_food = fig.add_subplot(gs[2, 1])
    ax_age2 = ax_age.twinx()
    ax_food2 = ax_food.twinx()
 
    # ---- Carte ----
    ax_map.set_xlim(0, Taille_carte)
    ax_map.set_ylim(0, Taille_carte)
    ax_map.set_aspect("equal")
    scat_p = ax_map.scatter([], [], c="forestgreen", s=12, marker="s", label="Plantes")
    scat_h = ax_map.scatter([], [], c="royalblue", s=28, label="Herbivores")
    scat_c = ax_map.scatter([], [], c="firebrick", s=38, label="Carnivores")
    ax_map.legend(loc="upper right", fontsize=8)
 
    # ---- Évolution des populations (tracée une seule fois, en entier) ----
    jours = [h["jour"] for h in history]
    nb_p = [len(h["plantes_pos"]) for h in history]
    nb_h = [len(h["herbivores"]) for h in history]
    nb_c = [len(h["carnivores"]) for h in history]
 
    ax_pop.plot(jours, nb_p, color="forestgreen", label="Plantes")
    ax_pop.plot(jours, nb_h, color="royalblue", label="Herbivores")
    ax_pop.set_title("Évolution des populations", fontsize=9)
 
    ax2 = ax_pop.twinx()
    ax2.plot(jours, nb_c, color="firebrick", label="Carnivores")
 
    lines1, labels1 = ax_pop.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_pop.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
 
    curseur_pop = ax_pop.axvline(0, color="black", lw=1, ls="--")
    curseur_pop2 = ax2.axvline(0, color="black", lw=1, ls="--")
 
    etat = {"jour": 0, "en_lecture": False}
 
    def maj_affichage(jour):
        jour = max(0, min(NB_JOURS, jour))
        etat["jour"] = jour
        h = history[jour]
 
        xp, yp = (zip(*h["plantes_pos"]) if h["plantes_pos"] else ([], []))
        xh, yh = (zip(*[pos for pos, age, nrj in h["herbivores"]]) if h["herbivores"] else ([], []))
        xc, yc = (zip(*[pos for pos, age, nrj in h["carnivores"]]) if h["carnivores"] else ([], []))
 
        scat_p.set_offsets(np.column_stack([xp, yp]) if xp else np.empty((0, 2)))
        scat_h.set_offsets(np.column_stack([xh, yh]) if xh else np.empty((0, 2)))
        scat_c.set_offsets(np.column_stack([xc, yc]) if xc else np.empty((0, 2)))
 
        ax_map.set_title(
            f"Jour {jour} — {len(xp)} plantes, {len(xh)} herbivores, {len(xc)} carnivores",
            fontsize=10,
        )
        curseur_pop.set_xdata([jour, jour])
        curseur_pop2.set_xdata([jour, jour])
 
        # histogrammes âge / nourriture (recréés à chaque frame)
        ax_age.clear()
        ax_age2.clear()

        ax_age.set_ylim(0, 120)
        ax_age2.set_ylim(0, 30)

        ages_h = [age for pos, age, nrj in h["herbivores"]]
        ages_c = [age for pos, age, nrj in h["carnivores"]]
        if ages_h:
            ax_age.hist(ages_h, bins=15, alpha=0.6, color="royalblue", label="Herbivores")
        ax_age.set_title("Répartition des âges", fontsize=9)
        ax_age.legend(fontsize=7, loc = 2)
        if ages_c:
            ax_age2.hist(ages_c, bins=15, alpha=0.6, color="firebrick", label="Carnivores")
        ax_age2.legend(fontsize=7, loc = 1)

 
        ax_food.clear()
        ax_food2.clear()

        ax_food.set_ylim(0, 120)
        ax_food2.set_ylim(0, 30)

        nrj_h = [nrj for pos, age, nrj in h["herbivores"]]
        nrj_c = [nrj for pos, age, nrj in h["carnivores"]]
        if nrj_h:
            ax_food.hist(nrj_h, bins=15, alpha=0.6, color="royalblue", label="Herbivores")
        ax_food.set_title("Réserves de nourriture", fontsize=9)
        ax_food.legend(fontsize=7, loc = 2)
        if nrj_c:
            ax_food2.hist(nrj_c, bins=15, alpha=0.6, color="firebrick", label="Carnivores")
        ax_food2.legend(fontsize=7, loc = 1)

        slider.eventson = False
        slider.set_val(jour)
        slider.eventson = True
 
        fig.canvas.draw_idle()
 
    # ---- Widgets ----
    ax_slider = fig.add_axes([0.08, 0.06, 0.6, 0.03])
    slider = Slider(ax_slider, "Jour", 0, NB_JOURS, valinit=0, valstep=1)
 
    ax_play = fig.add_axes([0.72, 0.05, 0.08, 0.05])
    btn_play = Button(ax_play, "Lecture")
 
    ax_prev = fig.add_axes([0.81, 0.05, 0.06, 0.05])
    btn_prev = Button(ax_prev, "<")
 
    ax_next = fig.add_axes([0.88, 0.05, 0.06, 0.05])
    btn_next = Button(ax_next, ">")
 
    timer = fig.canvas.new_timer(interval=150)
 
    def on_timer():
        if etat["jour"] >= NB_JOURS:
            etat["en_lecture"] = False
            btn_play.label.set_text("Lecture")
            timer.stop()
            return
        maj_affichage(etat["jour"] + 1)
 
    timer.add_callback(on_timer)
 
    def toggle_play(event):
        etat["en_lecture"] = not etat["en_lecture"]
        if etat["en_lecture"]:
            btn_play.label.set_text("Pause")
            timer.start()
        else:
            btn_play.label.set_text("Lecture")
            timer.stop()
 
    def step_prev(event):
        etat["en_lecture"] = False
        btn_play.label.set_text("Lecture")
        timer.stop()
        maj_affichage(etat["jour"] - 1)
 
    def step_next(event):
        etat["en_lecture"] = False
        btn_play.label.set_text("Lecture")
        timer.stop()
        maj_affichage(etat["jour"] + 1)
 
    def on_slide(val):
        etat["en_lecture"] = False
        btn_play.label.set_text("Lecture")
        timer.stop()
        maj_affichage(int(val))
 
    btn_play.on_clicked(toggle_play)
    btn_prev.on_clicked(step_prev)
    btn_next.on_clicked(step_next)
    slider.on_changed(on_slide)
 
    maj_affichage(0)
    plt.show()
    return fig
 
 
# ============================================================
#                          LANCEMENT
# ============================================================
#%% Affichage graphique fait pas Claude
Taille_carte = 100
NB_JOURS = 400          # nombre de jours simulés
 
if __name__ == "__main__":
    history = run_simulation(
        nplante=600, nherbivore=100, ncarnivore=25,
        NB_JOURS=NB_JOURS,
        plante_max=800,       # capacité de charge (nombre max de plantes)
        taux_repousse=0.22,   # fraction du "manque" qui repousse chaque jour
    )
    lancer_dashboard(history)
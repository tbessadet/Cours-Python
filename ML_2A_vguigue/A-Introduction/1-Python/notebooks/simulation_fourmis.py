import random as rd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class Fourmis:
    def __init__(self, n, x=0, y=0):
        self.n = n
        self.x = x
        self.y = y
        # On garde l'historique de toutes les positions successives
        self.historique = [(x, y)]

    def deplacement(self):
        self.x += rd.randint(-1, 1)
        self.y += rd.randint(-1, 1)
        self.historique.append((self.x, self.y))


N_FOURMIS = 100
N_ETAPES = 300

# On stocke directement les OBJETS fourmis (pas des tuples figés comme dans
# ton dict "Pos" original, qui capturait x et y au moment de la création
# et ne bougeait donc jamais)
fourmis_dict = {k: Fourmis(k) for k in range(N_FOURMIS)}

# --- 300 déplacements pour chaque fourmi ---
for etape in range(N_ETAPES):
    for f in fourmis_dict.values():
        f.deplacement()

# --- Position finale d'une fourmi k, en une ligne ---
k = 5
print(f"Position finale de la fourmi {k} :", (fourmis_dict[k].x, fourmis_dict[k].y))
# équivalent : fourmis_dict[k].historique[-1]

# --- Interface graphique avec slider (étape 0 à 300) ---
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.2)


def positions_a_l_etape(etape):
    xs = [f.historique[etape][0] for f in fourmis_dict.values()]
    ys = [f.historique[etape][1] for f in fourmis_dict.values()]
    return xs, ys


xs0, ys0 = positions_a_l_etape(0)
scatter = ax.scatter(xs0, ys0, s=20)
ax.set_xlim(-N_ETAPES / 3, N_ETAPES / 3)
ax.set_ylim(-N_ETAPES / 3, N_ETAPES / 3)
ax.set_title("Étape 0")
ax.set_aspect("equal")

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, "Étape", 0, N_ETAPES, valinit=0, valstep=1)


def update(val):
    etape = int(slider.val)
    xs, ys = positions_a_l_etape(etape)
    scatter.set_offsets(list(zip(xs, ys)))
    ax.set_title(f"Étape {etape}")
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()


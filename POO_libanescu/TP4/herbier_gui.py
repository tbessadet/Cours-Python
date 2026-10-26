# -*- coding: utf-8 -*-
"""
Herbier - interface graphique Tkinter
=====================================
Onglets : Accueil (ajout, sauvegarde/chargement du .csv) | Plantothèque | Quizz

Fichier CSV attendu (colonnes) : nom, nom_scientifique, famille, cycle, besoins, photo
  - besoins : plusieurs besoins séparés par ";"   (ex. soleil;arrosage faible)
  - photo   : nom du fichier image (ex. tomate.jpg), cherché à côté du CSV ou dans un dossier "photos"

Pour afficher les .jpg : pip install pillow   (sans Pillow, seuls les .png/.gif s'affichent)
"""
import csv
import os
import random
import unicodedata

# ============================================================================
#  PARAMÈTRES
# ============================================================================
DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_DEFAUT = os.path.join(DOSSIER, "herbier.csv")

COLONNES = ["nom", "nom_scientifique", "famille", "cycle", "besoins", "photo"]
TITRES = ["Nom", "Nom scientifique", "Famille", "Cycle", "Besoins", "Photo"]

PHOTO_VIDE = "aucune_photo.png"               # "null" des photos manquantes
VALEURS_VIDES = ("", "nan", "none", "null")
EXT_IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

TOUS = "(tous)"                                # valeur "pas de filtre" des menus déroulants
FILTRE_INSTANTANE = True                       # True : la liste change dès qu'on choisit | False : bouton "Filtrer"
SAUVEGARDE_AUTO = True                         # True : ajout/suppression écrits tout de suite dans le .csv courant
NB_QUESTIONS_QUIZZ = 5                         # longueur du quizz rapide

# Noms de familles qui ne suivent pas la règle français -acées -> latin -aceae
FAMILLES_EXCEPTIONS = {
    "compositae": "asteraceae", "composees": "asteraceae",
    "cruciferae": "brassicaceae", "cruciferes": "brassicaceae",
    "leguminosae": "fabaceae", "legumineuses": "fabaceae",
    "umbelliferae": "apiaceae", "ombelliferes": "apiaceae",
    "labiatae": "lamiaceae", "labiees": "lamiaceae",
    "gramineae": "poaceae", "graminees": "poaceae",
    "renonculacees": "ranunculaceae", "renonculacee": "ranunculaceae",
}


# ============================================================================
#  NETTOYAGE DES DONNÉES (même logique que la méthode `nettoyer` de l'herbier)
# ============================================================================
def sans_accent(texte):
    texte = texte.replace("œ", "oe").replace("æ", "ae")
    texte = unicodedata.normalize("NFD", texte)
    return "".join(c for c in texte if not unicodedata.combining(c))


def famille_latine(famille):
    """famille : déjà en minuscules et sans accent. Solanacées -> solanaceae."""
    if famille in FAMILLES_EXCEPTIONS:
        return FAMILLES_EXCEPTIONS[famille]
    for fin in ("acees", "acee"):
        if famille.endswith(fin):
            return famille[:-len(fin)] + "aceae"
    return famille


def corriger_encodage(mot):
    if any(t in mot for t in ("Ã", "â", "Â", "ï¿½", "ðŸ", "�")):
        for enc in ("latin-1", "cp1252"):
            try:
                return mot.encode(enc).decode("utf-8")
            except Exception:
                pass
    return mot


def nettoyer_ligne(ligne):
    """Renvoie (ligne_propre, colonnes_manquantes, photo_absente)."""
    propre, manquants, photo_absente = [], [], False
    for idx in range(len(COLONNES)):
        mot = str(ligne[idx]) if idx < len(ligne) else ""
        mot = corriger_encodage(mot.strip()).strip().replace("\ufeff", "")
        if idx != 5:                                           # tout sauf la photo
            mot = sans_accent(mot.lower()).replace("_", " ")
            mot = " ".join(mot.split())
            if idx == 4:                                       # besoins : "a ; b" -> "a;b"
                mot = ";".join(b.strip() for b in mot.split(";") if b.strip())
            if idx == 2:
                mot = famille_latine(mot)
            if mot in VALEURS_VIDES:
                mot = ""
                manquants.append(COLONNES[idx])
            if idx in (0, 1, 2):
                mot = mot[:1].upper() + mot[1:]
        elif mot.lower() in VALEURS_VIDES + (PHOTO_VIDE.lower(),):
            mot = PHOTO_VIDE
            photo_absente = True
        propre.append(mot)
    return propre, manquants, photo_absente


def nettoyer_tout(lignes):
    """Nettoie une liste de lignes, supprime les doublons. Renvoie (lignes, rapport)."""
    rows, vus = [], set()
    rapport = {"doublons": [], "manquants": [], "photos": []}
    for i, l in enumerate(lignes, start=2):                    # numéro de ligne dans le CSV (entête = 1)
        propre, manquants, photo_absente = nettoyer_ligne(l)
        if manquants:
            rapport["manquants"].append(f"ligne {i} : {', '.join(manquants)}")
        if photo_absente:
            rapport["photos"].append(i)
        cle = tuple(propre)
        if cle in vus:
            rapport["doublons"].append(i)
            continue
        vus.add(cle)
        rows.append(propre)
    return rows, rapport


def fmt_besoins(texte):
    return ", ".join(b for b in texte.split(";") if b)


# ============================================================================
#  BASE DE DONNÉES (liste de lignes + fichier CSV)
# ============================================================================
class Base:
    def __init__(self):
        self.rows = []
        self.chemin = FICHIER_DEFAUT

    def charger(self, chemin):
        with open(chemin, newline="", encoding="utf-8-sig", errors="replace") as f:
            lignes = [l for l in csv.reader(f) if any(c.strip() for c in l)]
        if lignes and lignes[0][0].strip().lower() == "nom":    # on retire l'entête
            lignes = lignes[1:]
        self.rows, rapport = nettoyer_tout(lignes)
        self.chemin = chemin
        return rapport

    def sauvegarder(self, chemin=None):
        chemin = chemin or self.chemin
        with open(chemin, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLONNES)
            w.writerows(self.rows)
        self.chemin = chemin

    def valeurs(self, col):
        """Valeurs distinctes d'une colonne (les besoins sont éclatés un par un)."""
        if col == 4:
            items = {b for r in self.rows for b in r[4].split(";") if b}
        else:
            items = {r[col] for r in self.rows if r[col]}
        return sorted(items, key=str.lower)

    def existe(self, ligne):
        return tuple(ligne) in {tuple(r) for r in self.rows}

    def ajouter(self, ligne):
        self.rows.append(ligne)

    def supprimer(self, ligne):
        self.rows = [r for r in self.rows if r is not ligne]

    def dossiers_photos(self):
        d = os.path.dirname(os.path.abspath(self.chemin))
        return [d, os.path.join(d, "photos"), DOSSIER, os.path.join(DOSSIER, "photos"), os.getcwd()]

    def trouver_photo(self, fichier):
        if not fichier or fichier == PHOTO_VIDE:
            return None
        if os.path.isabs(fichier):
            return fichier if os.path.isfile(fichier) else None
        for d in self.dossiers_photos():
            p = os.path.join(d, fichier)
            if os.path.isfile(p):
                return p
        return None


# ============================================================================
#  GÉNÉRATION DES QUESTIONS DU QUIZZ (sans interface)
# ============================================================================
def placer(bonne, autres, evite=None):
    """Place la bonne réponse parmi 3 autres, jamais à la position `evite` (= celle de la question d'avant)."""
    positions = [i for i in range(4) if i != evite] or [0, 1, 2, 3]
    pos = random.choice(positions)
    autres = random.sample(autres, len(autres))
    return autres[:pos] + [bonne] + autres[pos:], pos


def _q_famille(rows, p):
    autres = sorted({r[2] for r in rows if r[2] and r[2] != p[2]})
    if len(autres) < 3:
        return None
    return f"{p[0]} ({p[1]})\nQuelle est sa famille botanique ?", p[2], random.sample(autres, 3)


def _q_plante(rows, fam):
    dans = {r[0] for r in rows if r[2] == fam and r[0]}
    hors = sorted({r[0] for r in rows if r[0] and r[0] not in dans})
    if not dans or len(hors) < 3:
        return None
    return (f"Famille : {fam}\nQuelle plante appartient à cette famille ?",
            random.choice(sorted(dans)), random.sample(hors, 3))


def _q_besoins(rows, p):
    if not p[4]:
        return None

    def ens(s):
        return frozenset(b for b in s.split(";") if b)

    mien, pool = ens(p[4]), {}
    for r in rows:
        e = ens(r[4])
        if e and e != mien:                       # besoins réellement différents (l'ordre ne compte pas)
            pool.setdefault(e, r[4])
    if len(pool) < 3:
        return None
    autres = [fmt_besoins(s) for s in random.sample(list(pool.values()), 3)]
    return f"{p[0]} ({p[1]})\nQuels sont ses besoins ?", fmt_besoins(p[4]), autres


def _q_photo(rows, p, photo_ok):
    if not photo_ok(p):
        return None
    fams = sorted({r[2] for r in rows if r[2] and r[2] != p[2]})
    noms = sorted({r[0] for r in rows if r[0] and r[0] != p[0]})
    options = (["famille"] if len(fams) >= 3 else []) + (["plante"] if len(noms) >= 3 else [])
    if not options:
        return None
    if random.choice(options) == "famille":
        return "À quelle famille appartient cette plante ?", p[2], random.sample(fams, 3)
    return "Quelle est cette plante ?", p[0], random.sample(noms, 3)


def generer_questions(base, n, familles=None, photo_ok=None):
    """
    Prépare jusqu'à n questions toutes différentes.
    familles : liste de familles à réviser (None = toutes).
    photo_ok : fonction(ligne) -> True si la photo de la plante peut être affichée.
    Renvoie une liste de dicts {texte, photo, choix (4), bonne (index)}.
    """
    photo_ok = photo_ok or (lambda ligne: False)
    rows = base.rows
    toutes = sorted({r[2] for r in rows if r[2]})
    cibles = [f for f in (familles or toutes) if f in toutes]
    if not cibles:
        return []

    questions, cles, vues, pos_prec, essais = [], set(), set(), None, 0
    while len(questions) < n and essais < 60 * n + 200:
        essais += 1
        fam = random.choice(cibles)                       # 1) une famille au hasard
        type_q = random.choice(["famille", "plante", "besoins", "photo"])
        if type_q == "plante":
            p, cle = None, ("plante", fam)
        else:
            plantes_fam = [r for r in rows if r[2] == fam and r[0] and r[1]]
            if not plantes_fam:
                continue
            p = random.choice(plantes_fam)                # 2) une plante de cette famille
            cle = (type_q, tuple(p))
            if essais <= 30 * n and tuple(p) in vues:     # on privilégie des plantes encore jamais posées
                continue
        if cle in cles:                                   # jamais deux fois la même question
            continue

        if type_q == "famille":
            q = _q_famille(rows, p)
        elif type_q == "plante":
            q = _q_plante(rows, fam)
        elif type_q == "besoins":
            q = _q_besoins(rows, p)
        else:
            q = _q_photo(rows, p, photo_ok)
        if q is None:
            continue

        texte, bonne, autres = q
        choix, pos = placer(bonne, autres, pos_prec)      # bonne réponse jamais 2 fois au même endroit de suite
        pos_prec = pos
        questions.append({"texte": texte, "photo": p, "choix": choix, "bonne": pos})
        cles.add(cle)
        if p:
            vues.add(tuple(p))
    return questions


# ===== INTERFACE ============================================================
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

VERT, ROUGE, NEUTRE = "#43a047", "#e53935", "#e3f2fd"


def charger_image(chemin, taille):
    try:
        if PIL_OK:
            img = Image.open(chemin)
            img.thumbnail(taille)
            return ImageTk.PhotoImage(img)
        if chemin.lower().endswith((".png", ".gif")):
            img = tk.PhotoImage(file=chemin)
            facteur = max(1, -(-img.width() // taille[0]), -(-img.height() // taille[1]))
            return img.subsample(facteur) if facteur > 1 else img
    except Exception:
        pass
    return None


def liste_courte(elements, maxi=15):
    txt = ", ".join(str(e) for e in elements[:maxi])
    return txt + (" …" if len(elements) > maxi else "")


def entree_sans_chiffre(parent, variable, on_refus, width=36):
    """Champ texte qui refuse les chiffres (nom, famille, cycle...)."""
    return ttk.Entry(
        parent, textvariable=variable, width=width, validate="key",
        validatecommand=(parent.register(lambda P: not any(c.isdigit() for c in P)), "%P"),
        invalidcommand=(parent.register(on_refus),))


# ----------------------------------------------------------------------------
#  Widget : menu déroulant (valeur existante) OU champ texte (nouvelle valeur)
# ----------------------------------------------------------------------------
class ChoixOuTexte(ttk.Frame):
    """Un seul des deux est actif : choisir dans la liste bloque le texte, et inversement."""

    def __init__(self, parent, valeurs, multiple=False, on_refus=lambda: None):
        super().__init__(parent)
        self.valeurs, self.multiple, self.choix = valeurs, multiple, []
        self.var_combo = tk.StringVar()
        self.var_texte = tk.StringVar()
        self.var_resume = tk.StringVar(value="(aucun)")

        ttk.Label(self, text="Existant :").grid(row=0, column=0, sticky="w")
        if multiple:
            self.selecteur = ttk.Button(self, text="Cocher dans la liste ▼", command=self._ouvrir_popup)
            ttk.Label(self, textvariable=self.var_resume, foreground="#555",
                      wraplength=260).grid(row=2, column=1, sticky="w", padx=4)
        else:
            self.selecteur = ttk.Combobox(self, textvariable=self.var_combo, values=valeurs,
                                          state="readonly", width=33)
            self.selecteur.bind("<<ComboboxSelected>>", self._combo_choisi)
        self.selecteur.grid(row=0, column=1, sticky="w", padx=4, pady=1)

        ttk.Label(self, text="ou nouveau :").grid(row=1, column=0, sticky="w")
        self.entry = entree_sans_chiffre(self, self.var_texte, on_refus, width=36)
        self.entry.grid(row=1, column=1, sticky="w", padx=4, pady=1)

        ttk.Button(self, text="↺", width=3, command=self.reinitialiser).grid(row=0, column=2, rowspan=2, padx=6)
        self.var_texte.trace_add("write", lambda *a: self._maj_etats())

    def _maj_etats(self):
        a_choix = bool(self.choix)
        a_texte = bool(self.var_texte.get().strip())
        self.entry.configure(state="disabled" if a_choix else "normal")
        if self.multiple:
            self.selecteur.configure(state="disabled" if a_texte else "normal")
        else:
            self.selecteur.configure(state="disabled" if a_texte else "readonly")

    def _combo_choisi(self, _event=None):
        self.choix = [self.var_combo.get()] if self.var_combo.get() else []
        self._maj_etats()

    def _ouvrir_popup(self):
        fen = tk.Toplevel(self)
        fen.title("Cocher les besoins existants")
        fen.transient(self.winfo_toplevel())
        try:
            fen.grab_set()
        except tk.TclError:
            pass
        cases = {}
        if not self.valeurs:
            ttk.Label(fen, text="Aucune valeur existante pour l'instant.").pack(padx=20, pady=10)
        for v in self.valeurs:
            var = tk.BooleanVar(value=v in self.choix)
            ttk.Checkbutton(fen, text=v, variable=var).pack(anchor="w", padx=14, pady=2)
            cases[v] = var

        def valider():
            self.choix = [v for v, var in cases.items() if var.get()]
            self.var_resume.set(", ".join(self.choix) or "(aucun)")
            self._maj_etats()
            fen.destroy()
            try:
                self.winfo_toplevel().grab_set()          # on redonne la main à la fenêtre d'ajout
            except tk.TclError:
                pass

        ttk.Button(fen, text="OK", command=valider).pack(pady=10)

    def reinitialiser(self):
        self.choix = []
        self.var_combo.set("")
        self.var_texte.set("")
        self.var_resume.set("(aucun)")
        self._maj_etats()

    def est_texte(self):
        return not self.choix

    def valeur(self):
        return ";".join(self.choix) if self.choix else self.var_texte.get().strip()


# ----------------------------------------------------------------------------
#  Fenêtre "Ajouter une plante"
# ----------------------------------------------------------------------------
class FenetreAjout(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Ajouter une plante")
        self.transient(app)
        self.resizable(False, False)
        self.geometry(f"+{app.winfo_rootx() + 80}+{app.winfo_rooty() + 60}")
        base = app.base

        cadre = ttk.Frame(self, padding=14)
        cadre.pack(fill="both", expand=True)
        ttk.Label(cadre, foreground="#b9770e", wraplength=520, justify="left",
                  text="⚠ Avant de saisir un nouveau terme (famille, cycle, besoin), vérifiez d'abord "
                       "le menu déroulant : cela évite les doublons."
                  ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.var_msg = tk.StringVar()
        refus = lambda: self.var_msg.set("⚠ Les chiffres ne sont pas autorisés dans ce champ.")

        self.var_nom, self.var_sci, self.var_photo = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.famille = ChoixOuTexte(cadre, base.valeurs(2), on_refus=refus)
        self.cycle = ChoixOuTexte(cadre, base.valeurs(3), on_refus=refus)
        self.besoins = ChoixOuTexte(cadre, base.valeurs(4), multiple=True, on_refus=refus)

        photo = ttk.Frame(cadre)
        ttk.Entry(photo, textvariable=self.var_photo, width=36).pack(side="left")
        ttk.Button(photo, text="Parcourir…", command=self.parcourir).pack(side="left", padx=6)

        lignes = [
            ("Nom *", entree_sans_chiffre(cadre, self.var_nom, refus), None),
            ("Nom scientifique *", entree_sans_chiffre(cadre, self.var_sci, refus), "forme attendue : Genre espèce"),
            ("Famille *", self.famille, None),
            ("Cycle *", self.cycle, None),
            ("Besoins *", self.besoins, "plusieurs besoins : cochez-les, ou tapez-les séparés par « ; »"),
            ("Photo", photo, "facultatif (.png, .jpg…)"),
        ]
        r = 1
        for libelle, widget, aide in lignes:
            ttk.Label(cadre, text=libelle).grid(row=r, column=0, sticky="nw", pady=6, padx=(0, 10))
            widget.grid(row=r, column=1, sticky="w", pady=6)
            if aide:
                ttk.Label(cadre, text=aide, foreground="#777").grid(row=r + 1, column=1, sticky="w")
                r += 1
            r += 1

        ttk.Label(cadre, textvariable=self.var_msg, foreground="#c62828").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(8, 0))
        boutons = ttk.Frame(cadre)
        boutons.grid(row=r + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(boutons, text="Annuler", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(boutons, text="Ajouter la plante", command=self.valider).pack(side="left")

        self.after(100, self._prendre_la_main)

    def _prendre_la_main(self):
        try:
            self.grab_set()
            self.focus_set()
        except tk.TclError:
            pass

    def parcourir(self):
        p = filedialog.askopenfilename(
            parent=self, title="Choisir une photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.gif *.webp"), ("Tous les fichiers", "*.*")])
        if p:
            dossiers = [os.path.normcase(os.path.abspath(d)) for d in self.app.base.dossiers_photos()]
            if os.path.normcase(os.path.dirname(os.path.abspath(p))) in dossiers:
                p = os.path.basename(p)                   # dans un dossier connu : le nom suffit
            self.var_photo.set(p)
        self._prendre_la_main()

    def erreur(self, texte):
        self.var_msg.set("⚠ " + texte)

    def valider(self):
        nom, sci = self.var_nom.get().strip(), self.var_sci.get().strip()
        besoins = self.besoins.valeur().replace(",", ";")
        photo = self.var_photo.get().strip()
        champs = [("nom", nom, True), ("nom scientifique", sci, True),
                  ("famille", self.famille.valeur(), self.famille.est_texte()),
                  ("cycle", self.cycle.valeur(), self.cycle.est_texte()),
                  ("besoins", besoins, self.besoins.est_texte())]

        # --- garde-fous ---
        for libelle, valeur, saisi_a_la_main in champs:
            if not valeur:
                return self.erreur(f"Le champ « {libelle} » est obligatoire.")
            if saisi_a_la_main and any(c.isdigit() for c in valeur):
                return self.erreur(f"Le champ « {libelle} » ne doit pas contenir de chiffres.")
            if not any(c.isalpha() for c in valeur):
                return self.erreur(f"Le champ « {libelle} » doit contenir des lettres.")
        if len(sci.split()) < 2:
            return self.erreur("Le nom scientifique doit avoir la forme « Genre espèce ».")
        if photo:
            if not photo.lower().endswith(EXT_IMAGES):
                return self.erreur("La photo doit être un fichier image (.png, .jpg, .jpeg, .gif, .webp).")
            if not self.app.base.trouver_photo(photo):
                if not messagebox.askyesno("Photo introuvable",
                                           f"Le fichier « {photo} » est introuvable.\n"
                                           "Ajouter quand même la plante avec ce nom de photo ?", parent=self):
                    return

        ligne, _, _ = nettoyer_ligne([nom, sci, self.famille.valeur(), self.cycle.valeur(), besoins, photo])
        if self.app.base.existe(ligne):                   # comparaison APRÈS nettoyage (accents, casse, espaces...)
            return self.erreur("Cette plante existe déjà dans l'herbier.")

        self.app.base.ajouter(ligne)
        self.destroy()
        self.app.apres_modification(f"Plante « {ligne[0]} » ajoutée")


# ----------------------------------------------------------------------------
#  Tableau avec une croix rouge à la fin de chaque ligne
# ----------------------------------------------------------------------------
LARGEURS = [150, 180, 130, 90, 230, 130, 60]


class TableauPlantes(ttk.Frame):
    def __init__(self, parent, sur_suppression):
        super().__init__(parent)
        self.sur_suppression = sur_suppression

        entete = tk.Frame(self, bg="#2e7d32")
        entete.pack(fill="x", padx=(0, 17))               # 17 px ≈ largeur de la barre de défilement
        for i, (titre, larg) in enumerate(zip(TITRES + ["Supprimer"], LARGEURS)):
            tk.Label(entete, text=titre, bg="#2e7d32", fg="white", font=("Helvetica", 10, "bold"),
                     anchor="w", padx=6, pady=5).grid(row=0, column=i, sticky="nsew")
            entete.columnconfigure(i, minsize=larg, weight=1 if i < 6 else 0)

        zone = ttk.Frame(self)
        zone.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(zone, highlightthickness=0, bg="white")
        barre = ttk.Scrollbar(zone, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=barre.set)
        barre.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.interieur = tk.Frame(self.canvas, bg="white")
        self.fenetre = self.canvas.create_window((0, 0), window=self.interieur, anchor="nw")
        self.interieur.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.fenetre, width=e.width))
        for i, larg in enumerate(LARGEURS):
            self.interieur.columnconfigure(i, minsize=larg, weight=1 if i < 6 else 0)

        for evt in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(evt, self._molette, add="+")

    def _molette(self, e):
        try:
            sous_la_souris = self.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):
            return
        if sous_la_souris is None or not str(sous_la_souris).startswith(str(self.canvas)):
            return
        if e.num == 4:
            d = -1
        elif e.num == 5:
            d = 1
        else:
            d = -1 if e.delta > 0 else 1
        self.canvas.yview_scroll(d, "units")

    def afficher(self, lignes):
        for w in self.interieur.winfo_children():
            w.destroy()
        if not lignes:
            tk.Label(self.interieur, text="Aucune plante à afficher.", bg="white", fg="#777",
                     pady=24).grid(row=0, column=0, columnspan=7, sticky="ew")
        for i, r in enumerate(lignes):
            bg = "#f1f8e9" if i % 2 == 0 else "white"
            valeurs = [r[0], r[1], r[2], r[3], fmt_besoins(r[4]), "—" if r[5] == PHOTO_VIDE else r[5]]
            for j, (v, larg) in enumerate(zip(valeurs, LARGEURS)):
                police = ("Helvetica", 10, "italic") if j == 1 else ("Helvetica", 10)
                tk.Label(self.interieur, text=v, bg=bg, font=police, anchor="w", justify="left",
                         wraplength=larg - 12, padx=6, pady=5).grid(row=i, column=j, sticky="nsew")
            croix = tk.Label(self.interieur, text="✖", fg="#d32f2f", bg=bg, cursor="hand2",
                             font=("Helvetica", 13, "bold"))
            croix.grid(row=i, column=6, sticky="nsew")
            croix.bind("<Button-1>", lambda e, ligne=r: self.sur_suppression(ligne))
            croix.bind("<Enter>", lambda e, c=croix: c.configure(fg="#7f0000"))
            croix.bind("<Leave>", lambda e, c=croix: c.configure(fg="#d32f2f"))
        self.canvas.yview_moveto(0)


# ----------------------------------------------------------------------------
#  Onglet Plantothèque
# ----------------------------------------------------------------------------
class OngletPlantotheque(ttk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent)
        self.app = app

        barre = ttk.Frame(self, padding=(10, 8))
        barre.pack(fill="x")
        self.vars, self.combos = [], []
        for i in range(5):                                 # 5 filtres : nom, nom_scientifique, famille, cycle, besoins
            ttk.Label(barre, text=TITRES[i]).grid(row=0, column=i, sticky="w", padx=4)
            var = tk.StringVar(value=TOUS)
            cb = ttk.Combobox(barre, textvariable=var, values=[TOUS], state="readonly", width=19)
            cb.grid(row=1, column=i, padx=4)
            if FILTRE_INSTANTANE:
                cb.bind("<<ComboboxSelected>>", lambda e: self.appliquer())
            self.vars.append(var)
            self.combos.append(cb)

        col = 5
        if not FILTRE_INSTANTANE:
            ttk.Button(barre, text="Filtrer", command=self.appliquer).grid(row=1, column=col, padx=4)
            col += 1
        ttk.Button(barre, text="Réinitialiser", command=self.reinitialiser).grid(row=1, column=col, padx=4)
        ttk.Button(barre, text="➕ Ajouter une plante", command=app.ouvrir_ajout).grid(row=1, column=col + 1, padx=(16, 4))

        self.var_compte = tk.StringVar()
        ttk.Label(self, textvariable=self.var_compte, foreground="#555").pack(anchor="w", padx=14)
        self.tableau = TableauPlantes(self, app.demander_suppression)
        self.tableau.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    def reinitialiser(self):
        for v in self.vars:
            v.set(TOUS)
        self.appliquer()

    def rafraichir(self):
        for i, (var, cb) in enumerate(zip(self.vars, self.combos)):
            valeurs = [TOUS] + self.app.base.valeurs(i)
            cb.configure(values=valeurs)
            if var.get() not in valeurs:
                var.set(TOUS)
        self.appliquer()

    def appliquer(self):
        choix = [v.get() for v in self.vars]
        lignes = []
        for r in self.app.base.rows:
            ok = True
            for col, val in enumerate(choix):
                if val == TOUS:
                    continue
                if col == 4:
                    if val not in r[4].split(";"):
                        ok = False
                        break
                elif r[col] != val:
                    ok = False
                    break
            if ok:
                lignes.append(r)
        self.tableau.afficher(lignes)
        self.var_compte.set(f"{len(lignes)} plante(s) affichée(s) sur {len(self.app.base.rows)}")


# ----------------------------------------------------------------------------
#  Onglet Accueil (menu)
# ----------------------------------------------------------------------------
class OngletAccueil(ttk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent, padding=20)
        self.app = app
        self._dernier_chemin = None

        ttk.Label(self, text="🌿 Herbier", font=("Helvetica", 28, "bold")).pack(pady=(10, 2))
        self.var_info = tk.StringVar()
        ttk.Label(self, textvariable=self.var_info, foreground="#555").pack(pady=(0, 18))

        boutons = ttk.Frame(self)
        boutons.pack()
        ttk.Button(boutons, text="📚  Plantothèque", style="Gros.TButton",
                   command=lambda: app.aller(app.onglet_plantes)).grid(row=0, column=0, padx=8, pady=6)
        ttk.Button(boutons, text="➕  Ajouter une plante", style="Gros.TButton",
                   command=app.ouvrir_ajout).grid(row=0, column=1, padx=8, pady=6)
        ttk.Button(boutons, text="🎯  QUIZZ", style="Gros.TButton",
                   command=lambda: app.aller(app.onglet_quizz)).grid(row=0, column=2, padx=8, pady=6)

        cadre = ttk.LabelFrame(self, text="Sauvegarder / Charger l'herbier (fichier .csv)", padding=14)
        cadre.pack(pady=26, fill="x", padx=40)
        ttk.Label(cadre, text="Chemin du fichier :").grid(row=0, column=0, sticky="w")
        self.var_chemin = tk.StringVar()
        ttk.Entry(cadre, textvariable=self.var_chemin, width=60).grid(row=0, column=1, padx=8, sticky="ew")
        ttk.Button(cadre, text="Parcourir…", command=self.parcourir).grid(row=0, column=2)

        self.var_action = tk.StringVar(value="charger")
        choix = ttk.Frame(cadre)
        choix.grid(row=1, column=1, sticky="w", pady=8)
        ttk.Radiobutton(choix, text="Charger ce fichier", variable=self.var_action, value="charger").pack(side="left")
        ttk.Radiobutton(choix, text="Sauvegarder sous ce nom", variable=self.var_action,
                        value="sauvegarder").pack(side="left", padx=16)
        ttk.Button(cadre, text="Valider", command=self.valider).grid(row=1, column=2)
        cadre.columnconfigure(1, weight=1)

        if not PIL_OK:
            ttk.Label(self, foreground="#b9770e", wraplength=700, justify="center",
                      text="Pillow n'est pas installé : seules les photos .png et .gif s'afficheront "
                           "(pip install pillow pour les .jpg).").pack()

    def parcourir(self):
        if self.var_action.get() == "charger":
            p = filedialog.askopenfilename(title="Charger un herbier",
                                           filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")])
        else:
            p = filedialog.asksaveasfilename(title="Sauvegarder l'herbier", defaultextension=".csv",
                                             confirmoverwrite=False, filetypes=[("Fichiers CSV", "*.csv")])
        if p:
            self.var_chemin.set(p)

    def valider(self):
        chemin = self.var_chemin.get().strip().strip('"')
        if not chemin:
            messagebox.showwarning("Chemin manquant", "Entrez le chemin d'un fichier .csv.")
            return
        if self.var_action.get() == "charger":
            self.app.charger_fichier(chemin)
        else:
            self.app.sauvegarder_fichier(chemin)

    def rafraichir(self):
        base = self.app.base
        self.var_info.set(f"{len(base.rows)} plante(s) · {len(base.valeurs(2))} famille(s)")
        if base.chemin != self._dernier_chemin:
            self._dernier_chemin = base.chemin
            self.var_chemin.set(base.chemin)


# ----------------------------------------------------------------------------
#  Onglet Quizz
# ----------------------------------------------------------------------------
class OngletQuizz(ttk.Frame):
    def __init__(self, app, parent):
        super().__init__(parent)
        self.app = app
        self.questions, self.index, self.score = [], 0, 0
        self.reponse_donnee, self.params, self.img_ref = False, (None, NB_QUESTIONS_QUIZZ), None
        self._familles_affichees = None

        self.page_menu = ttk.Frame(self, padding=20)
        self.page_question = ttk.Frame(self, padding=20)
        self.page_fin = ttk.Frame(self, padding=20)
        self._construire_menu()
        self._construire_question()
        self._construire_fin()
        self.montrer(self.page_menu)

    def montrer(self, page):
        for p in (self.page_menu, self.page_question, self.page_fin):
            p.pack_forget()
        page.pack(fill="both", expand=True)

    # ---- page 1 : menu du quizz + mode révision
    def _construire_menu(self):
        p = self.page_menu
        ttk.Label(p, text="🎯 QUIZZ", font=("Helvetica", 22, "bold")).pack(pady=(0, 4))
        self.var_info = tk.StringVar()
        ttk.Label(p, textvariable=self.var_info, foreground="#555").pack()
        ttk.Label(p, text=f"Quizz rapide : {NB_QUESTIONS_QUIZZ} questions tirées au hasard dans tout l'herbier.").pack(pady=(14, 6))
        ttk.Button(p, text="▶  Lancer le quizz", style="Gros.TButton",
                   command=lambda: self.demarrer(None, NB_QUESTIONS_QUIZZ)).pack()

        rev = ttk.LabelFrame(p, text="Mode révision", padding=12)
        rev.pack(pady=20, fill="x", padx=60)
        ttk.Label(rev, text="Familles à réviser (Ctrl ou Maj pour en choisir plusieurs) :").pack(anchor="w")
        zone = ttk.Frame(rev)
        zone.pack(fill="x", pady=6)
        self.liste = tk.Listbox(zone, selectmode="extended", exportselection=False, height=7)
        barre = ttk.Scrollbar(zone, orient="vertical", command=self.liste.yview)
        self.liste.configure(yscrollcommand=barre.set)
        self.liste.pack(side="left", fill="x", expand=True)
        barre.pack(side="left", fill="y")

        bas = ttk.Frame(rev)
        bas.pack(fill="x", pady=(4, 0))
        ttk.Button(bas, text="Tout sélectionner",
                   command=lambda: self.liste.selection_set(0, "end")).pack(side="left")
        ttk.Button(bas, text="Tout désélectionner",
                   command=lambda: self.liste.selection_clear(0, "end")).pack(side="left", padx=6)
        self.var_nb = tk.StringVar(value="10")
        ttk.Spinbox(bas, from_=1, to=50, width=5, textvariable=self.var_nb, validate="key",
                    validatecommand=(self.register(lambda P: P.isdigit() or P == ""), "%P")).pack(side="right")
        ttk.Label(bas, text="Nombre de questions :").pack(side="right", padx=6)
        ttk.Button(rev, text="▶  Lancer la révision", command=self.lancer_revision).pack(pady=(10, 0))

    def lancer_revision(self):
        familles = [self.liste.get(i) for i in self.liste.curselection()]
        if not familles:
            messagebox.showwarning("Révision", "Choisissez au moins une famille dans la liste.")
            return
        try:
            n = int(self.var_nb.get())
            if not 1 <= n <= 50:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Révision", "Le nombre de questions doit être un entier entre 1 et 50.")
            return
        self.demarrer(familles, n)

    def rafraichir(self):
        base = self.app.base
        self.var_info.set(f"{len(base.rows)} plante(s) · {len(base.valeurs(2))} famille(s) dans l'herbier")
        familles = tuple(base.valeurs(2))
        if familles != self._familles_affichees:
            gardees = {self.liste.get(i) for i in self.liste.curselection()}
            self.liste.delete(0, "end")
            for i, f in enumerate(familles):
                self.liste.insert("end", f)
                if f in gardees:
                    self.liste.selection_set(i)
            self._familles_affichees = familles

    # ---- page 2 : une question
    def _construire_question(self):
        p = self.page_question
        self.var_entete = tk.StringVar()
        ttk.Label(p, textvariable=self.var_entete, font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.lbl_question = ttk.Label(p, text="", font=("Helvetica", 16, "bold"),
                                      wraplength=720, justify="center", anchor="center")
        self.lbl_question.pack(pady=10, fill="x")
        self.lbl_photo = tk.Label(p)
        self.lbl_photo.pack(pady=4)

        grille = ttk.Frame(p)
        grille.pack(pady=8, fill="x")
        grille.columnconfigure(0, weight=1, uniform="rep")
        grille.columnconfigure(1, weight=1, uniform="rep")
        self.boutons = []
        for i in range(4):
            b = tk.Label(grille, text="", font=("Helvetica", 12), bg=NEUTRE, relief="raised", bd=3,
                         wraplength=320, height=3, cursor="hand2")
            b.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            b.bind("<Button-1>", lambda e, k=i: self.repondre(k))
            self.boutons.append(b)

        self.lbl_retour = ttk.Label(p, text="", font=("Helvetica", 13, "bold"))
        self.lbl_retour.pack(pady=4)
        bas = ttk.Frame(p)
        bas.pack(fill="x", pady=6)
        ttk.Button(bas, text="Abandonner", command=self.abandonner).pack(side="left")
        self.btn_suivant = ttk.Button(bas, text="Question suivante ▶", state="disabled", command=self.suivante)
        self.btn_suivant.pack(side="right")

    def demarrer(self, familles, n):
        self.params = (familles, n)
        questions = generer_questions(self.app.base, n, familles,
                                      photo_ok=lambda ligne: self.app.image(ligne) is not None)
        if not questions:
            messagebox.showwarning(
                "Quizz impossible",
                "Pas assez de données pour poser des questions.\n"
                "Il faut au moins 4 familles différentes et quelques plantes "
                "(avec besoins, et photos pour les questions-photo).")
            return
        if len(questions) < n:
            messagebox.showinfo(
                "Quizz raccourci",
                f"Seulement {len(questions)} question(s) différentes possibles avec l'herbier actuel "
                f"(au lieu de {n}).")
        self.questions, self.index, self.score = questions, 0, 0
        self.montrer(self.page_question)
        self.afficher_question()

    def _maj_entete(self):
        self.var_entete.set(f"Question {self.index + 1} / {len(self.questions)}        Score : {self.score}")

    def afficher_question(self):
        q = self.questions[self.index]
        self.reponse_donnee = False
        self._maj_entete()
        self.lbl_question.configure(text=q["texte"])
        img = self.app.image(q["photo"]) if q["photo"] else None
        self.img_ref = img                                 # garder une référence, sinon l'image disparaît
        if img is not None:
            self.lbl_photo.configure(image=img)
        else:
            self.lbl_photo.configure(image="")
        for i, b in enumerate(self.boutons):
            b.configure(text=q["choix"][i], bg=NEUTRE, fg="black", relief="raised", bd=3, cursor="hand2")
        self.lbl_retour.configure(text="")
        derniere = self.index == len(self.questions) - 1
        self.btn_suivant.configure(state="disabled", text="Voir le score ▶" if derniere else "Question suivante ▶")

    def repondre(self, i):
        if self.reponse_donnee:
            return
        self.reponse_donnee = True
        q = self.questions[self.index]
        bonne = q["bonne"]
        if i == bonne:
            self.score += 1
            self.lbl_retour.configure(text="Bonne réponse !", foreground=VERT)
        else:
            self.lbl_retour.configure(text=f"La réponse était : {q['choix'][bonne]}", foreground=ROUGE)
        for k, b in enumerate(self.boutons):
            b.configure(bg=VERT if k == bonne else ROUGE, fg="white", cursor="arrow",
                        relief="sunken" if k == i else "flat", bd=4 if k == i else 3)
        self._maj_entete()
        self.btn_suivant.configure(state="normal")

    def suivante(self):
        self.index += 1
        if self.index >= len(self.questions):
            self.fin()
        else:
            self.afficher_question()

    def abandonner(self):
        if messagebox.askyesno("Abandonner", "Quitter le quizz en cours ?"):
            self.montrer(self.page_menu)

    # ---- page 3 : score final
    def _construire_fin(self):
        p = self.page_fin
        ttk.Label(p, text="Quizz terminé !", font=("Helvetica", 22, "bold")).pack(pady=(30, 10))
        self.var_score = tk.StringVar()
        ttk.Label(p, textvariable=self.var_score, font=("Helvetica", 16)).pack(pady=6)
        self.var_bilan = tk.StringVar()
        ttk.Label(p, textvariable=self.var_bilan, foreground="#555").pack(pady=4)
        ttk.Label(p, text="Voulez-vous recommencer ?", font=("Helvetica", 12)).pack(pady=(24, 8))
        boutons = ttk.Frame(p)
        boutons.pack()
        ttk.Button(boutons, text="🔁  Recommencer", style="Gros.TButton",
                   command=lambda: self.demarrer(*self.params)).pack(side="left", padx=8)
        ttk.Button(boutons, text="Menu du quizz", style="Gros.TButton",
                   command=lambda: self.montrer(self.page_menu)).pack(side="left", padx=8)

    def fin(self):
        n = len(self.questions)
        self.var_score.set(f"Score final : {self.score} / {n}")
        if self.score == n:
            self.var_bilan.set("Sans faute, bravo ! 🌟")
        elif self.score >= n / 2:
            self.var_bilan.set("Bien joué, continuez comme ça !")
        else:
            self.var_bilan.set("Courage, le mode révision est là pour ça.")
        self.montrer(self.page_fin)


# ----------------------------------------------------------------------------
#  Application
# ----------------------------------------------------------------------------
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌿 Herbier")
        self.geometry("1020x700")
        self.minsize(880, 580)
        self.base = Base()
        self.modifie = False
        self.cache_images = {}
        self.var_statut = tk.StringVar(value="Prêt.")

        style = ttk.Style(self)
        style.configure("Gros.TButton", font=("Helvetica", 12), padding=10)

        barre = ttk.Frame(self)
        barre.pack(side="bottom", fill="x")
        ttk.Label(barre, textvariable=self.var_statut, foreground="#444", padding=(10, 4)).pack(side="left")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        self.onglet_accueil = OngletAccueil(self, self.notebook)
        self.onglet_plantes = OngletPlantotheque(self, self.notebook)
        self.onglet_quizz = OngletQuizz(self, self.notebook)
        self.notebook.add(self.onglet_accueil, text="  Accueil  ")
        self.notebook.add(self.onglet_plantes, text="  Plantothèque  ")
        self.notebook.add(self.onglet_quizz, text="  Quizz  ")
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.rafraichir_tout())

        self.protocol("WM_DELETE_WINDOW", self.quitter)
        self.after(150, self.chargement_initial)

    # ---- utilitaires
    def statut(self, texte):
        self.var_statut.set(texte)

    def aller(self, onglet):
        self.notebook.select(onglet)

    def rafraichir_tout(self):
        self.onglet_accueil.rafraichir()
        self.onglet_plantes.rafraichir()
        self.onglet_quizz.rafraichir()

    def image(self, ligne, taille=(260, 260)):
        """Image Tk de la photo d'une plante (ou None si absente / illisible)."""
        chemin = self.base.trouver_photo(ligne[5])
        if not chemin:
            return None
        cle = (chemin, taille)
        if cle not in self.cache_images:
            self.cache_images[cle] = charger_image(chemin, taille)
        return self.cache_images[cle]

    # ---- fichier
    def chargement_initial(self):
        chemin = self.base.chemin
        if not os.path.isfile(chemin):
            self.rafraichir_tout()
            self.statut(f"« {os.path.basename(chemin)} » introuvable : herbier vide "
                        "(le fichier sera créé à la première modification).")
            return
        try:
            rapport = self.base.charger(chemin)
        except (OSError, csv.Error) as e:
            messagebox.showerror("Erreur de lecture", f"Impossible de lire le fichier :\n{e}")
            return
        self.rafraichir_tout()
        self.statut(f"{len(self.base.rows)} plante(s) chargée(s) depuis {chemin}")
        if any(rapport[k] for k in ("doublons", "manquants", "photos")):
            self.afficher_rapport(rapport, chemin)

    def afficher_rapport(self, rapport, chemin):
        texte = [f"{len(self.base.rows)} plante(s) chargée(s) depuis :\n{chemin}"]
        if rapport["doublons"]:
            texte.append(f"\nDoublons ignorés (lignes) : {liste_courte(rapport['doublons'])}")
        if rapport["manquants"]:
            texte.append("\nValeurs manquantes :\n  " + "\n  ".join(rapport["manquants"][:15])
                         + ("\n  …" if len(rapport["manquants"]) > 15 else ""))
        if rapport["photos"]:
            texte.append(f"\nPhotos manquantes, remplacées par « {PHOTO_VIDE} » (lignes) : "
                         f"{liste_courte(rapport['photos'])}")
        probleme = any(rapport[k] for k in ("doublons", "manquants", "photos"))
        (messagebox.showwarning if probleme else messagebox.showinfo)("Chargement", "\n".join(texte))

    def charger_fichier(self, chemin):
        if not os.path.isfile(chemin):
            messagebox.showerror("Fichier introuvable", f"Le fichier n'existe pas :\n{chemin}")
            return
        if self.modifie and not messagebox.askyesno(
                "Modifications non sauvegardées",
                "Des modifications n'ont pas été sauvegardées et seront perdues.\nCharger quand même ?"):
            return
        try:
            rapport = self.base.charger(chemin)
        except (OSError, csv.Error) as e:
            messagebox.showerror("Erreur de lecture", f"Impossible de lire le fichier :\n{e}")
            return
        self.modifie = False
        self.cache_images.clear()
        self.rafraichir_tout()
        self.statut(f"{len(self.base.rows)} plante(s) chargée(s) depuis {chemin}")
        self.afficher_rapport(rapport, chemin)

    def sauvegarder_fichier(self, chemin):
        if not chemin.lower().endswith(".csv"):
            chemin += ".csv"
        if not os.path.isdir(os.path.dirname(os.path.abspath(chemin))):
            messagebox.showerror("Dossier introuvable", f"Le dossier n'existe pas :\n{os.path.dirname(chemin)}")
            return
        if os.path.exists(chemin) and not messagebox.askyesno(
                "Fichier existant", f"Le fichier existe déjà :\n{chemin}\n\nVoulez-vous l'écraser ?"):
            return
        try:
            self.base.sauvegarder(chemin)
        except OSError as e:
            messagebox.showerror("Erreur d'écriture", f"Impossible d'écrire le fichier :\n{e}")
            return
        self.modifie = False
        self.rafraichir_tout()
        self.statut(f"Herbier sauvegardé dans {chemin}")
        messagebox.showinfo("Sauvegarde", f"{len(self.base.rows)} plante(s) sauvegardée(s) dans :\n{chemin}")

    # ---- modifications
    def apres_modification(self, message):
        if SAUVEGARDE_AUTO:
            try:
                self.base.sauvegarder()
                self.modifie = False
                message += " (fichier mis à jour)"
            except OSError as e:
                self.modifie = True
                messagebox.showerror("Erreur d'écriture", f"La modification n'a pas pu être écrite dans le fichier :\n{e}")
        else:
            self.modifie = True
            message += " (non sauvegardé)"
        self.rafraichir_tout()
        self.statut(message)

    def ouvrir_ajout(self):
        FenetreAjout(self)

    def demander_suppression(self, ligne):
        suite = "\nElle sera aussi retirée du fichier .csv." if SAUVEGARDE_AUTO else ""
        if messagebox.askyesno("Supprimer une plante",
                               f"Supprimer « {ligne[0]} » ({ligne[1]}) de l'herbier ?{suite}", icon="warning"):
            self.base.supprimer(ligne)
            self.apres_modification(f"Plante « {ligne[0]} » supprimée")

    def quitter(self):
        if self.modifie:
            rep = messagebox.askyesnocancel("Modifications non sauvegardées", "Sauvegarder avant de quitter ?")
            if rep is None:
                return
            if rep:
                try:
                    self.base.sauvegarder()
                except OSError as e:
                    messagebox.showerror("Erreur d'écriture", str(e))
                    return
        self.destroy()


if __name__ == "__main__":
    Application().mainloop()

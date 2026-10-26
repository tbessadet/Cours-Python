# 4- Visualisation

Outils de base : Histogramme pour voir la distribution
    On peut direct projeter les 50 histogrammes sur un graph croisé (scatter plot en croissant 2 barplot)

**Enjeu visualisation** : Trouver un mode de visualisation adapté à notre cervau

PCA : outil de base pour 

- réduction dimensionnalité et le bruit 

- visualisation de données grande dimension


LLE : ACP locale  
Idée : On crée un graph avec les proches voisins et on fais des ACP dessus
MSD : multi-dimensional scaling  
ISOMAP basé uniquement sur la distance, mais peu robuste au bruit

### T-SNE :
On maximise la vraissemblance et on y met une gaussienne dessus.
Si 2 points sont proches dans l'espace d'origine, ils doivent etre proches dans l'espace d'arrivé, sinon ils ne s'influencent pas sur la position dans l'espace d'arrivé.  
$=>$ Hyper-paramètre important : le sigma de la gaussienne !  
MAIS ici pas de prbl : Il existe des euristique pour regler le sigma par défaut qui marche super bien ! (possible de le regler : `perplexity`)
Attention la fonction n'est pas ..., cad que le graph d'arrivé n'est pas forcement le même, mais le resultat est reste identique.

### UMAP : combinaison PCA et T-SNE :
Permet de comblé la perte des données distances lointaines



#### Est-ce qu'on visualise au début du taff ou après ?
Permet de se lancer sur le prbl
Mais peut aussi biaiser notre vision du prbl

Mais dans tout les cas utile pour la detection des erreurs :
- Elles sont partout sur ttes les classes
- Elles sont dans un endroit donné : liés aux variables ? (besoin de + de features pour bien classé) ou demander au expert pk il y a tel ressemblance ? Ou faire un modèle qui regarde plus finement ce qu'il se passe a cet endroit la


### Librairy avec du callback :
Basé sur du matplotlib mais permet du callback 
Bokeh (un peu plus vieille)
pltoly
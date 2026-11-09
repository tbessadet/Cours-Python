## A expliquer :  
**Expliquer residual-layer**   

Pour trouver les boites qui contour des objets : 
Former un carré = 2 coordonées (x,y) + 2 dim (hauteur, largeur) => Quadruple Régression pour entourer un objet  
Puis classifier pour dire ce qu'il y a dans la boite  

**Pourquoi MLP fonctionne pas pour les images ?**  
Le MLP n'a pas le propriété invariance par translation, rotation ... qui permet la robustesse et l'adaptation au changement de contexte  

**Classement d'image avant le deep :**    
On crée des descipteur clef, on regarde quel descripteur tique ou et on compte, on fait du clustering et on detecte trouve des paquet "vocabulary"  
Puis on utilise un SVM et fait de la classification sur les descripteur  

**Classement après le Deep :**  
IoU (Intersection Over Union) : metrique pour savoir si on a bien detecter l'objet que l'on voulait detecter  

Box level annotation : contour de l'objet en boite carré  
Pixel level annotation : contour de l'objet au pixel prêt (zone colorié ...). Passe par convolution, maxpooling, upscaling ...  

**Interpretabilité réseau de neuronne :**  
Technique par Occlusion :  
- On prend notre image
- on la modifie en ocultant une partie avec un carée gris/noir ...
- On crée n image avec une occlusion qui se deplace
- On regarde si le modele classe bien ou non l'image en fonction de la position de l'oclusion  
$\to$ mapping des scores par pixel nous montre l'endroit de l'image important pour la classification  

**GradCam**  
On supprime la tête (réseau de neuronne ...)  
On le remplace par ... décision linéaire et on fait une saillance map  

### Graph Neural Network
**Passer convolution à logique sur graphs :**  
Convolution : "caractériser le pixel du centre p/r aux voisins"  
Graph : "caractériser mon noeud p/r aux voisins". Prbl : nbr de voisins changent tt le temps, poids ...  

**edge ($e_{ij}$):**  
Nature du lien entre les 2 noeuds, accésoire  

**nombres filtres ($n_i$):**  
$n_i$ dépend du nombre de voisins  

**Message Passing :**  
ψ : réseau de neuronnes (ex MLP)  
Possibilité de brancher des losses sur :
- noeuds (nodes)
- arcs (edges)
- graphes entier

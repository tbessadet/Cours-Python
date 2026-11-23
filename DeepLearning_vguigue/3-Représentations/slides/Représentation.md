## Representation  
### Architechtures :  

**Apprentissage de représentation :**  
On connais deja l'ACP, mais ne fonctionne que sur matrice carré. On a la `Singular Value Decomposition` (SVD) qui fait la "meme chose" sur matrice rectangle.  
Reconstruire une matrice X a partir de 2 matrices C et D : produit scalaire pondéré (activer gradient sur les matrices, faire le prod scalaire a la main puis MSE, backward ...)  
Exemple :  
- C : les films (representé en vecteur : durée, type ...)
- D : profil utilisateur
- X : Affinité de la personne pour les films => Possibilité de faire des recommandations
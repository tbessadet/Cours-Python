# Reseaux de Neurones Recurrents (RNN) :

Jusqu'a maintenant, les CNN prennent les series temporelles de taille fixe. Si jamais on veut traiter des series taille différentes :
- on supprime tout ce qui depasse taille_min
- on fait du "padding" (rajoute des imgaes) pour combler jusqua taille_max (peut poser des prbl, a peut pret gommer par maxpooling)

## RNN :
chaque étape (t de la serie temporelle T) depend de son observation $X_t$ mais aussi d'un output provenant des précedentes $h_{t-1}$  
Hyperparamètre clef est la taille de $W_1$ et $W_2$ : z (taille de l'espace latent). augmente le nombre de paramètre de manière quadratique
- $W_1$ : $(len(x_t), z)$
  - z = hidden_size
- $W_2$ : $(z, z)$
- $h_t = tanh( x_tW_1 + h_{t-1}W_2 + b)$


**Structure auto-régressive :** Le modele predit un terme a partir des précedents. Si un/des precedents est un terme predit par le modele, c'est auto regressif (un llm s'appui sur ses premier mot predit pour finir la phrase)   

## Specificitées & Difficultées

**Problème de prediction de plusieurs pas de temps dans le futur :** On parle de mémoire, pas "stable" dans le futur   

**Problème de vanishing / exploding gradient :** Plusieurs méthodes pour éviter :
- BatchNorm (entrainer des moyenens mobiles pour normaliser centree reduite), Gradient clipping (???), Residual architecture (x = f(x) + x)
- LSTM (Long short-term memory): 
  - Neuronnes apprennent ce qu'ils doivent garder et oublier
  - Porte d'oublie : ?
  - Mémory update : ?
  - Output : ?

## Les Bi-RNN :
On fait une lecture de gauche a droite, et de droite a gauche. On branche nos classifier sur les différentes sortie et on peut detecter un evenemment ... (plus fiable que regarder uniquement dans un sens)

## Architectures :

#%% Initialisation
import numpy as np
import pandas as pd

X_init = np.array(['A', 'C', 'C', 'G', 'A', 'C', 'T', 'T', 'A', 'G', 'A', 'C', 'A', 'G', 'G', 'T'], dtype=str)
Y_init = np.array(['T', 'T', 'A', 'C', 'C', 'G', 'A', 'C', 'G', 'T', 'A', 'T', 'A', 'C', 'A', 'G', 'C', 'G', 'T', 'A'],dtype=str)

X_test = np.copy(X_init)
Y_test = np.copy(Y_init)

#%% Fonction
def scoring(a,b):
    ## score de 'a' par rapport à 'b'
    if a == '-' or b == '-':
        return -4 # -4 si contient une deletion
    elif a == b:
        return 2  # +2 si identique
    else:
        return -2 # -2 si différent


def gap(A,i,n):
    # Fonction permettant d'inserrer n tiret à la pos i // Dans la pratique on n'en mais jamais 2 d'un coup
    M = np.copy(A)
    for _ in range(n):
        M = np.insert(M,i,'-')
    return M

def Sim(A,B,i,j):
    # Obtenir le score des de comparaison entra A[i:j] et B[i:j] // Au final inutile
    score = 0

    if len(A) < j:
        A = gap(A, len(A), j-len(A))
    if len(B) < j:
        B = gap(B, len(B), j-len(B))

    for k in range(i,j):
        score += scoring(A[k],B[k])

    return score

def max_score(M, A_bis, B_bis, i,j):
    # Recherche du score max entre les 3 possibilités (match, gap dans A, gap dans B)
    a = M.iloc[i-1,j-1] + scoring(A_bis[i],B_bis[j])
    b = M.iloc[i,j-1] + scoring('-',B_bis[j])
    c = M.iloc[i-1,j] + scoring(A_bis[i],'-')
    return a, b, c

def mat_score(A,B):
    # initialisation matrice de score
    n1 = len(A)
    n2 = len(B)

    A_bis = gap(A,0,1) # rajouter une ligne et une colonne pour les gaps initaux
    B_bis = gap(B,0,1)

    M = pd.DataFrame(np.zeros((n1+1,n2+1)), index = A_bis, columns = B_bis) # avec les entêtes

    # initialisation du score de départ
    M.iloc[0,0] = 0 

    # initialisation première ligne et première colonne
    for k in range(1,n1+1): 
        M.iloc[k,0] = scoring(A_bis[k],'-') + M.iloc[k-1,0]
    
    for k in range(1,n2+1):
        M.iloc[0,k] = scoring('-',B_bis[k]) + M.iloc[0,k-1]

    # remplissage de la matrice avec la règle de décision (max_score parmis les 3)
    for i in range(1,n1+1):
        for j in range(1,n2+1):
            M.iloc[i,j] = max(max_score(M, A_bis, B_bis, i, j))

    return M

def alignement(A, B, M=None):
    #Reconstruire l'alignement optimal en fonction de la matrice de score
    if M is None:
        M = mat_score(A, B)

    i = len(A)
    j = len(B)
    A_bis = gap(A,0,1)
    B_bis = gap(B,0,1)
    align_A = []
    align_B = []

    # On part de la dernière case de la matrice [i,j] et on remonte en [0,0] en suivant le chemin optimal
    # /!\ N'explore qu'un seul chemin, possibilité d'avoir des ex-aequo /!\
    while i > 0 or j > 0:
        a, _, c = max_score(M, A_bis, B_bis, i,j)

        # 2 carac bien alignés
        if (
            i > 0 and j > 0
            and M.iloc[i, j] == a
            ):
            align_A.append(A[i - 1])
            align_B.append(B[j - 1])
            i -= 1
            j -= 1

        # Gap dans B
        elif (
            i > 0
            and M.iloc[i, j] == c
            ):
            align_A.append(A[i - 1])
            align_B.append('-')
            i -= 1

        # Gap dans A
        else:
            align_A.append('-')
            align_B.append(B[j - 1])
            j -= 1

    return ''.join(reversed(align_A)), ''.join(reversed(align_B)) #remettre les listes dans le bon sens



#%% Run
M_score = mat_score(X_test, Y_test)
X_aligne, Y_aligne = alignement(X_test, Y_test, M_score)

print("Score optimal :", M_score.iloc[-1, -1])
print("Séquence optimales :")
print("X : " + X_aligne)
print("Y : " + Y_aligne)
# %%

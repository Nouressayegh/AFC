import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import copy as cp
import TP1_Analyse_en_composantes_principales as tp

D = pd.read_csv("C:/Users/dell/Documents/Ecole des Mines/Majeure/Analyse temporelle et fonctionelle/TP ACP/AFC.csv",sep =';' )

# Pour tester notre AFC on va se restreindre au sexe et à la fonction par exemple
# Codons maintenant la matrice de contingence, pour cela il nous faut les modalités

# Pour le sexe, on dira que 0 est non défini, 1 est une homme et 2 est une femme

D = pd.DataFrame([D["Fonction"],D[" temps travail"]])
D = D.transpose()
k = D.shape[0]
I = max(D["Fonction"])+1
J = max(D[" temps travail"])+1

mat_contingence = np.zeros((I,J))

for i in range(1,k): 
   mat_contingence[D.loc[i,"Fonction"] , D.loc[i," temps travail"]] = mat_contingence[D.loc[i,"Fonction"] ,D.loc[i," temps travail"]] + 1

mat_contingence = pd.DataFrame(mat_contingence)
mat_contingence = pd.DataFrame([[0,32,21,15,7,0,8,2],[0,10,7,11,5,2,3,0],[0,1,0,1,1,0,0,0],[0,19,7,44,8,5,6,6],[8,10,1,2,1,0,1,3]])
mat_contingence.columns = ["Non répondu","Administratif","Technicien (OS)","Ingénieur","Technicien supérieur", "Direction", "Contactuel S1", "Contractuel S2"]
mat_contingence = mat_contingence.transpose()
mat_contingence.columns = ["sous-chargé","correspondant","chargé adéquatement","sur-chargé","non répondu"]

# Profils lignes et profils colonnes

mat_sommes = cp.deepcopy(mat_contingence)

mat_sommes["Somme"] = [np.sum(mat_sommes.loc["Non répondu",:]),np.sum(mat_sommes.loc["Administratif",:]),np.sum(mat_sommes.loc["Technicien (OS)",:]),np.sum(mat_sommes.loc["Ingénieur",:]),np.sum(mat_sommes.loc["Technicien supérieur",:]),np.sum(mat_sommes.loc["Direction",:]),np.sum(mat_sommes.loc["Contactuel S1",:]),np.sum(mat_sommes.loc["Contractuel S2",:])]
T = [np.sum(mat_sommes.loc[:,"sous-chargé"]), np.sum(mat_sommes.loc[:,"correspondant"]),np.sum(mat_sommes.loc[:,"chargé adéquatement"]),np.sum(mat_sommes.loc[:,"sur-chargé"]),np.sum(mat_sommes.loc[:,"non répondu"])] + [np.sum(mat_sommes.loc[:,"Somme"])]
mat_sommes = mat_sommes.append(pd.Series(T, index=mat_sommes.columns), ignore_index=True)

# Matrice des fréquences

total = np.sum(np.sum(mat_contingence))
mat_frequence = mat_sommes/total

profile_ligne = cp.deepcopy(mat_frequence)
profile_colonne = cp.deepcopy(mat_frequence)

for i in range(8):
    profile_ligne.loc[i,:] = profile_ligne.loc[i,:]/profile_ligne.loc[i,"Somme"]

profile_colonne.columns = ["fonction {}".format(j) for j in range(mat_contingence.shape[1])] + ["Somme"]
profile_ligne.columns = ["fonction {}".format(j) for j in range(mat_contingence.shape[1])] + ["Somme"]

for j in range(mat_contingence.shape[1]):
    profile_colonne.loc[:,"fonction {}".format(j)] = profile_colonne.loc[:,"fonction {}".format(j)]/profile_colonne.loc[3][j]

#profile_colonne.columns = ["sous-chargé","correspondant","chargé adéquatement","sur-chargé","non répondu"] + ["Somme"]

# Matrices de tranformation
# Matrice diagonale des marges en lignes:

n = mat_contingence.shape[0]
p = mat_contingence.shape[1]

Dn = np.zeros((n,n))
Dp = np.zeros((p,p))

for i in range(n):
    Dn[i][i] = profile_colonne.loc[i,"Somme"]
    
for i in range(p):
    Dp[i][i] = profile_ligne.loc[3,"fonction {}".format(i)]

mat_frequence = mat_frequence.drop(3)
del mat_frequence["Somme"]
mat_frequence.columns = ["sous-chargé","correspondant","chargé adéquatement","sur-chargé","non répondu"]

# AFC dans l'espace Rp des fonctions (S = F'.Dn-1.F.Dp-1)

S = np.transpose(mat_frequence).dot(np.linalg.inv(Dn).dot(mat_frequence.dot(np.linalg.inv(Dp))))

# On voit bien que la matrice S est non symétique et réelle donc  non diagonalisable, il va falloir
# utiliser le complément de cours afin de la transformer en matrice diagonalisable

A_chapeau = np.transpose(mat_frequence).dot(np.linalg.inv(Dn).dot(mat_frequence))
A = np.sqrt(np.linalg.inv(Dp)).dot(A_chapeau.dot(np.sqrt(np.linalg.inv(Dp))))

valeur_propre, vecteur_propre = tp.classeur(A)
valeur_propre = np.real(np.array(valeur_propre))
vecteur_propre = np.array(vecteur_propre)

# On prend les deux premiers vecteurs propres comme axes factoriels

# Les nouvelles coordonées dans l'espace factoriel réduit 

nouvelles_coord1 = np.sqrt(Dp).dot(vecteur_propre[0])
nouvelle_coord2 = np.sqrt(Dp).dot(vecteur_propre[1])

axe1 = mat_frequence.dot(nouvelles_coord1)
axe2 = -mat_frequence.dot(nouvelle_coord2)

nouvelles_coord = pd.DataFrame([axe1, axe2])
nouvelles_coord = np.transpose(nouvelles_coord)
nouvelles_coord.columns = ["Axe 1", "Axe 2"]

nouvelles_coord = nouvelles_coord.transpose()
nouvelles_coord.columns = ["Non répondu","Administratif","Technicien (OS)","Ingénieur","Technicien supérieur", "Direction", "Contactuel S1", "Contractuel S2"]
nouvelles_coord = nouvelles_coord.transpose()

# Visualisation du nuage dans l'espace factoriel réduit

noms = ["Non répondu","Administratif","Technicien (OS)","Ingénieur","Technicien supérieur", "Direction", "Contactuel S1", "Contractuel S2"]
plt.scatter(nouvelles_coord.loc[:,"Axe 1"],nouvelles_coord.loc[:,"Axe 2"])
plt.grid()
plt.title("Représentation des fonctions dans le nouveau plan factoriel")
for i in range(3):
    plt.annotate(noms[i],(nouvelles_coord.loc[noms[i],"Axe 1"],nouvelles_coord.loc[noms[i],"Axe 2"]))
plt.show()

# Réduction de la dimension dans Rn (T = F.Dp-1.F'.Dn-1)

T = mat_frequence.dot(np.linalg.inv(Dp).dot(np.transpose(mat_frequence).dot(np.linalg.inv(Dn))))

# Comme dans l'AFC sur l'espace Rp cette matrice n'est pas symétrique

M_chapeau = mat_frequence.dot(np.linalg.inv(Dp).dot(np.transpose(mat_frequence)))
M = np.sqrt(np.linalg.inv(Dn)).dot(M_chapeau.dot(np.sqrt(np.linalg.inv(Dn))))

valeur_propre2, vecteur_propre2 = tp.classeur(M)
valeur_propre2 = np.real(np.array(valeur_propre2))
vecteur_propre2 = np.array(vecteur_propre2)

# Les nouvelles coordonées dans l'espace factoriel réduit 

nouvelles_coord3 = np.sqrt(Dn).dot(vecteur_propre2[0])
nouvelles_coord4 = np.sqrt(Dn).dot(vecteur_propre2[1])

axe3 = np.transpose(mat_frequence).dot(nouvelles_coord3)
axe4 = -np.transpose(mat_frequence).dot(nouvelles_coord4)

nouvelles_coord2 = pd.DataFrame([axe3, axe4])
nouvelles_coord2 = np.transpose(nouvelles_coord2)
nouvelles_coord2.columns = ["Axe 1", "Axe 2"]

# Visualisation du nuage dans l'espace factoriel réduit

noms2 =  ["sous-chargé","correspondant","chargé adéquatement","sur-chargé","non répondu"]
plt.scatter(nouvelles_coord2.loc[:,"Axe 1"],nouvelles_coord2.loc[:,"Axe 2"])
plt.grid()
plt.title("Représentation des fonctions dans le nouveau plan factoriel")
for i in range(5):
    plt.annotate(noms2[i],(nouvelles_coord2.loc[noms2[i],"Axe 1"],nouvelles_coord2.loc[noms2[i],"Axe 2"]))
plt.show()

# Plot simultané des nuages de points

plt.scatter(nouvelles_coord.loc[:,"Axe 1"],nouvelles_coord.loc[:,"Axe 2"], color = "black")
plt.scatter(nouvelles_coord2.loc[:,"Axe 1"],nouvelles_coord2.loc[:,"Axe 2"], color = "blue")
plt.grid()
#plt.ylim([-0.01,0.04])
plt.title("Représentation simultanée des genres et des fonctions")
for i in range(8):
    plt.annotate(noms[i],(nouvelles_coord.loc[noms[i],"Axe 1"],nouvelles_coord.loc[noms[i],"Axe 2"]),color = "black")

for i in range(5):
    plt.annotate(noms2[i],(nouvelles_coord2.loc[noms2[i],"Axe 1"],nouvelles_coord2.loc[noms2[i],"Axe 2"]),color = "blue")

plt.show()


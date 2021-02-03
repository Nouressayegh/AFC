'''
TP 3: Analyse factorielle des correspondences
15/10/2020
'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import copy as cp
import TP1_Analyse_en_composantes_principales as tp

D = pd.read_csv("C:/Users/dell/Documents/Ecole des Mines/Majeure/Analyse temporelle et fonctionelle/TP ACP/AFC.csv",sep =';' )

# Pour tester notre AFC on va se restreindre au sexe et à la fonction par exemple
# Codons maintenant la matrice de contingence, pour cela il nous faut les modalités

# Pour le sexe, on dira que 0 est non défini, 1 est une homme et 2 est une femme

T1 = [elm for elm in D.loc[:,"Sexe"]]
T2 = [elm for elm in D.loc[:,"Fonction"]]

occurence = np.array([T1,T2])
occurence = occurence.transpose()
mat_contingence = []

# Contrustion de la matrice de contingence:

for i in range(3):
    fonction = {}
    j = 0
    for elm in occurence[:,1]:
        if occurence[j][0] == i:
            if elm not in fonction.keys():
                fonction[elm] = 1
            else:
                fonction[elm] += 1
        j += 1
    
    for k in range(7):
        if k not in fonction.keys():
            fonction[k] = 0
    fonction_tri = {}
    fonctions = sorted(fonction)
    
    for key in fonctions:
        fonction_tri[key] = fonction[key]
    
    mat_contingence.append([fonction_tri[key] for key in fonction_tri.keys()])
   
n = len(mat_contingence[0])
        
mat_contingence = np.array([mat_contingence[0],mat_contingence[1],mat_contingence[2]])


mat_contingence = pd.DataFrame(mat_contingence)
mat_contingence.columns = ["fonction {}".format(i) for i in range(n)]
#mat_contingence.columns = ["Non répondu","Administratif","Technicien (OS)","Ingénieur","Technicien supérieur", "Direction", "Contactuel S1", "Contractuel S2"]
mat_sommes = cp.deepcopy(mat_contingence)

mat_sommes["Somme"] = [np.sum(mat_sommes.loc[0,:]),np.sum(mat_sommes.loc[1,:]),np.sum(mat_sommes.loc[2,:])]
T = [np.sum(mat_sommes.loc[:,"fonction {}".format(i)]) for i in range(n)] + [np.sum(mat_sommes.loc[:,"Somme"])]
mat_sommes = mat_sommes.append(pd.Series(T, index=mat_sommes.columns), ignore_index=True)

# Nuage des points dans l'espace "Sexe"

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('X Axes')
ax.set_ylabel('Y Axes')
ax.set_zlabel('Z Axes')

vecteur = np.transpose(mat_contingence)
ax.scatter(vecteur.loc[:,0],vecteur.loc[:,1],vecteur.loc[:,2])
plt.show()

# Corrélations

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_trisurf(vecteur.loc[:,0],vecteur.loc[:,1],vecteur.loc[:,2])
plt.show()

# Matrice des fréquences

total = np.sum(np.sum(mat_contingence))
mat_frequence = mat_sommes/total

# Matrice de distribution des correspondances si les deux variables qualitatives sont indépendantes

T0 = np.array([[mat_frequence.loc[i,"Somme"]*mat_frequence.loc[3,"fonction {}".format(j)] for j in range(n)] for i in range(3)])
T0 = pd.DataFrame(T0) * total
T0.columns = ["fonction {}".format(i) for i in range(n)]

# Ecart entre les données de la matrice de contingence et la matrice de contingence sous hypothèse d'indépendance

R = mat_contingence - T0

# On remarque que cette matrice est différente de la matrice nulle, par conséquent on dit qu'à priori, on 
# a dépendance entre les différentes variables (les fonctions), mais il faut vérifier ce propos avec la stat
# du khi 2

# Liaison entre deux variables qualitatives: test du khi 2

freq_ind = T0 / total
khi = 0
for i in range(3):
    for j in range(n):
        khi += ((mat_contingence.loc[i,"fonction {}".format(j)] - freq_ind.loc[i,"fonction {}".format(j)])**2)/freq_ind.loc[i,"fonction {}".format(j)]
        
khi *= total
print("La statistique du Khi 2 vaut: " + str(khi))

# L'intensité de liaion khi/n est très elevé on en conclut qu'on a dépendance.
# La statistique du Khi 2 est d'autant plus grande ce qui veut dire que nos fonctions sont très
# dépendantes.

# Profils lignes et profils colonnes

profile_ligne = cp.deepcopy(mat_frequence)
profile_colonne = cp.deepcopy(mat_frequence)

for i in range(3):
    profile_ligne.loc[i,:] = profile_ligne.loc[i,:]/profile_ligne.loc[i,"Somme"]

for j in range(n):
    profile_colonne.loc[:,"fonction {}".format(j)] = profile_colonne.loc[:,"fonction {}".format(j)]/profile_colonne.loc[3][j]

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
mat_frequence.columns = ["Non répondu","Administratif","Technicien (OS)","Ingénieur","Technicien supérieur", "Direction", "Contactuel S1", "Contractuel S2"]

# AFC dans l'espace Rp des fonctions (S = F'.Dn-1.F.Dp-1)

S = np.transpose(mat_frequence).dot(np.linalg.inv(Dn).dot(mat_frequence.dot(np.linalg.inv(Dp))))

# On voit bien que la matrice S est non symétique et réelle donc  non diagonalisable, il va falloir
# utiliser le complément de cours afin de la transformer en matrice diagonalisable

A_chapeau = np.transpose(mat_frequence).dot(np.linalg.inv(Dn).dot(mat_frequence))
A = np.sqrt(np.linalg.inv(Dp)).dot(A_chapeau.dot(np.sqrt(np.linalg.inv(Dp))))

valeur_propre, vecteur_propre = tp.classeur(A)
valeur_propre = np.real(np.array(valeur_propre))
vecteur_propre = np.array(vecteur_propre)

# Cascade des valeurs propres et inerties cumulées

# Cascade
plt.plot(np.arange(np.size(valeur_propre)),valeur_propre, color = "green")
plt.scatter(np.arange(np.size(valeur_propre)),valeur_propre,color = "magenta")
plt.grid()
plt.title("Cascade des valeurs propres")
plt.show()

# Inerties

inertie_cumul = []

for i in range(np.size(valeur_propre)):
    inertie_cumul.append(np.sum(valeur_propre[0:i]))

plt.hist(valeur_propre)
plt.hist(inertie_cumul)
plt.legend(("inerties","inerties cumulées"))
plt.title("Inerties et Inerties cumulées")
plt.show()

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
nouvelles_coord.columns = ["Non répondu","Homme","Femme"]
nouvelles_coord = nouvelles_coord.transpose()

# Visualisation du nuage dans l'espace factoriel réduit

noms = ["Non répondu","Homme","Femme"]
plt.scatter(nouvelles_coord.loc[:,"Axe 1"],nouvelles_coord.loc[:,"Axe 2"])
plt.grid()
plt.title("Représentation des fonctions dans le nouveau plan factoriel")
for i in range(3):
    plt.annotate(noms[i],(nouvelles_coord.loc[noms[i],"Axe 1"],nouvelles_coord.loc[noms[i],"Axe 2"]))
plt.show()


# Tableau des inerties et pourcentage des variances

tableau = np.transpose(np.array([valeur_propre,valeur_propre * 100 /np.sum(valeur_propre)]))
tableau = pd.DataFrame(tableau)
tableau.columns = ["Inerties","Pourcentages %"]
tableau = tableau.transpose()
tableau.columns = ["Axe 1", "Axe 2", "Axe 3","Axe 4","Axe 5","Axe 6","Axe 7","Axe 8"]
tableau = tableau.transpose()

# Qualité de projection

def quali_representation(val,vec,coord,i,nom):
    norme_xi = np.linalg.norm(coord.loc[nom[i],:])

    m = coord.shape[1]
    s = 0
    
    for j in range(m):
        axe_j = vec[j]
        axe_j = axe_j[:m]
        proj_xi_j = coord.loc[nom[i],:].dot(np.transpose(axe_j))
        s += proj_xi_j **2
        
    return s/(norme_xi**2)

qualite = [quali_representation(valeur_propre,vecteur_propre,nouvelles_coord,i,noms) for i in range(np.shape(nouvelles_coord)[0]) ]

# Réduction de la dimension dans Rn (T = F.Dp-1.F'.Dn-1)

T = mat_frequence.dot(np.linalg.inv(Dp).dot(np.transpose(mat_frequence).dot(np.linalg.inv(Dn))))

# Comme dans l'AFC sur l'espace Rp cette matrice n'est pas symétrique

M_chapeau = mat_frequence.dot(np.linalg.inv(Dp).dot(np.transpose(mat_frequence)))
M = np.sqrt(np.linalg.inv(Dn)).dot(M_chapeau.dot(np.sqrt(np.linalg.inv(Dn))))

valeur_propre2, vecteur_propre2 = tp.classeur(M)
valeur_propre2 = np.real(np.array(valeur_propre2))
vecteur_propre2 = np.array(vecteur_propre2)

# Cascade des valeurs propres et inerties cumulées

# Cascade
plt.plot(np.arange(np.size(valeur_propre2)),valeur_propre2, color = "blue")
plt.scatter(np.arange(np.size(valeur_propre2)),valeur_propre2,color = "orange")
plt.grid()
plt.title("Cascade des valeurs propres")
plt.show()

# Inerties

inertie_cumul2 = []

for i in range(np.size(valeur_propre2)):
    inertie_cumul2.append(np.sum(valeur_propre2[0:i]))

plt.hist(valeur_propre2)
plt.hist(inertie_cumul2)
plt.legend(("inerties","inerties cumulées"))
plt.title("Inerties et Inerties cumulées")
plt.show()

# On prend les deux premiers vecteurs propres comme axes factoriels, les autres 
# ont des valeurs propres correspondantes nulles si on arrondit à 3 chiffre après la virgule

# Les nouvelles coordonées dans l'espace factoriel réduit 

nouvelles_coord3 = np.sqrt(Dn).dot(vecteur_propre2[0])
nouvelles_coord4 = np.sqrt(Dn).dot(vecteur_propre2[1])

axe3 = np.transpose(mat_frequence).dot(nouvelles_coord3)
axe4 = -np.transpose(mat_frequence).dot(nouvelles_coord4)

nouvelles_coord2 = pd.DataFrame([axe3, axe4])
nouvelles_coord2 = np.transpose(nouvelles_coord2)
nouvelles_coord2.columns = ["Axe 1", "Axe 2"]

# Visualisation du nuage dans l'espace factoriel réduit

noms2 = ["Non répondu","Administratif","Technicien (OS)","Ingénieur","Technicien supérieur", "Direction", "Contactuel S1", "Contractuel S2"]
plt.scatter(nouvelles_coord2.loc[:,"Axe 1"],nouvelles_coord2.loc[:,"Axe 2"])
plt.grid()
plt.title("Représentation des fonctions dans le nouveau plan factoriel")
for i in range(8):
    plt.annotate(noms2[i],(nouvelles_coord2.loc[noms2[i],"Axe 1"],nouvelles_coord2.loc[noms2[i],"Axe 2"]))
plt.show()

# Tableau des inerties et pourcentage des variances

tableau2 = np.transpose(np.array([valeur_propre2,valeur_propre2 * 100 /np.sum(valeur_propre2)]))
tableau2 = pd.DataFrame(tableau2)
tableau2.columns = ["Inerties","Pourcentages %"]
tableau2 = tableau2.transpose()
tableau2.columns = ["Axe 1", "Axe 2", "Axe 3"]
tableau2 = tableau2.transpose()

# Qualité de projection

qualite2 = [quali_representation(valeur_propre2,vecteur_propre2,nouvelles_coord2,i,noms2) for i in range(np.shape(nouvelles_coord2)[0]) ]

# La qualité de projection est bonne en moyenne

# Plot simultané des nuages de points

plt.scatter(nouvelles_coord.loc[:,"Axe 1"],nouvelles_coord.loc[:,"Axe 2"], color = "black")
plt.scatter(nouvelles_coord2.loc[:,"Axe 1"],nouvelles_coord2.loc[:,"Axe 2"], color = "blue")
plt.grid()
plt.title("Représentation simultanée des genres et des fonctions")
for i in range(3):
    plt.annotate(noms[i],(nouvelles_coord.loc[noms[i],"Axe 1"],nouvelles_coord.loc[noms[i],"Axe 2"]),color = "black")

for i in range(8):
    plt.annotate(noms2[i],(nouvelles_coord2.loc[noms2[i],"Axe 1"],nouvelles_coord2.loc[noms2[i],"Axe 2"]),color = "blue")

plt.show()


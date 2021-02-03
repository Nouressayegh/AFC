'''TP AFC : Essayegh Nour, Amini Nada'''

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

import TP_ACP as tp1
#On importe la dataframe
X=pd.read_excel('TP_AFC_majeur1718_travail.xlsx', index_col=0)
print("La data frame est: \n")
print(X)

#Pour tester l'AFC qu'on va coder on va se restreindre aux variables "Sexe" et "Fonction"
X=X[["Sexe","Fonction"]]
#creation du tableau de contingence
(k,m)=X.shape

#on prend le nombre de modalité des deux variables
X1=X["Sexe"]
X2=X["Fonction"]
maxRow=X1.max()+1
maxCol=X2.max()+1


nomRow=['nonrepondu','H','F']
nomCol=['non_repondu' ,'Administratif',' Technicien (OS)' ,'ingénieur',' technicien supérieur','direction',' contractuel_S1','contractuel_S2']

V=np.zeros((maxRow,maxCol))

#il faut remplir le tableau de contingence

for i in range (1,k):
    V[X1[i],X2[i]]=V[X1[i],X2[i]]+1
    
Tab=pd.DataFrame(V,index=nomRow,columns=nomCol)

#on tranforme le tableau de contigence en tableau des frequences

Vf=V/k
Tabf=pd.DataFrame(Vf,index=nomRow,columns=nomCol)
#On crée les matrices Dn et Dp
Dn=np.zeros((maxRow,maxRow))
Dp=np.zeros((maxCol,maxCol))

#remplissage des matrices diagonales par les loi marginale.
for i in range(0,maxRow):
    d=0
    for j in range(0,maxCol):
        d+=Vf[i,j]
    Dn[i,i]=d
    
for j in range(0,maxCol):
    d=0
    for i in range(0,maxRow):
        d+=Vf[i,j]
    Dp[j,j]=d
    
ax = plt.axes(projection='3d')
ax.scatter3D(V[0,:], V[1,:], V[2,:], 'Greens')
plt.show()
    
#réation des profils lignes et colonnes
prof_lig=np.dot( np.linalg.inv(Dn),Vf)
prof_col=np.dot( np.linalg.inv(Dp),Vf.transpose())

#matrice à diagonalisé
#d'après le complèment de cours: pour assurer la 
'''
    analyse des proximités entre individus par rapport àl’origine : 
    on recherche l’axe d’inertie max du nuages profils lignes passant par origine O et
engendré par un vecteur unitaire u pour une métrique
'''
#dans Rp
#On commence par 
T=np.dot(Vf,prof_col,np.linalg.inv(Dn))

#transformer en une matrice diagonalisable.

inter1=np.dot(Vf,prof_col)
A1=np.dot(np.sqrt(np.linalg.inv(Dn)),inter1,np.sqrt(np.linalg.inv(Dn)))
valpT,v=tp1.classeur(A1)

#inertie
Inertie=valpT
InertieCum=[]
for i in range(0,len(valpT)):
    InertieCum.append(sum(valpT[0:i]))

    
plt.bar(np.arange(len(valpT)),valpT)
plt.title("Cascade de vecteurs propres")

#on prend les deux premières valeurs propres
v1=v[0,:].transpose()
v2=v[1,:].transpose()

#les deux directions des axes factoriels
v1=np.dot(np.sqrt(Dn),v1)
v2=np.dot(np.sqrt(Dn),v2)

#les nouvelles coordonées
C1=np.dot(Vf.transpose(),v1)
C2=np.dot(Vf.transpose(),v2)

#graphe annoté but ça marche pas
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
    
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    
for i in range(m):
    plt.annotate(Tabf.columns[i], (C1,C2))
plt.show()

'''
plt.plot(C1,C2,'o')
#khas les noms dial les colonnes p7al le cercles dial les cov
plt.show()'''

#dans Rp
S=np.dot(Vf.transpose(),prof_lig,np.linalg.inv(Dp))

#on transforme en matrice diagonalisable
inter2=np.dot(Vf.transpose(),prof_lig)
A2=np.dot(np.sqrt(np.linalg.inv(Dp)),inter2,np.sqrt(np.linalg.inv(Dp)))
valpS,u=tp1.classeur(S)

#inertie
Inertie=valpS
InertieCum=[]
for i in range(0,len(valpS)):
    InertieCum.append(sum(valpS[0:i]))

plt.bar(np.arange(len(valpS)),valpS)
plt.title("Cascade de vecteurs propres")
plt.show()
#On prend les deux premières valeurs propres
u1=u[0,:].transpose()
u2=u[1,:].transpose()

u1=np.dot(np.sqrt(Dp),u1)
u2=np.dot(np.sqrt(Dp),u2)

#les nouvelles coordonées
Z1=np.dot(Vf,u1)
Z2=np.dot(Vf,u2)

plt.plot(Z1,Z2,'o')
plt.show()

#il faut representer sur le meme graph avec les noms pour voir les tendances

#les test du chi2

test=chi2_contingency(V)
#Chi-square test examines whether rows and columns of a contingency table are statistically significantly associated






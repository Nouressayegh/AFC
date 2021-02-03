
install.packages(c("FactoMineR","factoextra"))

library("FactoMineR")
library("factoextra")

don <- read.table("data.txt", header = TRUE)
colnames(don) <- c("sexe","fonction","temps de tramat_contingenceail","qualité de mat_contingenceie")
don <- don[,1:2]
k <− length(don[,1])
I <− max(don[,1])+1
J <− max(don[,2])+1

mat_contingence <− matrix( 0 , nrow=I , ncol=J )
colnames (mat_contingence) <− c ("Fonction non repondue " ,"Administratif" , "Technicien (OS)" , "Ingénieur" , "Technicien supérieur", "Direction" , "Contractuel S1 " , "Contractuel S2 ")
rownames (mat_contingence) <− c (" Sexe non repondue " ,"Homme" , "Femme" )
for( i in 1:k) {
   mat_contingence[don[i,1]+1 ,don[i,2] +1] <− mat_contingence[don[i,1]+1 ,don[i,2]+1]+1
}
  
res.ca <- CA(mat_contingence)

eig.val <- get_eigenvalue(res.ca)


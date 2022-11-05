library(dplyr)
Drug_1520 = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/Drug_1520.csv",header= T)
Protein_1771 = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/Protein_1771.csv",header= T)

colnames(Drug_1520) = "Drug"
colnames(Protein_1771) = "Protein"
# KEGG
KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/case studies/KEGG/DTI_3920.csv",header= T)
CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/case studies/CHEMBL/DTI_6982.csv",header= T)

# get need DTIs (1520drug - 1771protein)
colnames(KEGG_DTI)[1] = "Drug"
colnames(KEGG_DTI)[2] = "Protein"
KEGG_DTI1 = semi_join(KEGG_DTI,Drug_1520,by="Drug")
KEGG_DTI2 = semi_join(KEGG_DTI1,Protein_1771,by="Protein")
KEGG_DTI3 = KEGG_DTI2[order(KEGG_DTI2$Drug),]

colnames(CHEMBL_DTI)[1] = "Drug"
colnames(CHEMBL_DTI)[2] = "Protein"
CHEMBL_DTI1 = semi_join(CHEMBL_DTI,Drug_1520,by="Drug")
CHEMBL_DTI2 = semi_join(CHEMBL_DTI1,Protein_1771,by="Protein")
CHEMBL_DTI3 = CHEMBL_DTI2[order(CHEMBL_DTI2$Drug),]

know_DTI = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/DTI_8207.csv",header= T)
colnames(know_DTI)[1] = "Drug"
colnames(know_DTI)[2] = "Protein"

extra_KEGG_DTI = anti_join(KEGG_DTI3,know_DTI,by=c("Drug","Protein"))
extra_CHEMBL_DTI = anti_join(CHEMBL_DTI3,know_DTI,by=c("Drug","Protein"))

# analysis shortest path length
lh = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/Dr_D_P_shortest_length.csv",header= T)
rownames(lh) = unlist(Drug_1520)
colnames(lh) = unlist(Protein_1771)

# view shortest path lengths of extra DTIs
extra_KEGG_DTI_scores = matrix(nr=length(extra_KEGG_DTI[,1]),nc=3)
for(i in 1:length(extra_KEGG_DTI[,1]))
{
  this_drug = extra_KEGG_DTI[i,1]
  this_protein = extra_KEGG_DTI[i,2]
  this_length = lh[this_drug, this_protein]
  extra_KEGG_DTI_scores[i,1] = this_drug
  extra_KEGG_DTI_scores[i,2] = this_protein
  extra_KEGG_DTI_scores[i,3] = this_length
}
m = as.numeric(extra_KEGG_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)

extra_CHEMBL_DTI_scores = matrix(nr=length(extra_CHEMBL_DTI[,1]),nc=3)
for(i in 1:length(extra_CHEMBL_DTI[,1]))
{
  this_drug = extra_CHEMBL_DTI[i,1]
  this_protein = extra_CHEMBL_DTI[i,2]
  this_length = lh[this_drug, this_protein]
  extra_CHEMBL_DTI_scores[i,1] = this_drug
  extra_CHEMBL_DTI_scores[i,2] = this_protein
  extra_CHEMBL_DTI_scores[i,3] = this_length
}
m = as.numeric(extra_CHEMBL_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)





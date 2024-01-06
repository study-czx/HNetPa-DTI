
library(dplyr)
Drug_1520 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/Drug_1520.csv",header= T)
Protein_1771 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/Protein_1771.csv",header= T)

colnames(Drug_1520) = "Drug"
colnames(Protein_1771) = "Protein"
# KEGG
KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/KEGG/DTI_2823.csv",header= T)
CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/CHEMBL/DTI_3475.csv",header= T)
Drugbank_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/DrugBank/DTI_8467.csv",header= T)

# get need DTIs (1520drug - 1771protein)
know_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI_8207.csv",header= T)
colnames(know_DTI)[1] = "Drug"
colnames(know_DTI)[2] = "Protein"

extra_KEGG_DTI = anti_join(KEGG_DTI,know_DTI,by=c("Drug","Protein"))
extra_CHEMBL_DTI = anti_join(CHEMBL_DTI,know_DTI,by=c("Drug","Protein"))
extra_Drugbank_DTI = anti_join(Drugbank_DTI,know_DTI,by=c("Drug","Protein"))

# analysis shortest path length
lh = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/Dr_D_P_shortest_length.csv",header= T)
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

extra_Drugbank_DTI_scores = matrix(nr=length(extra_Drugbank_DTI[,1]),nc=3)
for(i in 1:length(extra_Drugbank_DTI[,1]))
{
  this_drug = extra_Drugbank_DTI[i,1]
  this_protein = extra_Drugbank_DTI[i,2]
  this_length = lh[this_drug, this_protein]
  extra_Drugbank_DTI_scores[i,1] = this_drug
  extra_Drugbank_DTI_scores[i,2] = this_protein
  extra_Drugbank_DTI_scores[i,3] = this_length
}
m = as.numeric(extra_Drugbank_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)



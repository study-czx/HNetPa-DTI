Drug_1408 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv",header= T)
Protein_1648 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv",header= T)
colnames(Drug_1408) = "Drug"
colnames(Protein_1648) = "Protein"

KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/KEGG/DTI_3920.csv",header= T)
CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/CHEMBL/DTI_6982.csv",header= T)

# select need DTIs(1409 drugs-1648 proteins)
colnames(KEGG_DTI)[1] = "Drug"
colnames(KEGG_DTI)[2] = "Protein"
KEGG_DTI1 = semi_join(KEGG_DTI,Drug_1408,by="Drug")
KEGG_DTI2 = semi_join(KEGG_DTI1,Protein_1648,by="Protein")
KEGG_DTI3 = KEGG_DTI2[order(KEGG_DTI2$Drug),]

colnames(CHEMBL_DTI)[1] = "Drug"
colnames(CHEMBL_DTI)[2] = "Protein"
CHEMBL_DTI1 = semi_join(CHEMBL_DTI,Drug_1408,by="Drug")
CHEMBL_DTI2 = semi_join(CHEMBL_DTI1,Protein_1648,by="Protein")
CHEMBL_DTI3 = CHEMBL_DTI2[order(CHEMBL_DTI2$Drug),]

KEGG_DTIs = KEGG_DTI3
CHEMBL_DTIs = CHEMBL_DTI3

# remove DTIs in DrugBank
know_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/DTI_8020.csv",header= T)
colnames(know_DTI)[1] = "Drug"
colnames(know_DTI)[2] = "Protein"
extra_KEGG_DTI = anti_join(KEGG_DTIs,know_DTI,by=c("Drug","Protein"))
extra_CHEMBL_DTI = anti_join(CHEMBL_DTIs,know_DTI,by=c("Drug","Protein"))

# Predict scores
Predict_scores = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Predict_scores.csv",header= T)
colnames(Predict_scores)[1] = "Drug"
colnames(Predict_scores)[2] = "Protein"
extra_KEGG_DTI_scores = semi_join(Predict_scores,extra_KEGG_DTI,by=c("Drug","Protein"))
extra_CHEMBL_DTI_scores = semi_join(Predict_scores,extra_CHEMBL_DTI,by=c("Drug","Protein"))

# view distribution of scores
m = as.numeric(extra_KEGG_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)

m = as.numeric(extra_CHEMBL_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)
  
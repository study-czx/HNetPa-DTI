Drug_1408 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv",header= T)
Protein_1648 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv",header= T)
colnames(Drug_1408) = "Drug"
colnames(Protein_1648) = "Protein"

KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/KEGG/DTI_2823.csv",header= T)
CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI//case studies/CHEMBL/DTI_3475.csv",header= T)
DrugBank_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI//case studies/DrugBank/DTI_8467.csv",header= T)

# Predict scores
Predict_scores = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Predict_scores.csv",header= T)
colnames(Predict_scores)[1] = "Drug"
colnames(Predict_scores)[2] = "Protein"

extra_KEGG_DTI_scores = semi_join(Predict_scores,KEGG_DTI,by=c("Drug","Protein"))
extra_CHEMBL_DTI_scores = semi_join(Predict_scores,CHEMBL_DTI,by=c("Drug","Protein"))
extra_Drugbank_DTI_scores = semi_join(Predict_scores,DrugBank_DTI,by=c("Drug","Protein"))

# view distribution of scores
m = as.numeric(extra_KEGG_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)

m = as.numeric(extra_CHEMBL_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)

m = as.numeric(extra_Drugbank_DTI_scores[,3])
hist(m,breaks = 10,labels = TRUE)
mean(m)

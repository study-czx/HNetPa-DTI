KEGG_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/KEGG/DTI_2823.csv",header= T)
CHEMBL_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/CHEMBL/DTI_3475.csv",header= T)
DrugBank_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/DrugBank/DTI_8467.csv",header= T)

top10_drug = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/drug_top10.csv",header= T)
colnames(top10_drug) = c('Drug', 'Protein', 'Score')
KEGG_data = semi_join(KEGG_DTI,top10_drug,by=c('Drug', 'Protein'))
ChEMBL_data = semi_join(CHEMBL_DTI,top10_drug,by=c('Drug', 'Protein'))
DrugBank_data = semi_join(DrugBank_DTI,top10_drug,by=c('Drug', 'Protein'))

x = unique(rbind(KEGG_data,ChEMBL_data))
y = unique(rbind(x,DrugBank_data))


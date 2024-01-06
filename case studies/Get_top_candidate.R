library(dplyr)

Predict_scores = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Predict_scores.csv",header= T)

Drugs = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv",header= T)
Proteins = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv",header= T)

drug_top_10_candi = matrix(nc=3,nr=0)
drug_top_10_candi = data.frame(drug_top_10_candi)
colnames(drug_top_10_candi)[1]="drugbank_id"
colnames(drug_top_10_candi)[2]="uniprot_id"
colnames(drug_top_10_candi)[3]="scores"

for(i in 1:length(Drugs[,1]))
{
  this_drug = data.frame(drugbank_id=Drugs[i,1])
  this_scores = semi_join(Predict_scores,this_drug,by="drugbank_id")
  rank_scores = this_scores[order(this_scores$scores,decreasing = T),]
  top_scores = rank_scores[c(1:10),]
  drug_top_10_candi = rbind(drug_top_10_candi,top_scores)
}

write.csv(drug_top_10_candi,"D:/Users/czx/PycharmProjects/HNetPa-DTI/case studies/drug_top10.csv",row.names = FALSE)

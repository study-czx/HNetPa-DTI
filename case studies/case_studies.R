All_scores = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/All_scores_10fold.csv",header= T)

Drug_id = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv",header= T)
Protein_id = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv",header= T)
know_DTI = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/DTI-benchmark_set/DTI_8020.csv",header= T)
negative_samples = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/DrugBank dataset/negative samples/neg_DTI-net_8020.csv",header= T)

rownames(All_scores) = unlist(Drug_id)
colnames(All_scores) = unlist(Protein_id)

P_scores = list()
N_scores = list()
for(i in 1:length(know_DTI[,1]))
{
  this_drug = know_DTI[i,1]
  this_protein = know_DTI[i,2]
  P_scores = cbind(P_scores,All_scores[this_drug,this_protein])
}

for(i in 1:length(negative_samples[,1]))
{
  this_drug = negative_samples[i,1]
  this_protein = negative_samples[i,2]
  N_scores = cbind(N_scores,All_scores[this_drug,this_protein])
}

# view average scores of positive samples and negative samples
P_mean = mean(unlist(P_scores))
N_mean = mean(unlist(N_scores))

# get other scores
k=1
all_scores_list = matrix(nr=4000000,nc=3)
for (i in 1:length(Drug_id[,1])) {
  for (j in 1:length(Protein_id[,1])) {
    all_scores_list[k,1] = Drug_id[i,1]
    all_scores_list[k,2] = Protein_id[j,1]
    all_scores_list[k,3] = All_scores[Drug_id[i,1],Protein_id[j,1]]
    k=k+1
  }
}
all_scores_list = na.omit(all_scores_list)
all_scores_list = as.data.frame(all_scores_list)
all_scores_list[,3] = as.numeric(all_scores_list[,3])

colnames(all_scores_list)[1]='drugbank_id'
colnames(all_scores_list)[2]='uniprot_id'
colnames(all_scores_list)[3]='scores'
colnames(know_DTI)[1]='drugbank_id'
colnames(know_DTI)[2]='uniprot_id'
colnames(negative_samples)[1]='drugbank_id'
colnames(negative_samples)[2]='uniprot_id'

library(dplyr)
predict_scores = anti_join(all_scores_list,know_DTI,by=c("drugbank_id","uniprot_id"))
predict_scoress = anti_join(predict_scores,negative_samples,by=c("drugbank_id","uniprot_id"))

# view scores
m = as.numeric(predict_scoress [,3]) 
hist(m,breaks = 10,labels = TRUE)
my_predict_score = predict_scoress[order(predict_scoress$scores,decreasing = TRUE),]
write.csv(my_predict_score,"D:/Users/czx/PycharmProjects/HNGO-DTI/Predict_scores.csv",,row.names = F)


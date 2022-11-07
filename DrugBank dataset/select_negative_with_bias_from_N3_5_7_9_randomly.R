# select negative samples with hidden bias1
library(dplyr)
my_neg3 = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N3.csv",header=T)
my_neg5 = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N5.csv",header=T)
my_neg7 = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N7.csv",header=T)
my_neg9 = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N9.csv",header=T)

n_P = 8020
all_drug_id = sort(unique(my_neg3[,1]))
all_protein_id = sort(unique(my_neg3[,2]))

select_drug_id = sort(all_drug_id[sample(1:length(all_drug_id),1200)])
select_protein_id = sort(all_protein_id[sample(1:length(all_protein_id),1200)])
dr_id = data.frame(drugbank_id = select_drug_id)
p_id = data.frame(uniprot_id = select_protein_id)

negs = my_neg3
# negs = my_neg5, negs = my_neg7, negs = my_neg9
colnames(negs)[1] = 'drugbank_id'
colnames(negs)[2] = 'uniprot_id'

neg = semi_join(negs,dr_id,by="drugbank_id")
neg = semi_join(neg,p_id,by="uniprot_id")


n_1 = matrix(nr=length(dr_id[,1])*2,nc=2)
k = 1
for(i in 1:length(dr_id[,1]))
{
  id = data.frame(drugbank_id=dr_id[i,1])
  negative_drug = semi_join(neg,id,by="drugbank_id")
  n_negative_sample = length(negative_drug[,1])
  if(n_negative_sample>1)
  {
    random = sample(1:n_negative_sample,2)
    for(j in 1:length(random))
    {
      ids = random[j]
      n_1[k,1] = negative_drug[ids,1]
      n_1[k,2] = negative_drug[ids,2]
      k = k+1
    }
  }
  else
  {
    n_1[k,1] = negative_drug[1,1]
    n_1[k,2] = negative_drug[1,2]
    k = k+1
  }
}
n_1 = na.omit(n_1)

n_2 = matrix(nr=length(p_id[,1])*2,nc=2)
k = 1
for(i in 1:length(p_id[,1]))
{
  id = data.frame(uniprot_id = p_id[i,1])
  negative_protein = semi_join(neg,id,by="uniprot_id")
  n_negative_sample = length(negative_protein[,1])
  if(n_negative_sample>1)
  {
    random = sample(1:n_negative_sample,2)
    for(j in 1:length(random))
    {
      ids = random[j]
      n_2[k,1] = negative_protein[ids,1]
      n_2[k,2] = negative_protein[ids,2]
      k = k+1
    }
  }
  else
  {
    n_2[k,1] = negative_protein[1,1]
    n_2[k,2] = negative_protein[1,2]
    k = k+1
  }
}
n_2 = na.omit(n_2)

n_1_2 = rbind(n_1,n_2)
n_1_2_new = data.frame(unique(n_1_2))

colnames(n_1_2_new)[1] = 'drugbank_id'
colnames(n_1_2_new)[2] = 'uniprot_id'
num_remain = n_P-length(n_1_2_new[,1])

remain_n_2_3 = anti_join(neg,n_1_2_new,by=c("drugbank_id","uniprot_id"))
rand = sample(1:length(remain_n_2_3[,1]),num_remain)
k = 1
n_3 = matrix(nr=num_remain,nc=2)
for(i in 1:num_remain)
{
  m = rand[i]
  n_3[k,1] = remain_n_2_3[m,1]
  n_3[k,2] = remain_n_2_3[m,2]
  k = k+1
}
n_3 = data.frame(drugbank_id =n_3[,1],uniprot_id = n_3[,2])
n_all = rbind(n_1_2_new,n_3)
n_all = unique(n_all)

output_neg3 = n_all
# output_neg5 = n_all, output_neg7 = n_all, output_neg9 = n_all

write.csv(output_neg3,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/neg-bias3_5_7_9/neg3_8020.csv",row.names =FALSE,quote = F)
write.csv(output_neg5,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/neg-bias3_5_7_9/neg5_8020.csv",row.names =FALSE,quote = F)
write.csv(output_neg7,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/neg-bias3_5_7_9/neg7_8020.csv",row.names =FALSE,quote = F)
write.csv(output_neg9,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/neg-bias3_5_7_9/neg9_8020.csv",row.names =FALSE,quote = F)

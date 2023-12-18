library(dplyr)

dr_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/Drug_1409.csv",header=T)
p_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/Protein_1648.csv",header=T)
postive = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/DTI_8020.csv",header=T)

colnames(postive)[1] = 'drugbank_id'
colnames(postive)[2] = 'uniprot_id'


all_neg = matrix(nr=3000000,nc=2)
k = 1
for(i in 1:length(dr_id[,1]))
{
  for(j in 1:length(p_id[,1]))
  {
    all_neg[k,1] = dr_id[i,1]
    all_neg[k,2] = p_id[j,1]
    k = k+1
  }
}
all_neg = na.omit(all_neg)
all_neg = data.frame(drugbank_id=all_neg[,1],uniprot_id=all_neg[,2])
my_all_neg = anti_join(all_neg,postive)
n_P = length(postive[,1])

# select negative samples random, to avoid hidden bias1, first select for each drug and protein
neg = my_all_neg

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
n_1_2_new = unique(n_1_2)
num_remain = n_P-length(n_1_2_new[,1])

n_1_2_new = data.frame(drugbank_id = n_1_2_new[,1],uniprot_id = n_1_2_new[,2])
remain_n_2_3 = anti_join(neg,n_1_2_new,by=c("drugbank_id","uniprot_id"))

# random select to 8020
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

x = unique(n_all[,1])
y = unique(n_all[,2])
write.csv(n_all,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/neg_DTI-rand_8020.csv",row.names =FALSE,quote = F)



#########################################################################################################################################################
# for DTI-extra set
dr_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/Drug_111.csv",header=T)
p_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/Protein_123.csv",header=T)
postive = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/DTI_187.csv",header= T)

colnames(postive)[1] = 'drugbank_id'
colnames(postive)[2] = 'uniprot_id'

all_neg = matrix(nr=30000,nc=2)
k = 1
for(i in 1:length(dr_id[,1]))
{
  for(j in 1:length(p_id[,1]))
  {
    all_neg[k,1] = dr_id[i,1]
    all_neg[k,2] = p_id[j,1]
    k = k+1
  }
}
all_neg = na.omit(all_neg)
all_neg = data.frame(drugbank=all_neg[,1],uniprot=all_neg[,2])
my_all_neg = anti_join(all_neg,postive)
n_P = length(postive[,1])

neg = my_all_neg
dr_id = data.frame(dr_id = sort(unique(neg[,1])))
p_id = data.frame(p_id = sort(unique(neg[,2])))

colnames(neg)[1] = 'drugbank_id'
colnames(neg)[2] = 'uniprot_id'

n_1 = matrix(nr=length(dr_id[,1]),nc=2)
k = 1
for(i in 1:length(dr_id[,1]))
{
  id = data.frame(drugbank_id=dr_id[i,1])
  drug = semi_join(neg,id,by="drugbank_id")
  random = sample(1:length(drug[,1]),1)
  n_1[k,1] = drug[random,1]
  n_1[k,2] = drug[random,2]
  k = k+1
}

n_2 = matrix(nr=length(p_id[,1]),nc=2)
k = 1
for(i in 1:length(p_id[,1]))
{
  uniprot_id = p_id[i,1]
  id = data.frame(uniprot_id)
  protein = semi_join(neg,id,by="uniprot_id")
  random = sample(1:length(protein[,1]),1)
  n_2[k,1] = protein[random,1]
  n_2[k,2] = protein[random,2]
  k = k+1
}
n_1_2 = rbind(n_1,n_2)
n_1_2_new = unique(n_1_2)

rand = sample(1:length(n_1_2_new[,1]),n_P)
n_1_2_news = n_1_2_new[rand,]
write.csv(n_1_2_news,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/neg_DTI-rand_187.csv",row.names =FALSE,quote = F)










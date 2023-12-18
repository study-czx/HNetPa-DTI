library(dplyr)
my_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/DTI_8020.csv",header=T)
my_Drug = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/Drug_1409.csv",header=T)
my_Protein = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/Protein_1648.csv",header=T)

# candidate
neg3 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/my_need_neg3.csv",header=T)
neg5 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/my_need_neg5.csv",header=T)
colnames(neg3)[1] = 'drugbank_id'
colnames(neg3)[2] = 'uniprot_id'
colnames(neg5)[1] = 'drugbank_id'
colnames(neg5)[2] = 'uniprot_id'

neg_all = rbind(neg3,neg5)
x = unique(neg_all[,1])
y = unique(neg_all[,2])

# need_neg3 first
n_P = length(my_DTI[,1])
neg = neg3
dr_id = data.frame(dr_id = sort(unique(neg[,1])))
p_id = data.frame(p_id = sort(unique(neg[,2])))

# for each drug and protein, select 2 negative samples (if only 1,select 1 negative sample)
library(dplyr)

# for each drug in neg3
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

# for each protein in neg3
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

# combine them
negative3_1 = rbind(n_1,n_2)
negative3_1 = unique(negative3_1)

###########################################################################################
# for drug and protein not in need_neg3(l_b=3 and l_h>=3), other from need_neg5
dr_id2 = setdiff(unlist(neg5[,1]),unlist(neg3[,1]))
p_id2 = setdiff(unlist(neg5[,2]),unlist(neg3[,2]))

neg = neg5
dr_id = data.frame(dr_id = sort(unique(dr_id2)))
p_id = data.frame(p_id = sort(unique(p_id2)))


# for each drug
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
# for each protein
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

negative5_1 = rbind(n_1,n_2)
negative5_1 = unique(negative5_1)



################################################################################################################
# for drug and protein not in l_h>=3, 9 drugs and 2 proteins, select from N3
remain_drug = setdiff(unlist(my_Drug),unlist(unique(neg_all[,1])))
remain_protein = setdiff(unlist(my_Protein),unlist(unique(neg_all[,2])))
N3 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N3.csv",header=T)

neg = N3
colnames(neg)[1] = 'drugbank_id'
colnames(neg)[2] = 'uniprot_id'
dr_id = data.frame(dr_id = sort(unique(remain_drug)))
p_id = data.frame(p_id = sort(unique(remain_protein)))


# for each drug
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
# for each protein
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

negative_2 = rbind(n_1,n_2)
negative_2 = unique(negative_2)

# combine
n_this_1 = unique(rbind(negative3_1,negative5_1,negative_2))
x = unique(n_this_1[,1])
y = unique(n_this_1[,2])

# each drug and each protein has negative sample, then random select from need_neg3
num_remain = n_P-length(n_this_1[,1])

n_this_1_new = data.frame(drugbank_id = n_this_1[,1],uniprot_id = n_this_1[,2])
remain_negs = anti_join(neg3,n_this_1_new,by=c("drugbank_id","uniprot_id"))

rand = sample(1:length(remain_negs[,1]),num_remain)
k = 1
n_3 = matrix(nr=num_remain,nc=2)
for(i in 1:num_remain)
{
  m = rand[i]
  n_3[k,1] = remain_negs[m,1]
  n_3[k,2] = remain_negs[m,2]
  k = k+1
}
n_3 = data.frame(drugbank_id =n_3[,1],uniprot_id = n_3[,2])
n_all = rbind(n_this_1_new,n_3)
n_all = unique(n_all)

x = unique(n_all[,1])
y = unique(n_all[,2])

write.csv(n_all,file = "D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/neg_DTI-net_8020.csv",row.names =FALSE,quote = F)

###############################################################################################################

# for DTI-extra set
n0 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/extra_neg3.csv",header= T)
dr_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/Drug_111.csv",header=T)
p_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/Protein_123.csv",header=T)
postive = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/DTI_187.csv",header= T)

n_P = length(postive[,1])
neg = n0
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
# n_2
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

x = unique(n_1_2_news[,1])
y = unique(n_1_2_news[,2])

write.csv(n_1_2_news,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/neg_DTI-net_187.csv",row.names =FALSE,quote = F)




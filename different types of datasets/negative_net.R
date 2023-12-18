library(dplyr)

get_negative_data <- function(neg, dr_id, p_id, type_node) {
  k = 1
  if (type_node == 'drug')
  {
    negative_data = matrix(nr=length(dr_id[,1]),nc=2)
    this_id = dr_id
  }
  else
  {
    negative_data = matrix(nr=length(p_id[,1]),nc=2)
    this_id = p_id
  }
  for(i in 1:length(this_id[,1]))
  {
    if (type_node == 'drug')
    {
      id = data.frame(drugbank_id=dr_id[i,1])
      negative_node = semi_join(neg,id,by="drugbank_id")
    }
    else
    {
      id = data.frame(uniprot_id = p_id[i,1])
      negative_node = semi_join(neg,id,by="uniprot_id")
    }
    n_negative_sample = length(negative_node[,1])
    # cal
    if(n_negative_sample>1)
    {
      random = sample(1:n_negative_sample,1)
      for(j in 1:length(random))
      {
        ids = random[j]
        negative_data[k,1] = negative_node[ids,1]
        negative_data[k,2] = negative_node[ids,2]
        k = k+1
      }
    }
    else
    {
      negative_data[k,1] = negative_node[1,1]
      negative_data[k,2] = negative_node[1,2]
      k = k+1
    }
  }
  negative_data = na.omit(negative_data)
  return(negative_data)
}


enzymes_DTI = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/Enzymes.csv", header=T)
gpcr_DTI = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/GPCR.csv", header=T)
ic_DTI = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/IC.csv", header=T)
nc_DTI = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/NC.csv", header=T)
others_DTI = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/Others.csv", header=T)

other_DTIs = read.csv("D:/Users/czx/PycharmProjects/1-1HGDTI-code/case studies/KEGG&DrugC&CHEMBL_DTIs.csv",header=T)
colnames(other_DTIs) = c('drugbank_id','uniprot_id')

DTI_8020 = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI/new_DTI/max_DTI/DTI_8020.csv", header=T)
DTI_187 = read.csv("D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI/new_DTI/max_DTI/extra_DTI_187.csv", header=T)
colnames(DTI_8020) = c('drugbank_id','uniprot_id')
colnames(DTI_187) = c('drugbank_id','uniprot_id')

# 变量 enzymes_DTI， gpcr_DTI， ic_DTI， nc_DTI， others_DTI
positive = others_DTI
colnames(positive) = c('drugbank_id','uniprot_id')


# splict to positive1 and positive2
positive_1 = semi_join(positive, DTI_8020, by='drugbank_id')
positive_1 = semi_join(positive_1, DTI_8020, by='uniprot_id')
positive_2 = semi_join(positive, DTI_187, by='drugbank_id')
positive_2 = semi_join(positive_2, DTI_187, by='uniprot_id')
n_P = length(positive[,1])

########################################################################################################################
# negative2
n0_o = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/extra_neg3.csv",header= T)
colnames(n0_o) = c('drugbank_id','uniprot_id')

n0_o = semi_join(n0_o, positive_2, by='drugbank_id')
n0_o = semi_join(n0_o, positive_2, by='uniprot_id')
n0 = anti_join(n0_o,other_DTIs,by=c("drugbank_id","uniprot_id"))

x = unique(n0[,1])
y = unique(n0[,2])

neg = n0

dr_id = data.frame(sort(unique(positive_2$drugbank_id)))
p_id = data.frame(sort(unique(positive_2$uniprot_id)))

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

# rand = sample(1:length(n_1_2_new[,1]),n_P)
# n_1_2_news = n_1_2_new[rand,]
colnames(n_1_2_new) = c('drugbank_id','uniprot_id')

###############################################################################################################################
# negative1
# neg3， neg5
neg3_o = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/my_need_neg3.csv",header=T)
neg5_o = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/my_need_neg5.csv",header=T)
colnames(neg3_o) = c('drugbank_id','uniprot_id')
colnames(neg5_o) = c('drugbank_id','uniprot_id')

neg3_o = semi_join(neg3_o, positive_1, by='drugbank_id')
neg3_o = semi_join(neg3_o, positive_1, by='uniprot_id')
neg5_o = semi_join(neg5_o, positive_1, by='drugbank_id')
neg5_o = semi_join(neg5_o, positive_1, by='uniprot_id')

neg3 = anti_join(neg3_o,other_DTIs,by=c("drugbank_id","uniprot_id"))
neg5 = anti_join(neg5_o,other_DTIs,by=c("drugbank_id","uniprot_id"))

neg_all = rbind(neg3,neg5)
x = unique(neg_all[,1])
y = unique(neg_all[,2])

# need_neg3 first

neg = neg3
dr_id = data.frame(dr_id = sort(unique(neg[,1])))
p_id = data.frame(p_id = sort(unique(neg[,2])))

# for each drug in neg3
n_1 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'drug')
# for each protein in neg3
n_2 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'protein')

negative3_1 = unique(rbind(n_1,n_2))

# for drug and protein not in need_neg3(l_b=3 and l_h>=3), other from need_neg5
dr_id2 = setdiff(unlist(neg5[,1]),unlist(neg3[,1]))
p_id2 = setdiff(unlist(neg5[,2]),unlist(neg3[,2]))

neg = neg5
dr_id = data.frame(dr_id = sort(unique(dr_id2)))
p_id = data.frame(p_id = sort(unique(p_id2)))

# for each drug
n_1 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'drug')
# for each protein
n_2 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'protein')

negative5_1 = unique(rbind(n_1,n_2))
# negative5_1 = n_1

# for drug and protein not in l_h>=3,  select from N3
remain_drug = setdiff(unlist(positive_1$drugbank_id),unlist(unique(neg_all[,1])))
remain_protein = setdiff(unlist(positive_1$uniprot_id),unlist(unique(neg_all[,2])))
N3 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N3.csv",header=T)
N5 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N5.csv",header=T)
colnames(N3) = c('drugbank_id','uniprot_id')
colnames(N5) = c('drugbank_id','uniprot_id')

x = unique(N3[,1])
y = unique(N3[,2])

N3 = semi_join(N3, positive_1, by='drugbank_id')
N3 = semi_join(N3, positive_1, by='uniprot_id')
N3 = anti_join(N3,other_DTIs,by=c('drugbank_id','uniprot_id')) 

N5 = semi_join(N5, positive_1, by='drugbank_id')
N5 = semi_join(N5, positive_1, by='uniprot_id')
N5 = anti_join(N5,other_DTIs,by=c('drugbank_id','uniprot_id')) 


neg = N3
colnames(neg)[1] = 'drugbank_id'
colnames(neg)[2] = 'uniprot_id'

dr_id = data.frame(dr_id = sort(unique(remain_drug)))
p_id = data.frame(p_id = sort(unique(remain_protein)))

# for each drug
n_1 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'drug')
# for each protein
n_2 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'protein')
negative_2_1 = unique(rbind(n_1,n_2))
# negative_2_1 = n_1
############
neg = N5
colnames(neg)[1] = 'drugbank_id'
colnames(neg)[2] = 'uniprot_id'

dr_id = data.frame(dr_id = sort(unique(setdiff(remain_drug, unlist(negative_2_1[,1])))))
p_id = data.frame(p_id = sort(unique(setdiff(remain_protein, unlist(negative_2_1[,2])))))

# for each drug
n_1 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'drug')
# for each protein
n_2 = get_negative_data(neg, dr_id = dr_id, p_id = p_id, type_node = 'protein')
negative_2_2 = unique(rbind(n_1,n_2))
# negative_2_2 = n_1
# combine
n_this_1 = unique(rbind(negative3_1,negative5_1,negative_2_1,negative_2_2))
# n_this_1 = unique(rbind(negative3_1,negative5_1))
x = unique(n_this_1[,1])
y = unique(n_this_1[,2])


# each drug and each protein has negative sample, then random select from need_neg3
num_remain = n_P-length(n_this_1[,1])-length(n_1_2_new[,1])
# num_remain = n_P-length(n_this_1[,1])

n_this_1_new = data.frame(drugbank_id = n_this_1[,1],uniprot_id = n_this_1[,2])
remain_negs = anti_join(neg3,n_this_1_new,by=c("drugbank_id","uniprot_id"))
remain_negs2 = anti_join(neg5,n_this_1_new,by=c("drugbank_id","uniprot_id"))

if(num_remain<=length(remain_negs[,1]))
{
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
  n_all = unique(rbind(n_this_1_new,n_3))
} else{
  n_3 = remain_negs
  num_remain2 = n_P-length(n_this_1[,1])-length(n_1_2_new[,1])-length(remain_negs[,1])
  rand = sample(1:length(remain_negs2[,1]),num_remain2)
  k = 1
  n_5 = matrix(nr=num_remain2,nc=2)
  for(i in 1:num_remain2)
  {
    m = rand[i]
    n_5[k,1] = remain_negs2[m,1]
    n_5[k,2] = remain_negs2[m,2]
    k = k+1
  }
  n_5 = data.frame(drugbank_id =n_5[,1],uniprot_id = n_5[,2])
  n_all = unique(rbind(n_this_1_new,n_3,n_5))
}


# x = unique(n_all[,1])
# y = unique(n_all[,2])

n_all_out = unique(rbind(n_all, n_1_2_new))
# n_all_out = n_all
x = unique(n_all_out[,1])
y = unique(n_all_out[,2])

write.csv(n_all_out,file="D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/net_neg/Enzymes_neg.csv",row.names =FALSE,quote = F)
write.csv(n_all_out,file="D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/net_neg/GPCR_neg.csv",row.names =FALSE,quote = F)
write.csv(n_all_out,file="D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/net_neg/IC_neg.csv",row.names =FALSE,quote = F)
write.csv(n_all_out,file="D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/net_neg/NC_neg.csv",row.names =FALSE,quote = F)
write.csv(n_all_out,file="D:/Users/czx/PycharmProjects/DTI_data_Get/Drugbank/DTI_dataset_categorize/net_neg/Others_neg.csv",row.names =FALSE,quote = F)

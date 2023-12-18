library(dplyr)
all_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI_8207.csv",header=T)
all_Drug = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/Drug_1520.csv",header=T)
all_Protein = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/Protein_1771.csv",header=T)

# all_network
shortset_length = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/Dr_D_P_shortest_length.csv",header=T)
rownames(shortset_length) = unlist(all_Drug)
colnames(shortset_length) = unlist(all_Protein)

# DTI-benchmark set
my_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/DTI_8020.csv",header=T)
my_Drug = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/Drug_1409.csv",header=T)
my_Protein = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-benchmark_set/Protein_1648.csv",header=T)

# get shortest path length in DTI-benchmark set
my_shortest_length = shortset_length[unlist(my_Drug),unlist(my_Protein)]

# l_h >=3
dr_d_s_p = my_shortest_length
negative = matrix(nr=5000000,nc=2)
k = 1
for(i in 1:length(my_Drug[,1])){
  for(j in 1:length(my_Protein[,1])){
    if(dr_d_s_p[i,j]>=3)
    {
      negative[k,1] = my_Drug[i,1]
      negative[k,2] = my_Protein[j,1]
      k = k+1
    }
  }
}
neg1 = na.omit(negative)
x = unique(neg1[,1])
y = unique(neg1[,2])

# l_b =3 & l_b =5
N3 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N3.csv",header=T)
N5 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N5.csv",header=T)
neg_need = as.data.frame(neg1)

my_need_neg_lb3 = semi_join(N3,neg_need,by=c("V1","V2"))
my_need_neg_lb5 = semi_join(N5,neg_need,by=c("V1","V2"))

# both l_h>=3 & l_b=3
write.csv(my_need_neg_lb3,file = "D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/my_need_neg3.csv",row.names =FALSE,quote = F)
# both l_h>=3 & l_b=5
write.csv(my_need_neg_lb5,file = "D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/my_need_neg5.csv",row.names =FALSE,quote = F)


# DTI-extra set
extra_DTI = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/DTI_187.csv",header=T)
extra_Drug = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/Drug_111.csv",header=T)
extra_Protein = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/DTI-extra_set/Protein_123.csv",header=T)

# DTI-extra set    do not consider l_b
extra_shortest_length = shortset_length[unlist(extra_Drug),unlist(extra_Protein)]
dr_d_s_p = extra_shortest_length
negative = matrix(nr=5000000,nc=2)
k = 1
for(i in 1:length(extra_Drug[,1])){
  for(j in 1:length(extra_Protein[,1])){
    if(dr_d_s_p[i,j]>=3)
    {
      negative[k,1] = extra_Drug[i,1]
      negative[k,2] = extra_Protein[j,1]
      k = k+1
    }
  }
}
neg2 = na.omit(negative)
x = unique(neg1[,1])
y = unique(neg1[,2])
write.csv(neg2,file = "D:/Users/czx/PycharmProjects/HNetPa-DTI/Drugbank dataset/negative samples/extra_neg3.csv",row.names =FALSE,quote = F)

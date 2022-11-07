Only_Dr_P_shortest_length = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/Dr_P_shortest_length.csv",header=T)
all_Drug = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/Drug_1520.csv",header=T)
all_Protein = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/Protein_1771.csv",header=T)

rownames(Only_Dr_P_shortest_length) = unlist(all_Drug)
colnames(Only_Dr_P_shortest_length) = unlist(all_Protein)

my_Drug = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-benchmark_set/Drug_1409.csv",header=T)
my_Protein = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-benchmark_set/Protein_1648.csv",header=T)
my_only_Dr_P_shortest_length = Only_Dr_P_shortest_length[unlist(my_Drug),unlist(my_Protein)]

dr_d_s_p = my_only_Dr_P_shortest_length

negative3 = matrix(nr=5000000,nc=2)
negative5 = matrix(nr=5000000,nc=2)
negative7 = matrix(nr=5000000,nc=2)
negative9 = matrix(nr=5000000,nc=2)

k = 1
for(i in 1:length(my_Drug[,1])){
  for(j in 1:length(my_Protein[,1])){
    if(dr_d_s_p[i,j]==3)
    {
      negative3[k,1] = my_Drug[i,1]
      negative3[k,2] = my_Protein[j,1]
      k = k+1
    }
  }
}
neg3 = na.omit(negative3)
x = unique(neg3[,1])
y = unique(neg3[,2])

k = 1
for(i in 1:length(my_Drug[,1])){
  for(j in 1:length(my_Protein[,1])){
    if(dr_d_s_p[i,j]==5)
    {
      negative5[k,1] = my_Drug[i,1]
      negative5[k,2] = my_Protein[j,1]
      k = k+1
    }
  }
}
neg5 = na.omit(negative5)
x = unique(neg5[,1])
y = unique(neg5[,2])

k = 1
for(i in 1:length(my_Drug[,1])){
  for(j in 1:length(my_Protein[,1])){
    if(dr_d_s_p[i,j]==7)
    {
      negative7[k,1] = my_Drug[i,1]
      negative7[k,2] = my_Protein[j,1]
      k = k+1
    }
  }
}
neg7 = na.omit(negative7)
x = unique(neg7[,1])
y = unique(neg7[,2])

k = 1
for(i in 1:length(my_Drug[,1])){
  for(j in 1:length(my_Protein[,1])){
    if(dr_d_s_p[i,j]>=9)
    {
      negative9[k,1] = my_Drug[i,1]
      negative9[k,2] = my_Protein[j,1]
      k = k+1
    }
  }
}
neg9 = na.omit(negative9)
x = unique(neg9[,1])
y = unique(neg9[,2])

write.csv(neg3,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N3.csv",row.names =FALSE,quote = F)
write.csv(neg5,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N5.csv",row.names =FALSE,quote = F)
write.csv(neg7,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N7.csv",row.names =FALSE,quote = F)
write.csv(neg9,file = "D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/DTI benchmark N3_5_7_9/N9.csv",row.names =FALSE,quote = F)

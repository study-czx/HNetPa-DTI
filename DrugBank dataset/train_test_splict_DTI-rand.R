library(dplyr)
P = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-benchmark_set/DTI_8020.csv",header=T)
N = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/neg_DTI-rand_8020.csv",header=T)

colnames(P)[1] = 'drug'
colnames(P)[2] = 'protein'
colnames(N)[1] = 'drug'
colnames(N)[2] = 'protein'

# random, task SR
write.csv(P,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/random/P.csv",row.names = F,quote = F)
write.csv(N,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/random/N.csv",row.names = F,quote = F)


# new_drug, task SD
drug_id1 = unique(P[,1])
drug_id2 = unique(N[,1])
drug_id_all = data.frame(drug = sort(intersect(drug_id1,drug_id2)))

protein_id1 = unique(P[,2])
protein_id2 = unique(N[,2])
protein_id_all = data.frame(protein = sort(intersect(protein_id1,protein_id2)))

sample_drug = data.frame(drug=drug_id_all[sample(length(drug_id_all[,1]),200),])
P1 = semi_join(P,sample_drug,by="drug")
N1 = semi_join(N,sample_drug,by="drug")

train_P1 = anti_join(P,P1,by=c("drug","protein"))
train_N1 = anti_join(N,N1,by=c("drug","protein"))

write.csv(P1,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug/P_test.csv",row.names = F,quote = F)
write.csv(N1,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug/N_test.csv",row.names = F,quote = F)
write.csv(train_P1,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug/P_train.csv",row.names = F,quote = F)
write.csv(train_N1,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug/N_train.csv",row.names = F,quote = F)

# new_protein, task SP
sample_protein = data.frame(protein=protein_id_all[sample(length(protein_id_all[,1]),200),])
P2 = semi_join(P,sample_protein,by="protein")
N2 = semi_join(N,sample_protein,by="protein")
train_P2 = anti_join(P,P2,by=c("drug","protein"))
train_N2 = anti_join(N,N2,by=c("drug","protein"))

write.csv(P2,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_protein/P_test.csv",row.names = F,quote = F)
write.csv(N2,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_protein/N_test.csv",row.names = F,quote = F)
write.csv(train_P2,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_protein/P_train.csv",row.names = F,quote = F)
write.csv(train_N2,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_protein/N_train.csv",row.names = F,quote = F)

# new_drug_protein, task SDP
train_P = P
train_N = N
test_P = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-extra_set/DTI_187.csv",header=T)
test_N = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/negative samples/neg_DTI-rand_187.csv",header=T)

write.csv(train_P,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug_protein/P_train.csv",row.names = F,quote = F)
write.csv(train_N,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug_protein/N_train.csv",row.names = F,quote = F)
write.csv(test_P,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug_protein/P_test.csv",row.names = F,quote = F)
write.csv(test_N,"D:/Users/czx/PycharmProjects/HNGO-DTI/Drugbank dataset/DTI-rand/new_drug_protein/N_test.csv",row.names = F,quote = F)



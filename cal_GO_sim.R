library(GOSemSim)

all_MF_GOterms = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/GO/GO_terms/MF_terms_1849.csv",header=T,encoding = "UTF-8")
all_BP_GOterms = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/GO/GO_terms/BP_terms_5199.csv",header=T,encoding = "UTF-8")
all_CC_GOterms = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/GO/GO_terms/CC_terms_744.csv",header=T,encoding = "UTF-8")

n_MF = length(all_MF_GOterms)
n_BP = length(all_BP_GOterms)
n_CC = length(all_CC_GOterms)

MF_data <- godata('org.Hs.eg.db', ont="MF", computeIC=FALSE)
BP_data <- godata('org.Hs.eg.db', ont="BP", computeIC=FALSE)
CC_data <- godata('org.Hs.eg.db', ont="CC", computeIC=FALSE)

MF_GOterms_matrix = termSim(all_MF_GOterms,all_MF_GOterms,semData = MF_data, method ="Wang")
BP_GOterms_matrix = termSim(all_BP_GOterms,all_BP_GOterms,semData = BP_data, method ="Wang")
CC_GOterms_matrix = termSim(all_CC_GOterms,all_CC_GOterms,semData = CC_data, method ="Wang")

write.csv(MF_GOterms_matrix,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/GO/GO_sim/MF_sim_1879.csv",row.names =FALSE)
write.csv(BP_GOterms_matrix,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/GO/GO_sim/BP_sim_5199.csv",row.names =FALSE)
write.csv(CC_GOterms_matrix,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/GO/GO_sim/CC_sim_744.csv",row.names =FALSE)




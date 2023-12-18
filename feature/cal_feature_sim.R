library(philentropy)
# cal drug sim
# Pubchem = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/Pubchem.csv",header=FALSE)

Pubchem = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/Pubchem.csv",header=FALSE)
All_drug_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/network_node/All_drug_id_2223.csv",header=T)
drug_id_1520 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/Drug_1520.csv",header=T)

row.names(Pubchem) = unlist(All_drug_id$drugbank_id)
Pubchem_1520 = Pubchem[unlist(drug_id_1520),]

Pubchem_sim_1520 = 1-distance(Pubchem_1520, method="jaccard")
write.csv(Pubchem_sim_1520,"D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/Pubchem_sim_1520.csv",row.names = F)

# protein
KSCTriad = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/seq/KSCTriad.csv",header=FALSE)
All_protein_id = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/network_node/All_protein_id_13816.csv",header=T)
protein_id_1771 = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/DrugBank dataset/Protein_1771.csv",header=T)

row.names(KSCTriad) = unlist(All_protein_id$uniprot_id)
KSCTriad_1771 = KSCTriad[unlist(protein_id_1771),]
KSCTriad_sim_1771 = 1-distance(KSCTriad_1771, method="tanimoto")
write.csv(KSCTriad_sim_1771,"D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/KSCTriad_sim_1771.csv",row.names = F)




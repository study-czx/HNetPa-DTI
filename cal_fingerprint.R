library(rcdk)
library(philentropy)
library(fingerprint)

data = read.csv("D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/drug_structure_2223.csv",header=T)
id = data[,1]
smiles = data[,3]

sp <- get.smiles.parser()
mols <- parse.smiles(smiles)
# maccs,pubchem,standard,extended,graph
# circular.type='ECFP0','ECFP2','ECFP4','ECFP6','FCFP0','FCFP2','FCFP4','FCFP6'
fps = lapply(mols,get.fingerprint,type='pubchem')
f = fp.to.matrix(fps)


write.csv(f,file="D:/Users/czx/PycharmProjects/HNetPa-DTI/feature/Pubchem.csv",row.names =FALSE)

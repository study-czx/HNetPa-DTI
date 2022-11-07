library(seqinr)

Protein_Seq = read.csv("D:/Users/czx/PycharmProjects/HNGO-DTI/feature/protein_seq_13816.csv",header= T)
id = Protein_Seq[,1]
seq = Protein_Seq[,2]

header_name = matrix(nr=length(id),nc=1)

for(i in 1:length(id))
{
  id1 = id[i]
  header = paste(id1,1,sep = "|")
  header = paste(header,"training",sep = "|")
  header_name[i] = header
}
header_name = as.list(header_name)
sequence = as.list(seq)
write.fasta(sequence, names = header_name, file='D:/Users/czx/PycharmProjects/HNGO-DTI/feature/protein_13816.fasta', open = "w", nbchar = 60, as.string = FALSE)

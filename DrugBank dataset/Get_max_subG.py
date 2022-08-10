import networkx as nx
import numpy as np
import pandas as pd

Drug_id = np.loadtxt(r"./Drug_1520.csv", dtype=str, delimiter=",", skiprows=1)
Protein_id = np.loadtxt(r"./Protein_1771.csv", dtype=str, delimiter=",", skiprows=1)
Drug_Protein = np.loadtxt(r"./DTI_8207.csv", dtype=str, delimiter=",", skiprows=1)

all_DTI = pd.DataFrame(Drug_Protein)
all_Drug = pd.DataFrame(Drug_id)
all_Protein = pd.DataFrame(Protein_id)

n_drug = len(Drug_id)
n_protein = len(Protein_id)
n_target = len(Drug_Protein)
print(n_drug, n_protein, n_target)

H = nx.Graph()

for i in range(len(Drug_id)):
    H.add_node(Drug_id[i], node_type='drug')
for k in range(len(Protein_id)):
    H.add_node(Protein_id[k], node_type='protein')
for j in range(len(Drug_Protein)):
    H.add_edge(Drug_Protein[j][0], Drug_Protein[j][1], edge_type='binds_to')

max_H = list(H.subgraph(c) for c in nx.connected_components(H))[0]
print(max_H)
DTI_bench = list(max_H.edges)
DTI_bench = pd.DataFrame(DTI_bench)
DTI_Drug = sorted(list(set(DTI_bench[0])))
DTI_protein = sorted(list(set(DTI_bench[1])))
print(DTI_bench)
print(len(DTI_bench))
print(len(DTI_Drug), len(DTI_protein))
DTI_drug_bench = pd.DataFrame(DTI_Drug)
DTI_protein_bench = pd.DataFrame(DTI_protein)

DTI_bench.to_csv('DTI-benchmark_set/DTI_8020.csv', index=False)
DTI_drug_bench.to_csv('DTI-benchmark_set/Drug_1409.csv', index=False)
DTI_protein_bench.to_csv('DTI-benchmark_set/Protein_1648.csv', index=False)

DTI_extra = pd.concat([all_DTI, DTI_bench]).drop_duplicates(keep=False)
DTI_Drug2 = sorted(list(set(DTI_extra[0])))
DTI_protein2 = sorted(list(set(DTI_extra[1])))
print(len(DTI_extra))
print(len(DTI_Drug2), len(DTI_protein2))
DTI_drug_extra = pd.DataFrame(DTI_Drug2)
DTI_protein_extra = pd.DataFrame(DTI_protein2)

DTI_extra.to_csv('DTI-extra_set/DTI_187.csv', index=False)
DTI_drug_extra.to_csv('DTI-extra_set/Drug_111.csv', index=False)
DTI_protein_extra.to_csv('DTI-extra_set/Protein_123.csv', index=False)
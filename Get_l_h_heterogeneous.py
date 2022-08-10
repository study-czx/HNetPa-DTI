import networkx as nx
import numpy as np
import pandas as pd

Drug_id = np.loadtxt(r"./DrugBank dataset/Drug_1520.csv", dtype=object, delimiter=",", skiprows=1)
Protein_id = np.loadtxt(r"./DrugBank dataset/Protein_1771.csv", dtype=object, delimiter=",", skiprows=1)

All_Drug_id = np.loadtxt(r"./network_node/All_drug_id_2223.csv", dtype=object, delimiter=",", skiprows=1)
All_Disease_id = np.loadtxt(r"./network_node/All_disease_id_7061.csv", dtype=object, delimiter=",", skiprows=1)
All_Protein_id = np.loadtxt(r"./network_node/All_protein_id_13816.csv", dtype=object, delimiter=",", skiprows=1)

Drug_Protein = np.loadtxt(r"./DrugBank dataset/DTI_8207.csv", dtype=object, delimiter=",", skiprows=1)
Drug_Disease_T = np.loadtxt(r"./network/Dr_D_t_21908.csv", dtype=object, delimiter=",", skiprows=1)
Drug_Disease_M = np.loadtxt(r"./network/Dr_D_m_39187.csv", dtype=object, delimiter=",", skiprows=1)
Protein_Disease_T = np.loadtxt(r"./network/P_D_t_1933.csv", dtype=object, delimiter=",", skiprows=1)
Protein_Disease_M = np.loadtxt(r"./network/P_D_m_29201.csv", dtype=object, delimiter=",", skiprows=1)
Drug_Drug_Inter = np.loadtxt(r"./network/Drugbank_DDI_574616.csv", dtype=object, delimiter=",", skiprows=1)
Protein_Protein_Inter = np.loadtxt(r"./network/Uniprot_PPI_164797.csv", dtype=object, delimiter=",", skiprows=1)

print(len(Drug_Protein), len(Drug_Disease_T), len(Drug_Disease_M))
print(len(Protein_Disease_T), len(Protein_Disease_M))
print(len(Drug_Drug_Inter), len(Protein_Protein_Inter))

H = nx.Graph()

for i in range(len(All_Drug_id)):
    H.add_node(All_Drug_id[i], node_type='drug')
for j in range(len(All_Disease_id)):
    H.add_node(All_Disease_id[j], node_type='disease')
for k in range(len(All_Protein_id)):
    H.add_node(All_Protein_id[k], node_type='protein')

for i in range(len(Drug_Protein)):
    H.add_edge(Drug_Protein[i][0], Drug_Protein[i][1], edge_type='binds_to')
for i in range(len(Drug_Disease_T)):
    H.add_edge(Drug_Disease_T[i][0], Drug_Disease_T[i][1], edge_type='Dr_D_T')
for i in range(len(Drug_Disease_M)):
    H.add_edge(Drug_Disease_M[i][0], Drug_Disease_M[i][1], edge_type='Dr_D_M')
for i in range(len(Protein_Disease_T)):
    H.add_edge(Protein_Disease_T[i][0], Protein_Disease_T[i][1], edge_type='P_D_T')
for i in range(len(Protein_Disease_M)):
    H.add_edge(Protein_Disease_M[i][0], Protein_Disease_M[i][1], edge_type='P_D_M')
# 添加DDI与PPI
for i in range(len(Drug_Drug_Inter)):
    H.add_edge(Drug_Drug_Inter[i][0], Drug_Drug_Inter[i][1], edge_type='DDI')
for i in range(len(Protein_Protein_Inter)):
    H.add_edge(Protein_Protein_Inter[i][0], Protein_Protein_Inter[i][1], edge_type='PPI')

print(H)
shortest_path_length = [[0 for j in range(len(Protein_id))] for i in range(len(Drug_id))]

for i in range(len(Drug_id)):
    for j in range(len(Protein_id)):
        shortest_path_length[i][j] = nx.shortest_path_length(H, Drug_id[i], Protein_id[j], weight=None, method='dijkstra')

df = pd.DataFrame(shortest_path_length)
df.to_csv('Drugbank dataset/Dr_D_P_shortest_length.csv', index=False)

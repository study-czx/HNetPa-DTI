import networkx as nx
import numpy as np
import pandas as pd

Drug_id = np.loadtxt(r"./Drug_1520.csv", dtype=object, delimiter=",", skiprows=1)
Protein_id = np.loadtxt(r"./Protein_1771.csv", dtype=object, delimiter=",", skiprows=1)
Drug_Protein = np.loadtxt(r"./DTI_8207.csv", dtype=object, delimiter=",", skiprows=1)

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

print(H)
shortest_path_length = [[0 for j in range(n_protein)] for i in range(n_drug)]

# print(nx.shortest_path_length(H, "DB00091", "P05166", weight=None, method='dijkstra'))
for i in range(len(Drug_id)):
    for j in range(len(Protein_id)):
        if nx.has_path(H, Drug_id[i], Protein_id[j])==True:
            shortest_path_length[i][j] = nx.shortest_path_length(H, Drug_id[i], Protein_id[j], weight=None, method='dijkstra')
        else:
            shortest_path_length[i][j] = 0

df = pd.DataFrame(shortest_path_length)
df.to_csv('Dr_P_shortest_length.csv', index=False)

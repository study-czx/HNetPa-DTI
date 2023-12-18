import pandas as pd
import numpy as np

# load all predict scores
Predict_scores = pd.read_csv("Predict_scores.csv")
# load all drug ids and protein ids
All_Drug_id = np.loadtxt(r"./DrugBank dataset/Drug_1520.csv", dtype=object, delimiter=",", skiprows=1)
All_Protein_id = np.loadtxt(r"./DrugBank dataset/Protein_1771.csv", dtype=object, delimiter=",", skiprows=1)
all_dataset_id = []
for i in range(len(All_Drug_id)):
    all_dataset_id.append(All_Drug_id[i])
for j in range(len(All_Protein_id)):
    all_dataset_id.append(All_Protein_id[j])

# get DTI samples(P and N)
DTI_P = pd.read_csv("DrugBank dataset/DTI-benchmark_set/DTI_8020.csv")
DTI_N = pd.read_csv("DrugBank dataset/negative samples/neg_DTI-net_8020.csv")
# get drug ids and protein ids in predict scores
Drug_id = pd.read_csv("DrugBank dataset/DTI-benchmark_set/Drug_1409.csv")
Protein_id = pd.read_csv("DrugBank dataset/DTI-benchmark_set/Protein_1648.csv")

# Get protein-pathway
protein_pathway = pd.read_csv("Pathway/uniprot_pathways_25161.csv")

# Get protein-GO
protein_MF = pd.read_csv("GO/GO_uniprot/GO_MF_9071.csv")
protein_BP = pd.read_csv("GO/GO_uniprot/GO_BP_18737.csv")
protein_CC = pd.read_csv("GO/GO_uniprot/GO_CC_9990.csv")
protein_MF.columns = ['uniprot', 'GO']
protein_BP.columns = ['uniprot', 'GO']
protein_CC.columns = ['uniprot', 'GO']
protein_GO = pd.concat([protein_MF, protein_BP, protein_CC])

# Get drug PubChem similarity and protein KSCTriad similarity
drug_PubChem_sim = pd.read_csv("feature/Pubchem_sim_1520.csv")
protein_KSCTriad_sim = pd.read_csv("feature/KSCTriad_sim_1771.csv")
# set index and columns
drug_PubChem_sim.columns, drug_PubChem_sim.index = All_Drug_id, All_Drug_id
protein_KSCTriad_sim.columns, protein_KSCTriad_sim.index = All_Protein_id, All_Protein_id

# Get network one-order similarity
network_similarities = pd.read_csv("feature/Dr_D_P_one_order_similarities.csv")
network_similarities.columns, network_similarities.index = all_dataset_id, all_dataset_id

def cal_need_protein_similarity(predict_scores_cal, selected_DTI, ano, type):
    col_name = type
    if type == 'pathway':
        col_name = 'pathway_id'
    ano_sims = []
    for j in range(len(selected_DTI)):
        protein_id = selected_DTI.iloc[j, 1]
        protein_ano1 = set(ano.loc[ano['uniprot'] == protein_id, [col_name]][col_name])
        if len(protein_ano1) == 0:
            continue
        for k in range(len(predict_scores_cal)):
            protein_id_predict = predict_scores_cal.iloc[k, 1]
            protein_ano2 = set(ano.loc[ano['uniprot'] == protein_id_predict, [col_name]][col_name])
            if len(protein_ano1.union(protein_ano2)) == 0:
                continue
            else:
                jaccard_similarity = len(protein_ano1.intersection(protein_ano2)) / len(
                    protein_ano1.union(protein_ano2))
            ano_sims.append(jaccard_similarity)
    return ano_sims


def cal_need_drug_similarity(predict_scores_cal, selected_DTI, ano, type):
    ano_sims = []
    for j in range(len(selected_DTI)):
        drug_id = selected_DTI.iloc[j, 0]
        drug_ano1 = set(ano.loc[ano['drugbank'] == drug_id, [type]][type])
        if len(drug_ano1) == 0:
            continue
        for k in range(len(predict_scores_cal)):
            drug_id_predict = predict_scores_cal.iloc[k, 0]
            drug_ano2 = set(ano.loc[ano['drugbank'] == drug_id_predict, [type]][type])
            if len(drug_ano1.union(drug_ano2)) == 0:
                continue
            else:
                jaccard_similarity = len(drug_ano1.intersection(drug_ano2)) / len(
                    drug_ano1.union(drug_ano2))
            ano_sims.append(jaccard_similarity)
    return ano_sims


def get_drug_similarity(predict_scores_cal, selected_DTI, sim):
    need_sims = []
    for j in range(len(selected_DTI)):
        drug_id = selected_DTI.iloc[j, 0]
        for k in range(len(predict_scores_cal)):
            drug_id_predict = predict_scores_cal.iloc[k, 0]
            drug_sim_this = sim[drug_id_predict][drug_id]
            need_sims.append(drug_sim_this)
    return need_sims


def get_protein_similarity(predict_scores_cal, selected_DTI, sim):
    need_sims = []
    for j in range(len(selected_DTI)):
        protein_id = selected_DTI.iloc[j, 1]
        for k in range(len(predict_scores_cal)):
            protein_id_predict = predict_scores_cal.iloc[k, 1]
            protein_sim_this = sim[protein_id_predict][protein_id]
            need_sims.append(protein_sim_this)
    return need_sims


# type = 'GO' or 'Pathway'
def cal_GO_Pathway_similarity(type):
    all_ano_sims_high = []
    all_ano_sims_lower = []
    if type == 'GO':
        ano = protein_GO
    elif type == 'pathway':
        ano = protein_pathway
    for i in range(len(Drug_id)):
        drug_id = Drug_id.iloc[i, 0]
        # print(drug_id)
        drug_predict_scores = Predict_scores.loc[
            Predict_scores['drugbank_id'] == drug_id, ['drugbank_id', 'uniprot_id', 'scores']]
        drug_predict_scores_high = drug_predict_scores.loc[drug_predict_scores['scores'] > 0.99]
        drug_predict_scores_lower = drug_predict_scores.loc[drug_predict_scores['scores'] < 0.01]
        selected_DTI_P = DTI_P.loc[DTI_P['0'] == drug_id]
        drug_ano_sims_high = cal_need_protein_similarity(drug_predict_scores_high, selected_DTI_P, ano, type)
        drug_ano_sims_lower = cal_need_protein_similarity(drug_predict_scores_lower, selected_DTI_P, ano, type)
        if len(drug_ano_sims_high) != 0:
            all_ano_sims_high.append(np.mean(drug_ano_sims_high))
        if len(drug_ano_sims_lower) != 0:
            all_ano_sims_lower.append(np.mean(drug_ano_sims_lower))
    return np.mean(all_ano_sims_high), np.mean(all_ano_sims_lower)


# type = 'KSCTriad' or 'PubChem'
def cal_other_similarity(type):
    all_sims_high = []
    all_sims_lower = []
    if type == 'PubChem' or type == 'protein_network':
        if type == 'PubChem':
            similarities = drug_PubChem_sim
        else:
            similarities = network_similarities
        for i in range(len(Protein_id)):
            protein_id = Protein_id.iloc[i, 0]
            # print(protein_id)
            protein_predict_scores = Predict_scores.loc[
                Predict_scores['uniprot_id'] == protein_id, ['drugbank_id', 'uniprot_id', 'scores']]
            protein_predict_scores_high = protein_predict_scores.loc[protein_predict_scores['scores'] > 0.99]
            protein_predict_scores_lower = protein_predict_scores.loc[protein_predict_scores['scores'] < 0.01]
            selected_DTI_P = DTI_P.loc[DTI_P['1'] == protein_id]
            drug_sims_high = get_drug_similarity(protein_predict_scores_high, selected_DTI_P, similarities)
            drug_sims_lower = get_drug_similarity(protein_predict_scores_lower, selected_DTI_P, similarities)
            if len(drug_sims_high) != 0:
                all_sims_high.append(np.mean(drug_sims_high))
            if len(drug_sims_lower) != 0:
                all_sims_lower.append(np.mean(drug_sims_lower))
    elif type == 'KSCTriad' or type == 'drug_network':
        if type == 'KSCTriad':
            similarities = protein_KSCTriad_sim
        else:
            similarities = network_similarities
        for i in range(len(Drug_id)):
            drug_id = Drug_id.iloc[i, 0]
            # print(drug_id)
            drug_predict_scores = Predict_scores.loc[
                Predict_scores['drugbank_id'] == drug_id, ['drugbank_id', 'uniprot_id', 'scores']]
            drug_predict_scores_high = drug_predict_scores.loc[drug_predict_scores['scores'] > 0.99]
            drug_predict_scores_lower = drug_predict_scores.loc[drug_predict_scores['scores'] < 0.01]
            selected_DTI_P = DTI_P.loc[DTI_P['0'] == drug_id]
            protein_sims_high = get_protein_similarity(drug_predict_scores_high, selected_DTI_P, similarities)
            protein_sims_lower = get_protein_similarity(drug_predict_scores_lower, selected_DTI_P, similarities)
            if len(protein_sims_high) != 0:
                all_sims_high.append(np.mean(protein_sims_high))
            if len(protein_sims_lower) != 0:
                all_sims_lower.append(np.mean(protein_sims_lower))
    return np.mean(all_sims_high), np.mean(all_sims_lower)

# select different types, 'GO', 'pathway', 'PubChem', 'KSCTriad', 'drug_network', 'protein_network'

# types = ['GO', 'pathway']
# types = ['PubChem', 'KSCTriad', 'drug_network', 'protein_network']
types = ['GO', 'pathway', 'PubChem', 'KSCTriad', 'drug_network', 'protein_network']

for type in types:
    print('type: ', type)
    if type == 'GO' or type == 'pathway':
        cal_high_scores, cal_lower_scores = cal_GO_Pathway_similarity(type=type)
        print('high scores: ', np.mean(cal_high_scores))
        print('lower scores: ', np.mean(cal_lower_scores))
    elif type == 'KSCTriad' or type == 'PubChem' or type == 'drug_network' or type == 'protein_network':
        cal_high_scores, cal_lower_scores = cal_other_similarity(type=type)
        print('high scores: ', np.mean(cal_high_scores))
        print('lower scores: ', np.mean(cal_lower_scores))

import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import torch.nn.functional as F
import funcs
from dgl.nn.pytorch import HeteroGraphConv as HGCN
from dgl.nn.pytorch import GraphConv as GCN
from dgl.nn.pytorch import SAGEConv as SAGE
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
funcs.setup_seed(1)

agg = "pool"
drop_feat = 0
num_epoches = 300
n_hidden = 128
th = 0.005

data_type = "DrugBank dataset/DTI-net dataset/"
type = "random"

lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
wds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]
b_sizes = [16, 32, 64, 128, 256]

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# nodes in drug-protein-disease network
Drug_id = np.loadtxt(r"./network_node/All_drug_id_2223.csv", dtype=object, delimiter=",", skiprows=1)
Protein_id = np.loadtxt(r"./network_node/All_protein_id_13816.csv", dtype=object, delimiter=",", skiprows=1)
Disease_id = np.loadtxt(r"./network_node/All_disease_id_7061.csv", dtype=object, delimiter=",", skiprows=1)

n_drugs, n_proteins, n_diseases = len(Drug_id), len(Protein_id), len(Disease_id)
print("number of drugs in the network: ", n_drugs)
print("number of proteins in the network", n_proteins)
print("number of diseases in the network", n_diseases)

# GO terms
MF_terms = np.loadtxt(r"./GO/GO_terms/MF_terms_1849.csv", dtype=object, delimiter=",", skiprows=1)
BP_terms = np.loadtxt(r"./GO/GO_terms/BP_terms_5199.csv", dtype=object, delimiter=",", skiprows=1)
CC_terms = np.loadtxt(r"./GO/GO_terms/CC_terms_744.csv", dtype=object, delimiter=",", skiprows=1)

n_mf, n_bp, n_cc = len(MF_terms), len(BP_terms), len(CC_terms)
print("number of GO_MF terms: ", n_mf)
print("number of GO_BP terms: ", n_bp)
print("number of GO_CC terms: ", n_cc)


Pathways_id = np.loadtxt(r"./Pathway/all_pathway_id_2392.csv", dtype=object, delimiter=",", skiprows=1)

n_pathways = len(Pathways_id)
print("number of pathways: ", n_pathways)

# features of drugs and proteins
Pubchem = np.loadtxt(r"./feature/Pubchem.csv", dtype=float, delimiter=",", skiprows=0)
KSCTriad = np.loadtxt(r"./feature/KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)

# GO-protein
GO_MF = np.loadtxt(r"./GO/GO_uniprot/GO_MF_9071.csv", dtype=object, delimiter=",", skiprows=1)
GO_BP = np.loadtxt(r"./GO/GO_uniprot/GO_BP_18737.csv", dtype=object, delimiter=",", skiprows=1)
GO_CC = np.loadtxt(r"./GO/GO_uniprot/GO_CC_9990.csv", dtype=object, delimiter=",", skiprows=1)

# GO-sim
MF_sim = np.loadtxt(r"./GO/GO_sim/MF_sim_1849.csv", dtype=object, delimiter=",", skiprows=1)
BP_sim = np.loadtxt(r"./GO/GO_sim/BP_sim_5199.csv", dtype=object, delimiter=",", skiprows=1)
CC_sim = np.loadtxt(r"./GO/GO_sim/CC_sim_744.csv", dtype=object, delimiter=",", skiprows=1)

# Pathway-protein
Protein_Pathway = np.loadtxt(r"./Pathway/uniprot_pathways_25161.csv", dtype=object, delimiter=",", skiprows=1)

# DDI and PPI
Drugbank_DDI = np.loadtxt(r"./network/Drugbank_DDI_574616.csv", dtype=object, delimiter=",", skiprows=1)
Uniprot_PPI = np.loadtxt(r"./network/Uniprot_PPI_164797.csv", dtype=object, delimiter=",", skiprows=1)

# Disease related
Dr_D_m = np.loadtxt(r"./network/Dr_D_m_39187.csv", dtype=object, delimiter=",", skiprows=1)
Dr_D_t = np.loadtxt(r"./network/Dr_D_t_21908.csv", dtype=object, delimiter=",", skiprows=1)
P_D_m = np.loadtxt(r"./network/P_D_m_29201.csv", dtype=object, delimiter=",", skiprows=1)

# id map
dr_id_map, p_id_map, d_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id), funcs.id_map(Disease_id)
mf_id_map, bp_id_map, cc_id_map = funcs.id_map(MF_terms), funcs.id_map(BP_terms), funcs.id_map(CC_terms)
# kegg_id_map, reactome_id_map, wiki_id_map = funcs.id_map(Pathways_KEGG), funcs.id_map(Pathways_Reactome), funcs.id_map(Pathways_Wikipathways)
pathway_id_map = funcs.id_map(Pathways_id)
# id map to integer in data
Dr_D_m_data = funcs.Get_index(Dr_D_m, dr_id_map, d_id_map)
Dr_D_t_data = funcs.Get_index(Dr_D_t, dr_id_map, d_id_map)
P_D_m_data = funcs.Get_index(P_D_m, p_id_map, d_id_map)

P_MF_data = funcs.Get_index(GO_MF, p_id_map, mf_id_map)
P_BP_data = funcs.Get_index(GO_BP, p_id_map, bp_id_map)
P_CC_data = funcs.Get_index(GO_CC, p_id_map, cc_id_map)

D_D_data = funcs.Get_index(Drugbank_DDI, dr_id_map, dr_id_map)
P_P_data = funcs.Get_index(Uniprot_PPI, p_id_map, p_id_map)


P_Pathway = funcs.Get_index(Protein_Pathway, p_id_map, pathway_id_map)
num_nodes_dict_Pathway = {'pathway': n_pathways, 'protein': n_proteins}
num_nodes_pathways = n_pathways

# construct network
num_nodes_dict = {'drug': n_drugs, 'disease': n_diseases, 'protein': n_proteins}
num_nodes_dict_GO = {'MF': n_mf, 'BP': n_bp, 'CC': n_cc, 'protein': n_proteins}

num_nodes_mf = n_mf
num_nodes_bp = n_bp
num_nodes_cc = n_cc

# HNet-DrPD
my_G = funcs.Get_Net_Graph(Dr_D_m_data, Dr_D_t_data, P_D_m_data, D_D_data, P_P_data, num_nodes_dict)
# protein-GO
GO2P_G = funcs.Get_GO2P_Graph(P_MF_data, P_BP_data, P_CC_data, num_nodes_dict_GO)
# protein-pathway
Path2P_G = funcs.Get_Pathway2P_Graph_one(P_Pathway, num_nodes_dict_Pathway)

# keep top 0.5% scores of GO sim
new_MF_sim = funcs.delete_smalle_sim(MF_sim, th)
new_BP_sim = funcs.delete_smalle_sim(BP_sim, th)
new_CC_sim = funcs.delete_smalle_sim(CC_sim, th)
MF_sim_Graph, MF_weight = funcs.Get_weight_GOsim_graph(new_MF_sim, num_nodes_mf)
BP_sim_Graph, BP_weight = funcs.Get_weight_GOsim_graph(new_BP_sim, num_nodes_bp)
CC_sim_Graph, CC_weight = funcs.Get_weight_GOsim_graph(new_CC_sim, num_nodes_cc)

print(GO2P_G)
print(Path2P_G)
# to GPU
my_G = my_G.to(device)
GO2P_G = GO2P_G.to(device)
MF_sim_Graph = MF_sim_Graph.to(device)
BP_sim_Graph = BP_sim_Graph.to(device)
CC_sim_Graph = CC_sim_Graph.to(device)
Path2P_G = Path2P_G.to(device)

# initial feature
finger_feats = Pubchem
seq_feats = KSCTriad

# feature to GPU
finger_feats = torch.as_tensor(torch.from_numpy(finger_feats), dtype=torch.float32).to(device)
seq_feats = torch.as_tensor(torch.from_numpy(seq_feats), dtype=torch.float32).to(device)

# one-hot encoding
disease_feats = torch.as_tensor(torch.from_numpy(np.identity(n_diseases)), dtype=torch.float32).to(device)

MF_feat = torch.as_tensor(torch.from_numpy(np.identity(n_mf)), dtype=torch.float32).to(device)
BP_feat = torch.as_tensor(torch.from_numpy(np.identity(n_bp)), dtype=torch.float32).to(device)
CC_feat = torch.as_tensor(torch.from_numpy(np.identity(n_cc)), dtype=torch.float32).to(device)
pathway_feat = torch.as_tensor(torch.from_numpy(np.identity(n_pathways)), dtype=torch.float32).to(device)

n_diseases_feature, n_finger_feature, n_seq_feature = len(disease_feats[0]), len(finger_feats[0]), len(seq_feats[0])
n_mf_feature, n_bp_feature, n_cc_feature = len(MF_feat[0]), len(BP_feat[0]), len(CC_feat[0])
n_pathway_feature = len(pathway_feat[0])

print("Disease_feature_length:", n_diseases_feature)
print("Drug_finger_length:", n_finger_feature, "Protein_seq_length:", n_seq_feature)

def save_output(output_scores, number, best_dev_auc, best_t_result):
    output_scores[0][number], output_scores[1][number], output_scores[2][number], output_scores[3][number], \
    output_scores[4][number], output_scores[5][number], output_scores[6][number], output_scores[7][number] = best_dev_auc, \
    best_t_result[0], best_t_result[1], best_t_result[2], best_t_result[3], best_t_result[4], best_t_result[5], best_t_result[6]
    return output_scores

def get_eval_dict(this_output_score, epoch_type):
    mean_dev_auc, mean_acc, mean_auc, mean_aupr, mean_mcc, mean_f1, mean_recall, mean_precision = \
        np.nanmean(this_output_score[0]), np.nanmean(this_output_score[1]), np.nanmean(this_output_score[2]), \
        np.nanmean(this_output_score[3]), np.nanmean(this_output_score[4]), np.nanmean(this_output_score[5]), \
        np.nanmean(this_output_score[6]), np.nanmean(this_output_score[7])
    this_dict = {'lr': lr, 'wd': wd, 'b_size': b_size, 'num_epoches': epoch_type, 'mean_dev_auc': mean_dev_auc,
                 'mean_acc': mean_acc, 'mean_auc': mean_auc, 'mean_aupr': mean_aupr, 'mean_mcc': mean_mcc,
                 'mean_f1': mean_f1, 'mean_recall': mean_recall, 'mean_precision': mean_precision}
    return this_dict


class Dr_P_Embedding(nn.Module):
    def __init__(self):
        super(Dr_P_Embedding, self).__init__()
        self.drug_embedding = nn.Sequential(nn.Linear(n_finger_feature, n_hidden), nn.ReLU())
        self.protein_embedding = nn.Sequential(nn.Linear(n_seq_feature, n_hidden), nn.ReLU())

    def forward(self, Drug_feature, Protein_feature):
        h_dr = self.drug_embedding(Drug_feature)
        h_p = self.protein_embedding(Protein_feature)
        return h_dr, h_p


class Other_Embedding(nn.Module):
    def __init__(self):
        super(Other_Embedding, self).__init__()
        self.d_embedding = nn.Sequential(nn.Linear(in_features=n_diseases_feature, out_features=n_hidden), nn.ReLU())
        self.mf_embedding = nn.Sequential(nn.Linear(in_features=n_mf_feature, out_features=n_hidden), nn.ReLU())
        self.bp_embedding = nn.Sequential(nn.Linear(in_features=n_bp_feature, out_features=n_hidden), nn.ReLU())
        self.cc_embedding = nn.Sequential(nn.Linear(in_features=n_cc_feature, out_features=n_hidden), nn.ReLU())
        self.pathway_embedding = nn.Sequential(nn.Linear(in_features=n_pathway_feature, out_features=n_hidden), nn.ReLU())

    def forward(self, Disease_feature, MF_feature, BP_feature, CC_feature, Pathway_feature):
        h_d = self.d_embedding(Disease_feature)
        h_mf = self.mf_embedding(MF_feature)
        h_bp = self.bp_embedding(BP_feature)
        h_cc = self.cc_embedding(CC_feature)
        h_path = self.pathway_embedding(Pathway_feature)
        return h_d, h_mf, h_bp, h_cc, h_path


class All_Graph_Net(nn.Module):
    def __init__(self):
        super(All_Graph_Net, self).__init__()
        self.Graph_Net = HGCN(
            {'d-t-dr': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                            activation=F.relu),
             'd-m-dr': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                            activation=F.relu),
             'd-p': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                         activation=F.relu),
             'dr-t-d': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                            activation=F.relu),
             'dr-m-d': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                            activation=F.relu),
             'p-d': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                         activation=F.relu),
             'DDI': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                         activation=F.relu),
             'PPI': SAGE(in_feats=(n_hidden, n_hidden), out_feats=n_hidden, feat_drop=drop_feat, aggregator_type=agg,
                         activation=F.relu)},
            aggregate='sum')

    def forward(self, h_dr, h_p, h_d, my_G):
        h = {'drug': h_dr, 'protein': h_p, 'disease': h_d}
        h1 = self.Graph_Net(my_G, h)
        h2 = self.Graph_Net(my_G, h1)
        h_dr1, h_p1 = h1['drug'], h1['protein']
        h_dr2, h_p2 = h2['drug'], h2['protein']
        return h_dr1, h_p1, h_dr2, h_p2


class GO_sim_embedding(nn.Module):
    def __init__(self):
        super(GO_sim_embedding, self).__init__()
        self.MF_sim = GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu, allow_zero_in_degree=True)
        self.BP_sim = GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu, allow_zero_in_degree=True)
        self.CC_sim = GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu, allow_zero_in_degree=True)

    def forward(self, h_mf_new, h_bp_new, h_cc_new, MF_sim_Graph, BP_sim_Graph, CC_sim_Graph):
        mf_feat = self.MF_sim(MF_sim_Graph, h_mf_new) + h_mf_new
        bp_feat = self.BP_sim(BP_sim_Graph, h_bp_new) + h_bp_new
        cc_feat = self.CC_sim(CC_sim_Graph, h_cc_new) + h_cc_new
        return mf_feat, bp_feat, cc_feat

class GO_to_P(nn.Module):
    def __init__(self):
        super(GO_to_P, self).__init__()
        self.GO2P = HGCN(
            {'MF-p': GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu),
             'BP-p': GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu),
             'CC-p': GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu)},
            aggregate='sum')

    def forward(self, h_p, h_mf, h_bp, h_cc, GO2P_G):
        h = {'protein': h_p, 'MF': h_mf, 'BP': h_bp, 'CC': h_cc}
        h1 = self.GO2P(GO2P_G, h)
        h_p = h1['protein']
        return h_p

class Pathway_to_P(nn.Module):
    def __init__(self):
        super(Pathway_to_P, self).__init__()
        self.Path2P = HGCN({'path-p': GCN(n_hidden, n_hidden, norm='none', weight=True, activation=F.relu)},
                           aggregate='sum')

    def forward(self, h_p, h_path, Path2P_G):
        h = {'protein': h_p, 'pathway': h_path}
        h1 = self.Path2P(Path2P_G, h)
        h_p = h1['protein']
        return h_p

class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.dr_p_embedding = Dr_P_Embedding()
        self.other_embedding = Other_Embedding()
        self.GO_sim_embedding = GO_sim_embedding()
        self.GO2P = GO_to_P()
        self.Pathway2P = Pathway_to_P()
        self.All_Net = All_Graph_Net()
        self.connected_layer1 = nn.Sequential(nn.Linear(in_features=n_hidden * 7, out_features=n_hidden * 2), nn.BatchNorm1d(num_features=n_hidden * 2), nn.ReLU())
        self.connected_layer2 = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=n_hidden), nn.BatchNorm1d(num_features=n_hidden), nn.ReLU())
        self.connected_layer3 = nn.Sequential(nn.Linear(in_features=n_hidden, out_features=64),nn.BatchNorm1d(num_features=64), nn.ReLU())
        self.output = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_dr, x_p, finger_feats, seq_feats, disease_feat, MF_feat, BP_feat, CC_feat,
                Pathway_feat, GO2P_G, my_G, Path2P_G):
        h_dr_finger, h_p_seq = self.dr_p_embedding(finger_feats, seq_feats)
        h_d, h_mf, h_bp, h_cc, h_path = self.other_embedding(disease_feat, MF_feat, BP_feat, CC_feat, Pathway_feat)
        mf_feat, bp_feat, cc_feat = self.GO_sim_embedding(h_mf, h_bp, h_cc, MF_sim_Graph, BP_sim_Graph, CC_sim_Graph)
        h_p_GO = self.GO2P(h_p_seq, mf_feat, bp_feat, cc_feat, GO2P_G)
        h_p_pathway = self.Pathway2P(h_p_seq, h_path, Path2P_G)
        h_dr1, h_p1, h_dr2, h_p2 = self.All_Net(h_dr_finger, h_p_seq, h_d, my_G)
        # concat features
        dr_new = torch.cat((h_dr_finger, h_dr1, h_dr2), dim=1)
        p_new = torch.cat((h_p_seq, h_p1, h_p2, h_p_GO+h_p_pathway), dim=1)
        dr_feat, p_feat = dr_new[x_dr].squeeze(1), p_new[x_p].squeeze(1)
        h_dr_p = torch.cat((dr_feat, p_feat), dim=1)
        # DNN
        out = self.connected_layer3(self.connected_layer2(self.connected_layer1(h_dr_p)))
        out = self.sigmoid(self.output(out))
        return out

all_output_results = pd.DataFrame()

for lr in lrs:
    print('lr: ', lr)
    for wd in wds:
        print('wd: ', wd)
        for b_size in b_sizes:
            print('batch_size: ', b_size)
            output_score_epoch_50 = np.zeros(shape=(8, 5))
            output_score_epoch_100 = np.zeros(shape=(8, 5))
            output_score_epoch_150 = np.zeros(shape=(8, 5))
            output_score_epoch_200 = np.zeros(shape=(8, 5))
            output_score_epoch_250 = np.zeros(shape=(8, 5))
            output_score_epoch_300 = np.zeros(shape=(8, 5))
            for i in range(5):
                seed_type = "seed" + str(i + 1)
                train_P = np.loadtxt(data_type + seed_type + "/" + type + "/train_P.csv", dtype=str, delimiter=",", skiprows=1)
                dev_P = np.loadtxt(data_type + seed_type + "/" + type + "/dev_P.csv", dtype=str, delimiter=",", skiprows=1)
                test_P = np.loadtxt(data_type + seed_type + "/" + type + "/test_P.csv", dtype=str, delimiter=",", skiprows=1)
                train_N = np.loadtxt(data_type + seed_type + "/" + type + "/train_N.csv", dtype=str, delimiter=",", skiprows=1)
                dev_N = np.loadtxt(data_type + seed_type + "/" + type + "/dev_N.csv", dtype=str, delimiter=",", skiprows=1)
                test_N = np.loadtxt(data_type + seed_type + "/" + type + "/test_N.csv", dtype=str, delimiter=",", skiprows=1)
                print("number of DTI: ", len(train_P), len(dev_P), len(test_P))
                print("number of Negative DTI ", len(train_N), len(dev_N), len(test_N))
                train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map, p_id_map)
                dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map, p_id_map)
                test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map, p_id_map)
                train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
                dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
                test_loader = funcs.get_test_loader(test_X, test_Y, b_size)
                best_auc, best_epoch, best_extra = 0, 0, 0
                best_test = []
                losses = nn.BCELoss()
                model = My_Net().to(device)
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=40,verbose=False)
                # training
                for epoch in range(num_epoches):
                    train_loss = 0
                    train_scores, train_scores_label, train_labels = [], [], []
                    for step, (batch_x, batch_y) in enumerate(train_loader):
                        model.train()
                        b_x = batch_x.long().to(device)
                        b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = model(b_x_dr, b_x_p, finger_feats, seq_feats, disease_feats,
                                       MF_feat, BP_feat, CC_feat, pathway_feat, GO2P_G, my_G, Path2P_G)
                        score = torch.squeeze(output, dim=1)
                        loss = losses(score, b_y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        train_loss += loss.item()
                        scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                        train_scores = np.concatenate((train_scores, scores))
                        train_labels = np.concatenate((train_labels, label))
                    train_scores_label = funcs.computer_label(train_scores, 0.5)
                    train_avloss = train_loss / len(train_loader)
                    train_acc = skm.accuracy_score(train_labels, train_scores_label)
                    train_auc = skm.roc_auc_score(train_labels, train_scores)

                    dev_scores, dev_labels = [], []
                    test_scores, test_scores_label, test_labels = [], [], []
                    extra_scores, extra_scores_label, extra_labels = [], [], []
                    with torch.no_grad():
                        # validation
                        for step, (batch_x, batch_y) in enumerate(dev_loader):
                            model.eval()
                            b_x = batch_x.long().to(device)
                            b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                            b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                            b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                            output = model(b_x_dr, b_x_p, finger_feats, seq_feats, disease_feats,
                                           MF_feat, BP_feat, CC_feat, pathway_feat, GO2P_G, my_G,
                                           Path2P_G)
                            score = torch.squeeze(output, dim=1)
                            scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                            dev_scores = np.concatenate((dev_scores, scores))
                            dev_labels = np.concatenate((dev_labels, label))
                        dev_auc = skm.roc_auc_score(dev_labels, dev_scores)
                        # scheduler.step(dev_auc)
                        # testing
                        for step, (batch_x, batch_y) in enumerate(test_loader):
                            model.eval()
                            b_x = batch_x.long().to(device)
                            b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                            b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                            b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                            output = model(b_x_dr, b_x_p, finger_feats, seq_feats, disease_feats,
                                           MF_feat, BP_feat, CC_feat, pathway_feat, GO2P_G, my_G, Path2P_G)
                            score = torch.squeeze(output, dim=1)
                            scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                            test_scores = np.concatenate((test_scores, scores))
                            test_labels = np.concatenate((test_labels, label))
                        test_scores_label = funcs.computer_label(test_scores, 0.5)
                        test_acc = skm.accuracy_score(test_labels, test_scores_label)
                        test_auc = skm.roc_auc_score(test_labels, test_scores)
                        test_aupr = skm.average_precision_score(test_labels, test_scores)
                        test_mcc = skm.matthews_corrcoef(test_labels, test_scores_label)
                        test_F1 = skm.f1_score(test_labels, test_scores_label)
                        test_recall = skm.recall_score(test_labels, test_scores_label)
                        test_precision = skm.precision_score(test_labels, test_scores_label)
                    print(
                        'epoch:{},Train Loss: {:.4f},Train Acc: {:.4f},Train Auc: {:.4f},Dev Auc: {:.4f}, Test Acc: {:.4f},Test Auc: {:.4f},TestAUPR: {:.4f}'
                            .format(epoch, train_avloss, train_acc, train_auc, dev_auc, test_acc, test_auc, test_aupr))

                    if dev_auc >= best_auc:
                        best_auc = dev_auc
                        best_epoch = epoch
                        best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f'),
                                     format(test_mcc, '.4f'),
                                     format(test_F1, '.4f'), format(test_recall, '.4f'), format(test_precision, '.4f')]

                    # different epochs
                    if epoch == 49:
                        output_score_epoch_50 = save_output(output_score_epoch_50, i, best_auc, best_test)
                    if epoch == 99:
                        output_score_epoch_100 = save_output(output_score_epoch_100, i, best_auc, best_test)
                    if epoch == 149:
                        output_score_epoch_150 = save_output(output_score_epoch_150, i, best_auc, best_test)
                    if epoch == 199:
                        output_score_epoch_200 = save_output(output_score_epoch_200, i, best_auc, best_test)
                    if epoch == 249:
                        output_score_epoch_250 = save_output(output_score_epoch_250, i, best_auc, best_test)
                    if epoch == 299:
                        output_score_epoch_300 = save_output(output_score_epoch_300, i, best_auc, best_test)

            dict_50 = get_eval_dict(output_score_epoch_50, epoch_type=50)
            dict_100 = get_eval_dict(output_score_epoch_100, epoch_type=100)
            dict_150 = get_eval_dict(output_score_epoch_150, epoch_type=150)
            dict_200 = get_eval_dict(output_score_epoch_200, epoch_type=200)
            dict_250 = get_eval_dict(output_score_epoch_250, epoch_type=250)
            dict_300 = get_eval_dict(output_score_epoch_300, epoch_type=300)

            record_50 = pd.DataFrame.from_dict(dict_50, orient='index').T
            record_100 = pd.DataFrame.from_dict(dict_100, orient='index').T
            record_150 = pd.DataFrame.from_dict(dict_150, orient='index').T
            record_200 = pd.DataFrame.from_dict(dict_200, orient='index').T
            record_250 = pd.DataFrame.from_dict(dict_250, orient='index').T
            record_300 = pd.DataFrame.from_dict(dict_300, orient='index').T

            if all_output_results.empty:
                all_output_results = record_50
            else:
                all_output_results = pd.concat([all_output_results, record_50])

            all_output_results = pd.concat([all_output_results, record_100])
            all_output_results = pd.concat([all_output_results, record_150])
            all_output_results = pd.concat([all_output_results, record_200])
            all_output_results = pd.concat([all_output_results, record_250])
            all_output_results = pd.concat([all_output_results, record_300])

            all_output_results.to_csv('all_records_gridsearchcv.csv', index=False)








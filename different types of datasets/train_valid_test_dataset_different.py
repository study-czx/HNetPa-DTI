import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold

types = ["random"]

seeds = [1,2,3,4,5]

# data_type = "DTI-net"
data_types = ['Enzymes', 'GPCR', 'IC', 'NC', 'Others']
# neg_type = 'net'
neg_type = 'rand'

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio, seed):
    dataset = shuffle_dataset(dataset, seed)
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def Get_XY_dataset(P, N):
    P_list, N_list = [], []
    P_label, N_label = [], []
    for i in range(len(P)):
        P_list.append([P[i][0], P[i][1]])
        P_label.append(1)
    for j in range(len(N)):
        N_list.append([N[j][0], N[j][1]])
        N_label.append(0)
    X = np.concatenate((P_list, N_list))
    Y = np.concatenate((P_label, N_label))
    return X, Y

def trans_P_N(X_data, Y_data):
    P_data = []
    N_data = []
    for i in range(len(X_data)):
        if Y_data[i] == 1:
            P_data.append(X_data[i])
        elif Y_data[i] == 0:
            N_data.append(X_data[i])
    return P_data, N_data

def Get_5fold_data(data_type, output_path, neg_type):
    if neg_type == 'rand':
        P = np.loadtxt(data_type + "/" + data_type + ".csv", dtype=object, delimiter=",", skiprows=1)
        N = np.loadtxt(data_type + "/" + data_type + "_neg_rand.csv", dtype=object, delimiter=",", skiprows=1)
    else:
        P = np.loadtxt(data_type + "/" + data_type + ".csv", dtype=object, delimiter=",", skiprows=1)
        N = np.loadtxt(data_type + "/" + data_type + "_neg.csv", dtype=object, delimiter=",", skiprows=1)
    X, Y = Get_XY_dataset(P, N)
    k_folds = 5
    Kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=1)
    skf = Kfold.split(X, Y)
    n_fold = 0
    for train_index, test_index in skf:
        seed_type = 'fold' + str(n_fold + 1)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Train_P, Train_N = trans_P_N(X_train, Y_train)
        Test_P, Test_N = trans_P_N(X_test, Y_test)
        Train_P, Dev_P = split_dataset(Train_P, 0.9, seed=1)
        Train_N, Dev_N = split_dataset(Train_N, 0.9, seed=1)
        print(len(Train_P), len(Dev_P), len(Test_P))
        print(len(Train_N), len(Dev_N), len(Test_N))
        Train_P, Dev_P, Test_P = pd.DataFrame(Train_P), pd.DataFrame(Dev_P), pd.DataFrame(Test_P)
        Train_N, Dev_N, Test_N = pd.DataFrame(Train_N), pd.DataFrame(Dev_N), pd.DataFrame(Test_N)

        Train_P.to_csv(output_path + "/" + seed_type + '/train_P.csv', index=False)
        Dev_P.to_csv(output_path + "/" + seed_type + '/dev_P.csv', index=False)
        Test_P.to_csv(output_path + "/" + seed_type + '/test_P.csv', index=False)

        Train_N.to_csv(output_path + "/" + seed_type + '/train_N.csv', index=False)
        Dev_N.to_csv(output_path + "/" + seed_type + '/dev_N.csv', index=False)
        Test_N.to_csv(output_path + "/" + seed_type + '/test_N.csv', index=False)
        n_fold = n_fold+1

for data_type in data_types:
    print(data_type)
    output_path = data_type + "/" + neg_type
    Get_5fold_data(data_type, output_path, neg_type)

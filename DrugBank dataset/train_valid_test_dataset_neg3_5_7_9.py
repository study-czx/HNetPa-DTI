import pandas as pd
import numpy as np
import random

types = ["neg3","neg5","neg7","neg9"]

seeds = [1,2,3,4,5]

data_types = ["neg3_5_7_9","neg-bias3_5_7_9"]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio, seed):
    dataset = shuffle_dataset(dataset, seed)
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def Get_my_data(data_type, type, seed):
    P = np.loadtxt("DTI-benchmark_set/DTI_8020.csv", dtype=object, delimiter=",", skiprows=1)
    N = np.loadtxt("negative samples/" + data_type + "/" + type + "_8020.csv", dtype=object, delimiter=",", skiprows=1)
    Train_P, Test_P = split_dataset(P, 0.8, seed)
    Train_N, Test_N = split_dataset(N, 0.8, seed)
    Dev_P, Test_P = split_dataset(Test_P, 0.5, seed)
    Dev_N, Test_N = split_dataset(Test_N, 0.5, seed)
    return Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N

for data_type in data_types:
    output_path = "DTI-"+data_type
    for type in types:
        for seed in seeds:
            seed_type = "seed" + str(seed)
            Train_P, Dev_P, Test_P, Train_N, Dev_N, Test_N = Get_my_data(data_type, type, seed)
            print(len(Train_P), len(Dev_P), len(Test_P))
            print(len(Train_N), len(Dev_N), len(Test_N))
            print(Test_N[0])

            Train_P, Dev_P, Test_P = pd.DataFrame(Train_P), pd.DataFrame(Dev_P), pd.DataFrame(Test_P)
            Train_N, Dev_N, Test_N = pd.DataFrame(Train_N), pd.DataFrame(Dev_N), pd.DataFrame(Test_N)

            Train_P.to_csv(output_path + "/" + seed_type + "/" + type + '/train_P.csv', index=False)
            Dev_P.to_csv(output_path + "/" + seed_type + "/" + type + '/dev_P.csv', index=False)
            Test_P.to_csv(output_path + "/" + seed_type + "/" + type + '/test_P.csv', index=False)

            Train_N.to_csv(output_path + "/" + seed_type + "/" + type + '/train_N.csv', index=False)
            Dev_N.to_csv(output_path + "/" + seed_type + "/" + type + '/dev_N.csv', index=False)
            Test_N.to_csv(output_path + "/" + seed_type + "/" + type + '/test_N.csv', index=False)



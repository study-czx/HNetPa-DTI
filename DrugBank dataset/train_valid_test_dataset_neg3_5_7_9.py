from Kfold_funcs import Get_5fold_data
import numpy as np

types = ["neg3", "neg5", "neg7", "neg9"]

seeds = [1, 2, 3, 4, 5]

data_types = ["neg3_5_7_9", "neg-bias3_5_7_9"]


for data_type in data_types:
    print(data_type)
    output_path = "DTI-"+data_type
    for type in types:
        print(type)
        P = np.loadtxt("DTI-benchmark_set/DTI_8020.csv", dtype=object, delimiter=",", skiprows=1)
        N = np.loadtxt("negative samples/" + data_type + "/" + type + "_8020.csv", dtype=object, delimiter=",",skiprows=1)
        Get_5fold_data(P, N, type, output_path, k_folds=5, fold_name='fold')




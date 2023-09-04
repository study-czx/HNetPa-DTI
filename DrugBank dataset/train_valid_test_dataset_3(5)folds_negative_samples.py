import numpy as np
from Kfold_funcs import Get_5fold_data


# run seed = 1,2,3,4,5
fold = 5
fold_type = str(fold)
n_neg_samples = str(fold*8020)

types = [3, 5]
data_types = ['DTI-net dataset_with_3(5)fold_neg']


for data_type in data_types:
    print(data_type)
    output_path = data_type
    for type in types:
        n_neg_samples = str(type * 8020)
        save_type = str(type) + '_negative'
        print(save_type)
        P = np.loadtxt("DTI-benchmark_set/DTI_8020.csv", dtype=object, delimiter=",", skiprows=1)
        N = np.loadtxt("negative samples/neg_DTI-net_" + n_neg_samples + ".csv", dtype=object, delimiter=",",
                       skiprows=1)
        Get_5fold_data(P, N, save_type, output_path, k_folds=5, fold_name='fold')



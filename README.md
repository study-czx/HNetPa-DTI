HNGDTI: A drug-target interaction prediction framework based on heterogeneous network and gene ontology annotations
====
 First, we extracted the representations of drugs and proteins in the heterogeneous network with the heterogeneous graph neural networks. Furthermore, considering the correlation between GO terms, we utilized the graph neural networks in the GO term semantic similarity networks to enhance the representations of GO terms, which were passed to proteins via GO term-protein bipartite networks with graph neural networks. Finally, we concatenated the final drug and protein representations and fed them into the deep neural network.
    
The environment of HNGDTI
===
 python 3.8.8<br>
 cuda 10.2 - cudnn 7.6.5<br>
 pytorch 1.10.0<br>
 dgl 0.6.1<br>
 scikit-learn 0.22.1
 
Usage
===
 All data are csv files of binary relational data<br>
 Unzip the folders DTI-rand.rar, DTI-net.rar, neg3-9.rar, neg-b3-9.rar and GO.rar<br>
 python HNGDTI.py<br>
 For different settings in the paper, run xxx.py file with a different name (such as Only_add_GO.py).
    
Code and data
===
Construction of datasets（DTI-rand and DTI-net）
------
 First, the DrugBank dataset folder contains the files DTI_8207.csv, Drug_1520.csv, Protein_1771.csv.
 
### The detailed steps for the construction of the DTI-net dataset are as follows:
（1） Under the DrugBank dataset folder, run `Get_max_subG.py` to divide DTIs(8207) into two parts, i.e. DTI-benchmark set (8020) and DTI-extra set (187), and place them under the corresponding folders (DTI-benchmark_set and DTI-extra_set).<br>
（2）Under the DrugBank dataset folder, run `Get_shortest_length.py` to get the shortest path length l_b of all drug-protein pairs in the drug-protein bipartite network (Dr_P_shortest_length.csv).<br>
（3）Under the DrugBank dataset folder, run `Get_N3_N5_N7_N9.R` to classify all unlabeled drug-protein pairs in the DTI-benchmark set into 4 groups according to the shortest path length l_b in the drug-protein bipartite network, and name them as N3, N5, N7, N9 according to the shortest path length l_b (negative samples/DTI benchmark N3_5_7_9).<br>
（4）Run `Get_l_h_heterogeneous.py` to get the shortest path length l_h of all drug-protein pairs in the complete drug-protein-disease heterogeneous network (Dr_D_P_shortest_length.csv).<br>
（5）Under the DrugBank dataset folder, run `Get_need_neg.R` to select a reliable set of candidate negative samples based on l_h>=3. Specifically, to avoid hidden bias2 (i.e., l_b should be as short as possible), we filter out the set of all negative samples that satisfy both l_h>=3 and l_b=3 conditions (negative samples/my_need_neg3.csv) and filter out the set of all negative samples that satisfy both l_h>=3 and l_b=5 conditions (negative samples/my_need_neg5.csv).<br>
In addition, according to the drugs and proteins in the DTI-extra set (drugs and proteins outside the maximum connected subgraph in the drug-protein bipartite network), the set satisfying the condition l_h>=3 was filtered (negative samples/extra_neg3.csv).<br>
（6）Under the DrugBank dataset folder, run `select_negative_by_network.R` to randomly select negative samples and obtained neg_DTI-net_8020.csv and neg_DTI-net_187.csv, which combined with DTI_8020.csv and DTI_187.csv to form the DTI-net dataset.<br>
Specifically, to avoid hidden bias1, first, 2 negative samples are selected for each drug and protein from my_need_neg3 (1 negative sample if only 1 is available), for drugs and proteins not present in my_need_neg3, 2 negative samples are selected for each drug and protein from my_need_neg5 (1 negative sample if only 1 is available), and the remaining negative samples are randomly selected from my_need_neg3. (There are 9 drugs and 2 proteins without corresponding drug-protein pairs satisfying l_h>=3, and we select 2 negative samples from N3 for each of them)<br>
 
### The detailed steps for the construction of the DTI-rand dataset are as follows:
 Under the DrugBank dataset folder，run `select_negative_randomly.R` to randomly select negative samples and obtained neg_DTI-rand_8020.csv and neg_DTI-rand_187.csv, which combined with DTI_8020.csv and DTI_187.csv to form the DTI-rand dataset.<br>
 Specifically, to avoid hidden bias1, first, 2 negative samples are selected for each drug and protein from all unlabeled drug-protein pairs. Then, remaining negative samples are randomly selected from all unlabeled drug-protein pairs.
         
### Divide the testing set according to different prediction tasks (SR, SD, SP, and SDP)：
（1）Under the DrugBank dataset folder, run `train_test_splict_DTI-rand.R` to divide the training set and testing set according to different tasks (SR, SD, SP, SDP) on the DTI-rand dataset. (DTI-rand folder)<br>
（2）Under the DrugBank dataset folder, run `train_test_splict_DTI-net.R` to divide the training set and testing set according to different tasks (SR, SD, SP, SDP) on the DTI-net dataset. (DTI-net folder)

### The process of dividing the training set, validation set, and testing set is as follows：
Under the DrugBank dataset folder，run `train_valid_test_dataset.py` to divide the DTI-rand dataset and DTI-net dataset into training set, validation set and testing set, and divide them 5 times by random number seed 1-5. (DTI-rand dataset/ and DTI-net dataset/)

Equal numbers of negative samples are randomly selected from N3, N5, N7, and N9 for predicting DTIs (i.e., neg3, neg5, neg7, neg9, and neg3-b, neg5-b, neg7-b, neg9-b in the paper)
------
（1）Under the DrugBank dataset folder, run `select_negative_from_N3_5_7_9_randomly.R` to get 8020 negative samples from N3, N5, N7 and N9. (negative samples/neg3_5_7_9/neg3_8020.csv, neg5_8020.csv, neg7_8020.csv and neg9_8020.csv)<br>
（2）Under the DrugBank dataset folder, run `select_negative_with_bias_from_N3_5_7_9_randomly.R` to get 8020 negative samples with hidden bias1 from N3, N5, N7 and N9. (negative samples/neg_bias3_5_7_9/neg3_8020.csv, neg5_8020.csv, neg7_8020.csv and neg9_8020.csv)<br>
（3）Under the DrugBank dataset folder, run `train_valid_test_dataset_neg3_5_7_9.py` to divide the datasets into training set, validation set and testing set, and divide them 5 times by random number seed 1-5.（DTI-neg3_5_7_9/ and DTI-neg-bias3_5_7_9/）

Construction of the imbalance DTI-net dataset (Positive : Negative =1:3 and Positive : Negative =1:5)
------
（1）Under the DrugBank dataset folder, run `select_negative_with_3(5)folds_by_network.R` to select negative samples. (negative samples/neg_DTI-net_24060.csv and neg_DTI-net_40100.csv)<br>
（2）Under the DrugBank dataset folder, run `train_valid_test_dataset_3(5)folds_negative_samples.py` to divide the dataset into training set, validation set and testing set, and divide them 5 times by random number seed 1-5. (DTI-net dataset_with_3(5)fold_neg/3_negative/ and DTI-net dataset_with_3(5)fold_neg/5 negative/)

Calculation of feature
------
### Calculation of GO term similarities
Run `cal_GO_sim.R` to calculate the similarity values between GO terms. (GO/GO_sim/)

### Calculation of molecular fingerprint and protein descriptor
（1）Run `cal_fingerprint.R` to calculate the PubChem molecular fingerprint of drugs according to the SMILES strings of drugs.(feature/)<br>
（2）Run `trans_to_fasta.R` to convert the protein sequence protein_seq_13816.csv to protein_13816.fasta, and then use the iLearnPlus platform to calculate the KSCTriad descriptors for proteins.(feature/)

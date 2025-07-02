import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, nargs='?', default="./Data",
                       help="root dataset path")
    parse.add_argument('-dataset', '--dataset_topofallfeature', type=str, nargs='?', default="KIBA",
                       help="setting the dataset:davis or DrugBank or KIBA")

    # parameters for getting drug feature matrix
    parse.add_argument('-drug_macc_file', '--drug_macc_file_topofallfeature', type=str,
                       default="drug_features/MACC_features.xlsx")
    parse.add_argument('-drug_mogan_file', '--drug_mogan_file_topofallfeature', type=str,
                       default="drug_features/Mogan_features.xlsx")
    parse.add_argument('-drug_rdk_file', '--drug_rdk_file_topofallfeature', type=str,
                       default="drug_features/Topological_features.xlsx")

    # parameters for getting target feature matrix
    parse.add_argument('-AAC_file', '--AAC_file_topofallfeature', type=str, default="protein_features/AAC_features.txt")
    parse.add_argument('-CTD_file', '--CTD_file_topofallfeature', type=str, default="protein_features/CTD_features.txt")
    parse.add_argument('-Moran_file', '--Moran_file_topofallfeature', type=str,
                       default="protein_features/Moran_Correlation_features.txt")
    parse.add_argument('-PAAC_file', '--PAAC_file_topofallfeature', type=str,
                       default="protein_features/PAAC_features.txt")

    # parameters for the model
    parse.add_argument('-device', '--device_topofallfeature', type=str, nargs='?', default="cuda:0",
                       help="setting the cuda device")
    parse.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, nargs='?', default=10, help="k fold")

    parse.add_argument('-drug_sim_file', '--drug_sim_file_topofallfeature', type=str, nargs='?',
                       default="drug_affinity_mat.txt", help="setting the drug similarity file")
    parse.add_argument('-target_sim_file', '--target_sim_file_topofallfeature', type=str, nargs='?',
                       default="target_affinity_mat.txt", help="setting the target similarity file")
    parse.add_argument('-dti_mat', '--dti_mat_topofallfeature', type=str, nargs='?', default="dti_mat.xlsx",
                       help="setting the dti matrix file")

    parse.add_argument('-init_dim', '--init_gcn_dim', type=int, nargs='?', default=512,
                       help='defining the size of initial layer of GCN.')
    parse.add_argument('-hgcn_dim', '--hgcn_dim_topofallfeature', type=int, nargs='?', default=512,
                       help='defining the size of hidden layer of GCN.')
    parse.add_argument('-dropout', '--dropout_topofallfeature', type=float, nargs='?', default=0.5,
                       help='ratio of drop the graph nodes.')
    parse.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, nargs='?', default=2400,
                       help='number of epoch.')
    parse.add_argument('-lr', '--lr_topofallfeature', type=float, nargs='?', default=0.001, help='learning rate.')
    parse.add_argument('-topk', '--topk_topofallfeature', type=int, nargs='?', default=1,
                       help='ratio of positive samples and negative samples, i.e. 1 or 10.')
    parse.add_argument('-epoch_interv', '--epoch_interv_topofallfeature', type=int, nargs='?', default=100,
                       help='interval for showing the loss')
    parse.add_argument('-temperature', '--temperature_topofallfeature', type=int, nargs='?', default=0.2)

    parse.add_argument('-Semantic Drug Threshold', '--SemDT', type=int, nargs='?', default=0.1)
    parse.add_argument('-Semantic Target Threshold', '--SemTT', type=int, nargs='?', default=0.1)
    parse.add_argument('-Phenotype Drug Threshold', '--PheDT', type=int, nargs='?', default=0.35)
    parse.add_argument('-Phenotype Target Threshold', '--PheTT', type=int, nargs='?', default=0.3)

    parse.add_argument('-Sem Loss hyperparameter', '--SemLoss', type=int, nargs='?', default=0.003)
    parse.add_argument('-Phe Loss hyperparameter', '--PheLoss', type=int, nargs='?', default=0.18)

    config = parse.parse_args()
    return config

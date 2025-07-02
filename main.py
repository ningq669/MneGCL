import numpy as np
import pandas as pd
from utils import *
from data_reading import *
from models import GCN_decoder
import torch
import torch.nn as nn
from train_test_split import kf_split
from config_init import get_config
from cluster import Cluster

torch.cuda.manual_seed(42)
import os

if __name__ == "__main__":

    # get the parameters
    config = get_config()

    torch.manual_seed(42)
    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    drug_sim_file = config.drug_sim_file_topofallfeature
    target_sim_file = config.target_sim_file_topofallfeature
    dti_mat = config.dti_mat_topofallfeature
    device = config.device_topofallfeature
    n_splits = config.n_splits_topofallfeature
    init_dim = config.init_gcn_dim
    hgcn_dim = config.hgcn_dim_topofallfeature
    dropout = config.dropout_topofallfeature
    epoch_num = config.epoch_num_topofallfeature
    lr = config.lr_topofallfeature
    topk = config.topk_topofallfeature
    epoch_interv = config.epoch_interv_topofallfeature
    temperature = config.temperature_topofallfeature
    Sem_DT = config.SemDT
    Sem_TT = config.SemTT
    Phe_DT = config.PheDT
    Phe_TT = config.PheTT
    Sem_Loss = config.SemLoss
    Phe_Loss = config.PheLoss

    # data reading
    data_folder = os.path.join(root_path, dataset)
    drug_sim_path = os.path.join(data_folder, drug_sim_file)
    target_sim_path = os.path.join(data_folder, target_sim_file)
    DTI_path = os.path.join(data_folder, dti_mat)
    SR, SD, A_orig, A_orig_arr, known_sample, select_neg = read_data(drug_sim_path, target_sim_path,
                                                                     DTI_path)
    SR = SR[300:]
    SR = SR.flatten()
    SR = string_float(SR)
    SR = SR.reshape(1720, 1720)
    SD = SD[200:]
    SD = SD.flatten()
    SD = string_float(SD)
    SD = SD.reshape(220, 220)
    SR = Global_Normalize(SR)  # SR is a drug similarity matrix.
    SD = Global_Normalize(SD)  # SD is the target similarity matrix.
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    A_orig_list = A_orig.flatten()  # A_orig is a drug-target interaction matrix.

    drug_gau = get_gaussian(A_orig)
    tar_gau = get_gaussian(A_orig.T)
    np.fill_diagonal(drug_gau, 0)
    np.fill_diagonal(tar_gau, 0)

    # Neighbourhood-enhanced graph clustering
    cluster_drugs1, indices_drugs1 = Cluster(SR, Sem_DT)
    cluster_targets1, indices_targets1 = Cluster(SD, Sem_TT)
    cluster_drugs2, indices_drugs2 = Cluster(drug_gau, Phe_DT)
    cluster_targets2, indices_targets2 = Cluster(tar_gau, Phe_TT)

    # kfold CV
    train_all, test_all = kf_split(known_sample, n_splits)
    train_neg, test_neg = kf_split(select_neg, n_splits)

    overall_auroc = 0
    overall_aupr = 0
    overall_f1 = 0
    overall_acc = 0
    overall_recall = 0
    overall_specificity = 0
    overall_precision = 0

    fold_res = []

    for fold_int in range(n_splits):
        a = fold_int + 1
        print('fold_int:', fold_int)
        A_train_id = train_all[fold_int]
        A_test_id = test_all[fold_int]
        A_train = known_sample[A_train_id]
        A_test = known_sample[A_test_id]

        Atr_neg_id = train_neg[fold_int]
        Ate_neg_id = test_neg[fold_int]
        Atr_neg = select_neg[Atr_neg_id]
        Ate_neg = select_neg[Ate_neg_id]

        A_train_tensor = torch.LongTensor(A_train)
        A_test_tensor = torch.LongTensor(A_test)
        A_train_list = np.zeros_like(A_orig_arr)
        A_train_list[A_train] = 1
        A_test_list = np.zeros_like(A_orig_arr)
        A_test_list[A_test] = 1
        A_train_mask = A_train_list.reshape((A_orig.shape[0], A_orig.shape[1]))
        A_test_mask = A_test_list.reshape((A_orig.shape[0], A_orig.shape[1]))

        A_train_mat = A_train_mask
        # G is the normalized adjacent matrix
        G = Construct_G(A_train_mat, SR, SD).to(device)
        H_Data = H_data(config)
        training_negative_index = torch.LongTensor(Atr_neg)
        # initizalize the model
        train_W = torch.randn(hgcn_dim, hgcn_dim).to(device)
        train_W = nn.init.xavier_normal_(train_W)
        gcn_model = GCN_decoder(H_Data, in_dim=init_dim, hgcn_dim=hgcn_dim, train_W=train_W, dropout=dropout).to(device)
        gcn_optimizer = torch.optim.Adam(list(gcn_model.parameters()), lr=lr)
        # train procedure
        gcn_model.train()
        for epoch in range(epoch_num):
            A_hat, features = gcn_model(H_Data, G, drug_num, target_num)
            A_hat_list = A_hat.view(1, -1)
            train_sample = A_hat_list[0][A_train_tensor]
            train_score = torch.sigmoid(train_sample)
            nega_sample = A_hat_list[0][training_negative_index]
            nega_score = torch.sigmoid(nega_sample)
            true_label = np.hstack((np.ones(train_score.shape[0]), np.zeros(nega_score.shape[0])))
            true_label = np.array(true_label, dtype='float32')
            true_score = torch.from_numpy(true_label)
            pre_score = torch.cat([train_score, nega_score], dim=0)

            # calculate the loss
            loss_r = BCEloss(pre_score, true_score.cuda())

            drugs_feature = features[0:drug_num]
            targets_feature = features[drug_num:]

            drugs1_view1 = Cluster_view(drugs_feature, cluster_drugs1)
            drugs1_view3 = Cluster_core(drugs_feature, cluster_drugs1)
            drugs1_view2 = drugs_feature[indices_drugs1]
            drugs_cl_loss1 = InfoNCE_Se(drugs_feature, drugs1_view1, drugs1_view3, temperature)
            targets1_view1 = Cluster_view(targets_feature, cluster_targets1)
            targets1_view3 = Cluster_core(targets_feature, cluster_targets1)
            targets1_view2 = targets_feature[indices_targets1]
            targets_cl_loss1 = InfoNCE_Se(targets_feature, targets1_view1, targets1_view3, temperature)
            cl_loss1 = drugs_cl_loss1 + targets_cl_loss1

            drugs2_view1 = Cluster_view(drugs_feature, cluster_drugs2)
            drugs2_view3 = Cluster_core(drugs_feature, cluster_drugs2)
            drugs2_view2 = drugs_feature[indices_drugs2]
            drugs_cl_loss2 = InfoNCE(drugs2_view1, drugs2_view2, drugs2_view3, temperature)
            targets2_view1 = Cluster_view(targets_feature, cluster_targets2)
            targets2_view3 = Cluster_core(targets_feature, cluster_targets2)
            targets2_view2 = targets_feature[indices_targets2]
            targets_cl_loss2 = InfoNCE(targets2_view1, targets2_view2, targets2_view3, temperature)
            cl_loss2 = drugs_cl_loss2 + targets_cl_loss2

            loss = loss_r + (Sem_Loss * cl_loss1) + (Phe_Loss * cl_loss2)

            los_ = loss.detach().item()
            if epoch % epoch_interv == 0:
                print('loss:', los_)
            gcn_optimizer.zero_grad()
            loss.backward()
            gcn_optimizer.step()

        # test procedure
        gcn_model.eval()

        A_pre = torch.sigmoid(A_hat)
        A_prescore = A_pre.detach().cpu().numpy()
        test_negative_index = torch.LongTensor(Ate_neg)
        positive_result = A_hat_list[0][A_test_tensor]
        positive_score = torch.sigmoid(positive_result)
        negative_result = A_hat_list[0][test_negative_index]
        negative_score = torch.sigmoid(negative_result)
        positive_samples = positive_score.detach().cpu().numpy()
        negative_samples = negative_score.detach().cpu().numpy()
        positive_labels = np.ones_like(positive_samples)
        negative_labels = np.zeros_like(negative_samples)
        labels = np.hstack((positive_labels, negative_labels))
        scores = np.hstack((positive_samples, negative_samples))

        # calculate indicators
        TP, FP, FN, TN, fpr, tpr, auroc, aupr, f1_score, accuracy, recall, specificity, precision = get_metric(labels,
                                                                                                               scores)
        print('auroc:', auroc)
        print('aupr:', aupr)
        print('f1_score:', f1_score)
        print('acc:', accuracy)
        print('recall:', recall)
        print('specificity:', specificity)
        print('precision:', precision)
        res = [auroc, aupr, f1_score, accuracy, recall, precision]
        fold_res.append(res)
        overall_auroc += auroc
        overall_aupr += aupr
        overall_f1 += f1_score
        overall_acc += accuracy
        overall_recall += recall
        overall_specificity += specificity
        overall_precision += precision
    auroc_ = overall_auroc / n_splits
    aupr_ = overall_aupr / n_splits
    f1_ = overall_f1 / n_splits
    acc_ = overall_acc / n_splits
    recall_ = overall_recall / n_splits
    specificity_ = overall_specificity / n_splits
    precision_ = overall_precision / n_splits
    print('mean_auroc:', auroc_)
    print('mean_aupr:', aupr_)
    print('mean_f1:', f1_)
    print('mean_acc:', acc_)
    print('mean_recall:', recall_)
    print('mean_specificity:', specificity_)
    print('mean_precision:', precision_)

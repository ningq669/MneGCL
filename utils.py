import numpy as np
import torch
import torch.nn.functional as F
import os
import pandas as pd
import math


def get_metric(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])
    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return TP, FP, FN, TN, fpr, tpr, auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision


# Binary cross entropy loss
def BCEloss(pre, ture):
    bce_loss = torch.nn.BCELoss(reduction='mean')
    loss = bce_loss(pre, ture)
    return loss


# Contrastive learning loss
def InfoNCE(view1, view2, view3, temperature):
    view1, view2, view3 = F.normalize(view1, dim=1), F.normalize(view2, dim=1), F.normalize(view3, dim=1)
    pos_score = (view1 * view3).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def InfoNCE_Se(fea, view1, view3, temperature):
    fea = F.normalize(fea, dim=1)
    view1, view3 = F.normalize(view1, dim=1), F.normalize(view3, dim=1)
    pos_score = (view1 * view3).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    ttl_score = torch.matmul(fea, fea.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


def Cluster_view(view, cluster_point):
    cluster_view = []
    for cluster in cluster_point:
        for index in cluster:
            cluster_view.append(view[index])
    cluster_view = torch.stack(cluster_view)
    return cluster_view


def Cluster_core(view, cluster_point):
    cluster_core = [[lst[0] for _ in lst] for lst in cluster_point]
    flat_cluster_core = [item for sublist in cluster_core for item in sublist]
    cluster_view = []
    for index in flat_cluster_core:
        cluster_view.append(view[index])
    cluster_view = torch.stack(cluster_view)
    return cluster_view


def Construct_G(A_train_mat, SR, SD):
    SR_ = np.zeros_like(SR, dtype=float)
    SD_ = np.zeros_like(SD, dtype=float)
    A_row1 = np.hstack((SR_, A_train_mat))
    A_row2 = np.hstack((A_train_mat.T, SD_))
    G = np.vstack((A_row1, A_row2))
    G = G.astype(np.float64)
    G = torch.FloatTensor(G)
    G = Normalize_adj(G)
    return G


def H_data(config):
    Hdata = dict()

    root_path = config.root_path_topofallfeature
    dataset = config.dataset_topofallfeature
    macc_file_path = config.drug_macc_file_topofallfeature
    mogan_file_path = config.drug_mogan_file_topofallfeature
    rdk_file_path = config.drug_rdk_file_topofallfeature

    AAC_file = config.AAC_file_topofallfeature
    CTD_file = config.CTD_file_topofallfeature
    Moran_file = config.Moran_file_topofallfeature
    PAAC_file = config.PAAC_file_topofallfeature

    # Drug Features or Target features
    macc_path = os.path.join(root_path, dataset, macc_file_path)
    mogan_path = os.path.join(root_path, dataset, mogan_file_path)
    rdk_path = os.path.join(root_path, dataset, rdk_file_path)

    macc = pd.read_excel(macc_path, header=None).values
    macc = macc[1:, 1:]
    mogan = pd.read_excel(mogan_path, header=None).values
    mogan = mogan[1:, 1:]
    rdk = pd.read_excel(rdk_path, header=None).values
    rdk = rdk[1:, 1:]
    drug_fea = np.concatenate((macc, mogan, rdk), axis=1)

    # Drug Features or Target features
    AAC_path = os.path.join(root_path, dataset, AAC_file)
    CTD_path = os.path.join(root_path, dataset, CTD_file)
    Moran_path = os.path.join(root_path, dataset, Moran_file)
    PAAC_path = os.path.join(root_path, dataset, PAAC_file)

    aac = pd.read_csv(AAC_path, header=None, sep="\t").values
    aac = aac[1:, 1:]
    aac = aac.astype(float)
    ctd = pd.read_csv(CTD_path, header=None, sep="\t").values
    ctd = ctd[1:, 1:]
    ctd = ctd.astype(float)
    moran = pd.read_csv(Moran_path, header=None, sep="\t").values
    moran = moran[1:, 1:]
    moran = moran.astype(float)
    paac = pd.read_csv(PAAC_path, header=None, sep="\t").values
    paac = paac[1:, 1:]
    paac = paac.astype(float)
    target_fea = np.concatenate((aac, ctd, moran, paac), axis=1)

    Hdata['drug_F'] = torch.FloatTensor(drug_fea)
    Hdata['target_F'] = torch.FloatTensor(target_fea)

    return Hdata


# transform non-symmetric adjacency to symmetric ones
def Normalize_adj(adj):
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    return adj


def Global_Normalize(mat):
    max_val = np.max(mat)
    min_val = np.min(mat)
    mat = (mat - min_val) / (max_val - min_val)
    return mat


def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))

    return Gaussian

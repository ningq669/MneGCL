import os
import pandas as pd
import numpy as np


def read_data(drug_sim_path, target_sim_path, DTI_path):
    SR = pd.read_csv(drug_sim_path, header=None,
                     dtype='str').values  # data_folder is the path to the data folder and drug_sim_path is the path to the drug similarity file.
    SD = pd.read_csv(target_sim_path, header=None,
                     dtype='str').values  # target_sim_path is the path to the target similarity file.
    A_orig = pd.read_excel(DTI_path,
                           header=None).values  # DTI_path is the path to the drug target interaction file.
    A_orig_arr = A_orig.flatten()
    known_sample = np.nonzero(A_orig_arr)[0]

    rng = np.random.default_rng(seed=42)

    neg_samples = np.where(A_orig_arr == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :known_sample.shape[0]]
    neg_S = neg_samples_shuffled.flatten()

    return SR, SD, A_orig, A_orig_arr, known_sample, neg_S


def string_float(arr):
    for i in range(len(arr)):
        if isinstance(arr[i], str):
            arr[i] = float(arr[i])
        else:
            arr[i] = arr[i]
    return arr

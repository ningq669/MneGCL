import pandas as pd
import numpy as np
import os
import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common train config')
    # parameters for the data reading and train-test-splitting
    parse.add_argument('-root_path', '--root_path_topofallfeature', type=str, default="../Data/KIBA/protein_features")
    parse.add_argument('-ctdc_path', '--ctdc_path_topofallfeature', type=str, default="CTDC_features.txt")
    parse.add_argument('-ctdt_path', '--ctdt_path_topofallfeature', type=str, default="CTDT_features.txt")
    parse.add_argument('-ctdd_path', '--ctdd_path_topofallfeature', type=str, default="CTDD_features.txt")
    parse.add_argument('-out', '--out_topofallfeature', type=str, default="CTD_feature.xlsx")
    config = parse.parse_args()
    return config


def concate_ctd(root_path, ctdc_path, ctdt_path, ctdd_path, out):
    CTDC_path = os.path.join(root_path, ctdc_path)
    CTDC = pd.read_csv(CTDC_path)
    CTDT_path = os.path.join(root_path, ctdt_path)
    CTDT = pd.read_csv(CTDT_path)
    CTDD_path = os.path.join(root_path, ctdd_path)
    CTDD = pd.read_csv(CTDD_path)
    CTD = pd.concat([CTDC, CTDT, CTDD], axis=1)
    out_path = os.path.join(root_path, out)
    CTD.to_csv(out_path)


if __name__ == "__main__":
    config = get_config()
    root_path = config.root_path_topofallfeature
    ctdc_path = config.ctdc_path_topofallfeature
    ctdt_path = config.ctdt_path_topofallfeature
    ctdd_path = config.ctdd_path_topofallfeature
    out = config.out_topofallfeature
    concate_ctd(root_path, ctdc_path, ctdt_path, ctdd_path, out)

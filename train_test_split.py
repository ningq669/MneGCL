from sklearn.model_selection import KFold


# You can test it with seed=1.
# The final result is the average of seeds with none, and MneGCL is closer when seed=1.
def kf_split(known_sample, n_splits):
    kf = KFold(n_splits, shuffle=True, random_state=None)  # 10 fold
    train_all = []
    test_all = []
    for train_ind, test_ind in kf.split(known_sample):
        train_all.append(train_ind)
        test_all.append(test_ind)
    return train_all, test_all

import numpy as np

# arr_outcomes = np.load('../processed_data/arr_outcomes.npy', allow_pickle=True)

# split randomization over folds
"""Use 8:1:1 split"""
p_train = 0.80
p_val = 0.10
p_test = 0.10

n = 5333
n_train = round(n*p_train)
n_val = round(n*p_val)
n_test = n - (n_train+n_val)
print(n_train, n_val, n_test)
Nsplits = 5
for j in range(Nsplits):
    p = np.random.permutation(n)
    idx_train = p[:n_train]
    idx_val = p[n_train:n_train+n_val]
    idx_test = p[n_train+n_val:]
    np.save('../splits/PAMAP2_split'+str(j+1)+'.npy', (idx_train, idx_val, idx_test))

    # np.save('../splits/phy12_split_subset'+str(j+1)+'.npy', (idx_train, idx_val, idx_test))
print('split IDs saved')

# # check 128 split
# idx_train,idx_val,idx_test = np.load('../splits/phy12_split1.npy', allow_pickle=True)
# print(len(idx_train), len(idx_val), len(idx_test))
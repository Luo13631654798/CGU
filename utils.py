# pylint: disable=E1101
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import chain
from torch.nn import Dropout, Linear
import numpy as np
import math
from physionet import PhysioNet, get_data_min_max
from sklearn import model_selection
from sklearn import metrics
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.sparse.linalg import eigs

def nufft(x, t):
    N = len(x)
    f = np.array([i / N for i in range(N)])
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * t * f[k]))
    return X, f



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def get_mimiciii_data(batch_size=64, split=0):
    input_dim = 12
    x = np.load('data/MIMIC-III/final_input.npy')
    y = np.load('data/MIMIC-III/final_output.npy')
    x = x[:, :25, :]
    x = np.transpose(x, (0, 2, 1))

    # normalize values and time
    observed_vals, observed_mask, observed_tp = x[:, :,
                                                  :input_dim], x[:, :, input_dim:2*input_dim], x[:, :, -1]
    if np.max(observed_tp) != 0.:
        observed_tp = observed_tp / np.max(observed_tp)

    for k in range(input_dim):
        data_min, data_max = float('inf'), 0.
        for i in range(observed_vals.shape[0]):
            for j in range(observed_vals.shape[1]):
                if observed_mask[i, j, k]:
                    data_min = min(data_min, observed_vals[i, j, k])
                    data_max = max(data_max, observed_vals[i, j, k])
        #print(data_min, data_max)
        if data_max == 0:
            data_max = 1
        observed_vals[:, :, k] = (observed_vals[:, :, k] - data_min)/(data_max - data_min)
    # set masked out elements back to zero
    observed_vals[observed_mask == 0] = 0
    print(observed_vals[0], observed_tp[0])
    print(x.shape, y.shape)
    kfold = model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=0)
    splits = [(train_inds, test_inds)
              for train_inds, test_inds in kfold.split(np.zeros(len(y)), y)]   # 分为五折交叉验证
    x_train, y_train = x[splits[split][0]], y[splits[split][0]]
    test_data_x, test_data_y = x[splits[split]
                                 [1]], y[splits[split][1]]

    frac = int(0.8 * x_train.shape[0])    #  划分训练集和验证集
    train_data_x, val_data_x = x_train[:frac], x_train[frac:]
    train_data_y, val_data_y = y_train[:frac], y_train[frac:]

    print(train_data_x.shape, train_data_y.shape, val_data_x.shape, val_data_y.shape,
          test_data_x.shape, test_data_y.shape)
    print(np.sum(test_data_y))
    train_data_combined = TensorDataset(torch.from_numpy(train_data_x).float(),
                                        torch.from_numpy(train_data_y).long().squeeze())
    val_data_combined = TensorDataset(torch.from_numpy(val_data_x).float(),
                                      torch.from_numpy(val_data_y).long().squeeze())
    test_data_combined = TensorDataset(torch.from_numpy(test_data_x).float(),
                                       torch.from_numpy(test_data_y).long().squeeze())
    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)

    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "val_dataloader": val_dataloader,
                    "input_dim": input_dim}
    return data_objects

def get_p12_data(args):
    # 导入数据集字典列表 id length extended_static(6) arr(60,43) time(60)
    Pdict_list = np.load('./data/P12data/processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load('./data/P12data/processed_data/arr_outcomes.npy', allow_pickle=True)
    dataset_prefix = 'P12_'

    # interp_np = np.load("interp_data_48.npy", allow_pickle=True)
    # print(inter_np)

    # 分割数据集
    # k_fold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # idx_list = [(train_idx, test_idx) for train_idx, test_idx in k_fold.split(np.zeros(len(Pdict_list)), np.zeros(len(Pdict_list)))]
    fold = 5
    data_objects = []

    for i in range(fold):
        idx_base_path = 'data/P12data/splits/'
        idx_split_path = 'phy12_split' + str(i+1) + '.npy'
        idx_train, idx_val, idx_test = np.load(idx_base_path + idx_split_path, allow_pickle=True)

        # idx_train, idx_test = idx_list[i]
        # idx_val = idx_test[: len(idx_test) // 2]
        # idx_test = idx_test[len(idx_test) // 2:]
        #  输入
        Ptrain = Pdict_list[idx_train]
        Pval = Pdict_list[idx_val]
        Ptest = Pdict_list[idx_test]

        # 标签
        y = arr_outcomes[:, -1].reshape((-1, 1))
        ytrain = y[idx_train]  # 9590,1
        yval = y[idx_val]  # 1199,1
        ytest = y[idx_test]  # 1199,1

        # interp_train = interp_np[idx_train]
        # interp_val = interp_np[idx_val]
        # interp_test = interp_np[idx_test]

        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])

        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset='P12')
        #
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                     stdf, ms, ss, None)
        Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss, None)
        Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms,
                                                                                                 ss, None)
        # Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
        # Pval_tensor = Pval_tensor.permute(1, 0, 2)
        # Ptest_tensor = Ptest_tensor.permute(1, 0, 2)
        #
        # Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
        # Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        # Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)
        train_data_combined = TensorDataset(
            Ptrain_tensor, ytrain_tensor.long())
        val_data_combined = TensorDataset(
            Pval_tensor, yval_tensor.long())
        test_data_combined = TensorDataset(
            Ptest_tensor, ytest_tensor.long())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=args.batch_size, shuffle=False)
        val_dataloader = DataLoader(
            val_data_combined, batch_size=args.batch_size, shuffle=False)

        data_object = {"train_dataloader": train_dataloader,
                        "test_dataloader": test_dataloader,
                        "val_dataloader": val_dataloader,
                        "input_dim": F,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
        data_objects.append(data_object)
    return data_objects

def get_p19_data(args):
    # 导入数据集字典列表 id length extended_static(6) arr(60,43) time(60)
    Pdict_list = np.load('./data/P19data/processed_data/PT_dict_list_6.npy', allow_pickle=True)
    arr_outcomes = np.load('./data/P19data/processed_data/arr_outcomes_6.npy', allow_pickle=True)
    dataset_prefix = 'P19_'

    # 分割数据集
    # k_fold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # idx_list = [(train_idx, test_idx) for train_idx, test_idx in k_fold.split(np.zeros(len(Pdict_list)), np.zeros(len(Pdict_list)))]

    fold = 5
    data_objects = []

    for i in range(fold):
        idx_base_path = 'data/P19data/splits/'
        idx_split_path = 'phy19_split' + str(i+1) + '_new.npy'
        idx_train, idx_val, idx_test = np.load(idx_base_path + idx_split_path, allow_pickle=True)

        # idx_train, idx_test = idx_list[i]
        # idx_val = idx_test[: len(idx_test) // 2]
        # idx_test = idx_test[len(idx_test) // 2:]
        #  输入
        Ptrain = Pdict_list[idx_train]
        Pval = Pdict_list[idx_val]
        Ptest = Pdict_list[idx_test]

        # 标签
        y = arr_outcomes[:, -1].reshape((-1, 1))
        ytrain = y[idx_train]  # 9590,1
        yval = y[idx_val]  # 1199,1
        ytest = y[idx_test]  # 1199,1


        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])

        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset='P12')

        Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf,
                                                                                                     stdf, ms, ss, None)
        Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms, ss, None)
        Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf, ms,
                                                                                                 ss, None)
        # Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
        # Pval_tensor = Pval_tensor.permute(1, 0, 2)
        # Ptest_tensor = Ptest_tensor.permute(1, 0, 2)
        #
        # Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
        # Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
        # Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)
        train_data_combined = TensorDataset(
            Ptrain_tensor, ytrain_tensor.long())
        val_data_combined = TensorDataset(
            Pval_tensor, yval_tensor.long())
        test_data_combined = TensorDataset(
            Ptest_tensor, ytest_tensor.long())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=args.batch_size, shuffle=False)
        val_dataloader = DataLoader(
            val_data_combined, batch_size=args.batch_size, shuffle=False)

        data_object = {"train_dataloader": train_dataloader,
                        "test_dataloader": test_dataloader,
                        "val_dataloader": val_dataloader,
                        "input_dim": 36,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
        data_objects.append(data_object)
    return data_objects

def get_PAM_data(args):
    # 导入数据集 (5333,600,17)
    Pdict_list = np.load('./data/PAMdata/processed_data/PTdict_list.npy', allow_pickle=True)
    arr_outcomes = np.load('./data/PAMdata/processed_data/arr_outcomes.npy', allow_pickle=True)
    dataset_prefix = ''  # not applicable
    # 分割数据集
    k_fold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    idx_list = [(train_idx, test_idx) for train_idx, test_idx in k_fold.split(np.zeros(len(Pdict_list)), np.zeros(len(Pdict_list)))]

    data_objects = []

    for i in range(len(idx_list)):
        idx_train, idx_test = idx_list[i]
        idx_val = idx_test[: len(idx_test) // 2]
        idx_test = idx_test[len(idx_test) // 2:]
        #  输入
        Ptrain = Pdict_list[idx_train]
        Pval = Pdict_list[idx_val]
        Ptest = Pdict_list[idx_test]

        # 标签
        y = arr_outcomes[:, -1].reshape((-1, 1))
        ytrain = y[idx_train]  # 4266,1
        yval = y[idx_val]  # 533,1
        ytest = y[idx_test]  # 534,1

        T, F = Ptrain[0].shape
        D = 1

        Ptrain_tensor = Ptrain
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        mf, stdf = getStats(Ptrain)
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_time_tensor, ytrain_tensor = tensorize_normalize_other(Ptrain, ytrain,
                                                                                                           mf, stdf)
        Pval_tensor, Pval_static_tensor, Pval_time_tensor, yval_tensor = tensorize_normalize_other(Pval, yval, mf, stdf)
        Ptest_tensor, Ptest_static_tensor, Ptest_time_tensor, ytest_tensor = tensorize_normalize_other(Ptest, ytest, mf,
                                                                                                       stdf)

        train_data_combined = TensorDataset(
            Ptrain_tensor, ytrain_tensor.long())
        val_data_combined = TensorDataset(
            Pval_tensor, yval_tensor.long())
        test_data_combined = TensorDataset(
            Ptest_tensor, ytest_tensor.long())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=args.batch_size, shuffle=False)
        val_dataloader = DataLoader(
            val_data_combined, batch_size=args.batch_size, shuffle=False)

        data_object = {"train_dataloader": train_dataloader,
                        "test_dataloader": test_dataloader,
                        "val_dataloader": val_dataloader,
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
        data_objects.append(data_object)
    return data_objects

def get_physionet_data(args):
    total_dataset = PhysioNet('data/physionet', train=True,
                                  download=True, n_samples=8000
                                  )
    # total_dataset = train_dataset_obj[:len(train_dataset_obj)]
    print(len(total_dataset))

    # 分割数据集
    k_fold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    idx_list = [(train_idx, test_idx) for train_idx, test_idx in
                k_fold.split(np.zeros(len(total_dataset)), np.zeros(len(total_dataset)))]

    data_objects = []
    max_len = 215

    for i in range(len(idx_list)):
        idx_train, idx_test = idx_list[i]
        idx_val = idx_test[: len(idx_test) // 2]
        idx_test = idx_test[len(idx_test) // 2:]

        train_data = []
        test_data = []
        val_data = []
        for i in range(len(idx_train)):
            train_data.append(total_dataset[idx_train[i]])
        for i in range(len(idx_test)):
            test_data.append(total_dataset[idx_test[i]])
        for i in range(len(idx_val)):
            val_data.append(total_dataset[idx_val[i]])

        record_id, tt, vals, mask, labels = train_data[0]

        # n_samples = len(total_dataset)
        input_dim = vals.size(-1)
        batch_size = min(len(total_dataset), args.batch_size)

        data_min, data_max = get_data_min_max(total_dataset)


        train_data_combined = variable_time_collate_fn_physionet(
            train_data, classify=True,data_min=data_min,data_max=data_max)
        test_data_combined = variable_time_collate_fn_physionet(
            test_data,classify=True,data_min=data_min,data_max=data_max)
        val_data_combined = variable_time_collate_fn_physionet(
            val_data, classify=True,data_min=data_min,data_max=data_max)

        train_data_combined = TensorDataset(
            train_data_combined[0], train_data_combined[1])
        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1])
        test_data_combined = TensorDataset(
            test_data_combined[0], test_data_combined[1])

        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

        data_object = {"train_dataloader": train_dataloader,
                        "val_dataloader": val_dataloader,
                        "test_dataloader": test_dataloader,
                        "input_dim": input_dim,
                        "n_labels": 1}  # optional

        data_objects.append(data_object)
    return data_objects

def my_evaluate_classifier(model, test_loader, dec=None, binary_classification=True, classifier=None,
                        dim=36, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    prob = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        if batch_len != 50:
            continue
        with torch.no_grad():
            out = model(test_batch)
            out_prob = torch.softmax(out, dim=1)
            label = label.unsqueeze(0).repeat_interleave(
                num_sample, 0).view(-1)
            test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        prob.append(out_prob.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    prob = np.concatenate(prob, 0)
    true = np.concatenate(true, 0)
    pred_class = pred.argmax(1)
    acc = np.mean(pred.argmax(1) == true)
    if binary_classification ==True:
        auc = metrics.roc_auc_score(
            true, prob[:, 1])
        auprc = metrics.average_precision_score(true, prob[:, 1])
        return test_loss / pred.shape[0], acc, auc, auprc
    else:
        precision = metrics.precision_score(true, pred_class, average='macro')
        recall = metrics.recall_score(true, pred_class, average='macro')
        F1 = metrics.f1_score(true, pred_class, average='macro')
        return test_loss/pred.shape[0], acc, precision, recall, F1

def GaussianProcess(total_dataset):
    # 生成高斯过程回归模型
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))  # 常数核*径向基核函数
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    interp_dataset = []
    # 生成历史数据
    for b, (record_id, tt, vals, mask, labels) in enumerate(total_dataset):
        x_set = np.arange(0, 48, 0.25).reshape(-1, 1)
        y_pre = []
        y_uncertainty = []
        data_b = []
        for i in range(vals.size(-1)):

            xobs = np.array(tt[mask[:, i] == 1]).reshape(-1, 1)

            yobs = np.array(vals[:, i][mask[:, i] == 1])

            if xobs.size == 0:
                y_pre.append(torch.zeros([x_set.size]))
                y_uncertainty.append(torch.zeros([x_set.size]))
            else:
                # 使用历史数据拟合模型
                gp.fit(xobs, yobs)

                # 预测
                means, sigmas = gp.predict(x_set, return_std=True)
                y_pre.append(torch.tensor(means))
                v_n = torch.tensor(1.96 * np.sqrt(sigmas))
                y_uncertainty.append((torch.max(v_n) - v_n) / (torch.max(v_n) - (torch.min(v_n))))
            # print(means, sigmas)
            print('已插补完成{}的第{}个特征'.format(b, i))

        tuple = (record_id, torch.tensor(torch.tensor([item.cpu().detach().numpy() for item in y_pre]).T),
                 torch.tensor(torch.tensor([item.cpu().detach().numpy() for item in y_uncertainty]).T), labels)
        interp_dataset.append(tuple)

    return interp_dataset




def get_physionet_interp_data(args, device, q, flag=1):
    inter_np = np.load("interp_data.npy", allow_pickle=True)
    print(inter_np)


    input_dim = 41
    batch_size = 32
    interp_dataset = inter_np.tolist()

    data_min, data_max = my_get_data_min_max(interp_dataset, device)

    interp_train_data,interp_test_data = model_selection.train_test_split(interp_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)



    # test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
    #                                               data_min=data_min, data_max=data_max)
    test_data_combined = my_variable_time_collate_fn(interp_test_data,device,classify=True,data_min=data_min,data_max=data_max)

    train_data, val_data = model_selection.train_test_split(interp_train_data, train_size=0.8,
                                                            random_state=11, shuffle=True)
    # train_data, val_data = model_selection.train_test_split(interp_train_data, train_size=0.8,
    #                                                         random_state=11, shuffle=True)
    train_data_combined = my_variable_time_collate_fn(
        train_data, device, classify=True,data_min=data_min,data_max=data_max)
    val_data_combined = my_variable_time_collate_fn(
        val_data, device, classify=True,data_min=data_min,data_max=data_max)
    # print(train_data_combined[1].sum(
    # ), val_data_combined[1].sum(), test_data_combined[1].sum())
    # print(train_data_combined[0].size(), train_data_combined[1].size(),
    #       val_data_combined[0].size(), val_data_combined[1].size(),
    #       test_data_combined[0].size(), test_data_combined[1].size())

    train_data_combined = TensorDataset(
        train_data_combined[0], train_data_combined[1])
    val_data_combined = TensorDataset(
        val_data_combined[0], val_data_combined[1])
    test_data_combined = TensorDataset(
        test_data_combined[0], test_data_combined[1])


    train_dataloader = DataLoader(
        train_data_combined, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(
        test_data_combined, batch_size=batch_size, shuffle=False)


    # attr_names = train_dataset_obj.params
    data_objects = {"train_dataloader": train_dataloader,
                    "test_dataloader": test_dataloader,
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    # "attr": attr_names,  # optional
                    "classif_per_tp": False,  # optional
                    "n_labels": 1}  # optional
    val_dataloader = DataLoader(
        val_data_combined, batch_size=batch_size, shuffle=False)
    data_objects["val_dataloader"] = val_dataloader
    return data_objects

def my_get_data_min_max(records, device):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  data_min, data_max = None, None
  inf = torch.Tensor([float("Inf")])[0].to(device)

  for b, (record_id, vals, uncertainty, labels) in enumerate(records):
    n_features = vals.size(-1)

    batch_min = []
    batch_max = []
    for i in range(n_features):
      non_missing_vals = vals[:,i]
      if len(non_missing_vals) == 0:
        batch_min.append(inf)
        batch_max.append(-inf)
      else:
        batch_min.append(torch.min(non_missing_vals))
        batch_max.append(torch.max(non_missing_vals))

    batch_min = torch.stack(batch_min)
    batch_max = torch.stack(batch_max)

    if (data_min is None) and (data_max is None):
      data_min = batch_min
      data_max = batch_max
    else:
      data_min = torch.min(data_min, batch_min)
      data_max = torch.max(data_max, batch_max)

  return data_min.to(device), data_max.to(device)

def my_variable_time_collate_fn(batch, device=torch.device('cuda:0'), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][1].shape[1]
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    # enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_uncertainty = torch.zeros([len(batch), maxlen, D]).to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)

    for b, (record_id, vals, uncertainty, labels) in enumerate(batch):
        currlen = 48
        # enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_uncertainty[b, :currlen] = uncertainty.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_data(enc_combined_vals.to(device), data_min, data_max)

    # if torch.max(inter_timestamp) != 0.:
    #     enc_combined_tt = inter_timestamp / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals.float(), enc_combined_uncertainty), 2)
    if classify:
        return combined_data, combined_labels.squeeze(-1)
    else:
        return combined_data

def normalize_data(data,att_min,att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")


    return data_norm, att_min, att_max

def get_set(tensor):
    input_dim = int(tensor.size(-1)/2)
    observersions = tensor[:, :, :input_dim]
    masks = tensor[:, :, input_dim:2*input_dim]
    times = tensor[:, :, -1]

    time_set = []
    value_set = []
    modality_set = []

    for i in range(tensor[:1000].size(0)):
        position = torch.where(masks[i])

        time_i = []
        m_i = []
        z_i = []
        for j in range(0, len(position[0])):
            X_index = position[0][j]
            Y_index = position[1][j]
            time_i.append(times[i][X_index])
            m_i.append(Y_index)
            z_i.append(observersions[i][X_index][Y_index])
        time_set.append(torch.tensor(time_i))
        modality_set.append(torch.tensor(m_i))
        value_set.append(torch.tensor(z_i))
    return time_set, modality_set, value_set

def getStats(P_tensor):  # 返回每个变量的均值和方差
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if vals_f.size == 0:
            mf[f] = 0.0
        else:
            mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.max([stdf[f], eps])
    return mf, stdf

def getStats_static(P_tensor, dataset='P12'):  # 返回静态变量的均值和方差
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12' or dataset == 'physionet':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == 'eICU':
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical[s] == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss

def tensorize_normalize(P, y, mf, stdf, ms, ss, interp):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_static_tensor[i] = P[i]['extended_static']
        P_length[i] = P[i]['length']
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        if P[i]['length'] < T:
            P_delta_t[i][P[i]['length']] = 0
    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_tensor = torch.cat((P_tensor, P_time), dim=2)

    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor

def tensorize_normalize_density(P, y, mf, stdf, ms, ss, interp):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    P_density_tensor = np.zeros((len(P), F))

    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        for j in range(F):
            idx_not_zero = np.where(P_tensor[i][:, j])
            if len(idx_not_zero[0]) == 0:
                P_density_tensor[i][j] = 0
            elif len(idx_not_zero[0]) == 1:
                P_density_tensor[i][j] = 1
            else:
                P_density_tensor[i][j] = len(idx_not_zero[0]) / (idx_not_zero[0][-1] - idx_not_zero[0][0] + 1)
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_static_tensor[i] = P[i]['extended_static']
        P_length[i] = P[i]['length']
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        if P[i]['length'] < T:
            P_delta_t[i][P[i]['length']] = 0
    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_tensor = torch.cat((P_tensor, P_time), dim=2)

    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, torch.FloatTensor(P_density_tensor), P_static_tensor, \
        torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor

def tensorize_normalize_varlength(P, y, mf, stdf, ms, ss, interp):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    P_var_length_tensor = np.zeros((len(P), F))
    P_var_last_obs_tp_tensor = np.zeros((len(P), F))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        for j in range(F):
            idx_not_zero = np.where(P_tensor[i][:, j])
            if len(idx_not_zero[0]) == 0:
                P_var_length_tensor[i][j] = 0
                P_var_last_obs_tp_tensor[i][j] = 0
            elif len(idx_not_zero[0]) == 1:
                P_var_length_tensor[i][j] = 1
                P_var_last_obs_tp_tensor[i][j] = idx_not_zero[0][0]
            else:
                P_var_length_tensor[i][j] = len(idx_not_zero[0])
                P_var_last_obs_tp_tensor[i][j] = idx_not_zero[0][-1]

        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_static_tensor[i] = P[i]['extended_static']
        P_length[i] = P[i]['length']
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        if P[i]['length'] < T:
            P_delta_t[i][P[i]['length']] = 0
    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_tensor = torch.cat((P_tensor, P_time), dim=2)

    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, torch.FloatTensor(P_var_length_tensor), torch.FloatTensor(P_var_last_obs_tp_tensor), \
        P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
#def tensorize_normalize_with_nufft(P, y, mf, stdf, ms, ss, interp):
#    T, F = P[0]['arr'].shape
#    D = len(P[0]['extended_static'])
#
#    P_tensor = np.zeros((len(P), T, F))
#    P_fft_tensor = np.zeros((len(P), 3, T, F))
#    P_time = np.zeros((len(P), T, 1))
#    P_delta_t = np.zeros((len(P), T, 1))
#    P_length = np.zeros([len(P), 1])
#    P_static_tensor = np.zeros((len(P), D))
#    for i in range(len(P)):
#        P_tensor[i] = P[i]['arr']
#        P_time[i] = P[i]['time']  / 60.0
#        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
#        P_static_tensor[i] = P[i]['extended_static']
#        P_length[i] = P[i]['length']
#        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
#        if P[i]['length'] < T:
#            P_delta_t[i][P[i]['length']] = 0
#    P_tensor = mask_normalize(P_tensor, mf, stdf)
#    for i in range(len(P)):
#        for j in range(F):
#            idx_not_zero = np.where(P_tensor[i][:, j])
#            if len(idx_not_zero[0]) != 0:
#                x = P_tensor[i][:, j][idx_not_zero]
#                t = P_time[i][idx_not_zero]
#                nufft_complex, f = nufft(x, t)
#                P_fft_tensor[i, 0, :len(idx_not_zero[0]), j] = f
#                P_fft_tensor[i, 1, :len(idx_not_zero[0]), j] = \
#                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
#                P_fft_tensor[i, 2, :len(idx_not_zero[0]), j] = \
#                    np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
#    P_tensor = torch.Tensor(P_tensor)
#
#    P_time = torch.Tensor(P_time)  # convert mins to hours
#    P_tensor = torch.cat((P_tensor, P_time), dim=2)
#
#    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
#    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
#    P_static_tensor = torch.Tensor(P_static_tensor)
#    y_tensor = y
#    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
#    return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
#        torch.tensor(P_length), P_time, y_tensor
def tensorize_normalize_with_nufft(P, y, mf, stdf, ms, ss, interp):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_fft_tensor = np.zeros((len(P), 3, T, F))
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']  / 60.0
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_static_tensor[i] = P[i]['extended_static']
        P_length[i] = P[i]['length']
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        if P[i]['length'] < T:
            P_delta_t[i][P[i]['length']] = 0
    P_tensor = mask_normalize(P_tensor, mf, stdf)
    for i in range(len(P)):
        for j in range(F):
            idx_not_zero = np.where(P_tensor[i][:, j])
            if len(idx_not_zero[0]) > 1:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P_time[i][idx_not_zero]
                nufft_complex, f = nufft(x, t)
                interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                # P_fft_tensor[i, 0, :len(idx_not_zero[0]), j] = f
                P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                # P_fft_tensor[i, 1, :len(idx_not_zero[0]), j] = \
                #     np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                # P_fft_tensor[i, 2, :len(idx_not_zero[0]), j] = \
                #     np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))
                P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                    np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time)  # convert mins to hours
    P_tensor = torch.cat((P_tensor, P_time), dim=2)

    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
        torch.FloatTensor(P_length), P_time, y_tensor
def tensorize_normalize_mimic3(P, y, mf, stdf):
    # [_, 48, 200] [_] [12, 1] [12, 1]
    F, T = P[0].shape[0] // 4 , P[0].shape[1]
    P_tensor = P[:, :F, :].transpose(0, 2, 1)
    P_time = np.expand_dims(P[:, 24, :], axis=-1)
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    mask = P[:, F:2 * F, :].transpose(0, 2, 1)
    count = 0
    for i in range(len(P)):
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_length[i] = length_i = np.max(np.where(mask[i] != 0)[0]) + 1
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        if P_length[i] < T:
            P_delta_t[i][length_i] = 0

    P_tensor = mask_normalize(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_tensor = torch.cat((P_tensor, P_time), dim=2)

    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    # P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    # P_static_tensor = torch.Tensor(P_static_tensor)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor).type(torch.LongTensor)
    return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor


def tensorize_normalize_IP_Net(P, y, mf, stdf, ms, ss, interp):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_static_tensor = np.zeros((len(P), D))
    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_static_tensor[i] = P[i]['extended_static']
        P_length[i] = P[i]['length']
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        if P[i]['length'] < T:
            P_delta_t[i][P[i]['length']] = 0
    P_tensor = mask_normalize_IP_Net(P_tensor, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_tensor = torch.cat((P_tensor, P_time), dim=2)

    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor

def tensorize_normalize_misssing(P, y, mf, stdf, ms, ss, missingtype, missingratio, idx = None):
    origin_T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            for i in range(len(P)):
                idx = np.sort(
                    np.random.choice(P[i]['length'], math.ceil(P[i]['length'] * (1 - missingratio)), replace=False))
                length = idx.size
                P_tensor[i][:length] = P[i]['arr'][idx, :]
                P_time[i][:length] = P[i]['time'][idx, :]
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = length
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if length < T:
                    P_delta_t[i][length] = 0
            P_tensor = mask_normalize(P_tensor, mf, stdf)
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                P_length), P_time, y_tensor
        elif missingtype == 'variable':
                T = origin_T
                F = round((1 - missingratio) * F)
                P_tensor = np.zeros((len(P), T, F))
                P_time = np.zeros((len(P), T, 1))
                P_delta_t = np.zeros((len(P), T, 1))
                P_length = np.zeros([len(P), 1])
                P_static_tensor = np.zeros((len(P), D))
                for i in range(len(P)):
                    P_tensor[i] = P[i]['arr'][:, idx]
                    P_time[i] = P[i]['time']
                    P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                    P_static_tensor[i] = P[i]['extended_static']
                    P_length[i] = P[i]['length']
                    P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                    if P[i]['length'] < T:
                        P_delta_t[i][P[i]['length']] = 0
                P_tensor = mask_normalize(P_tensor, mf[idx], stdf[idx])
                P_tensor = torch.Tensor(P_tensor)

                P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
                P_tensor = torch.cat((P_tensor, P_time), dim=2)

                P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
                P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
                P_static_tensor = torch.Tensor(P_static_tensor)
                y_tensor = y
                y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
                return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                    P_length), P_time, y_tensor
    else:
        return tensorize_normalize(P, y, mf, stdf, ms, ss, None)

def tensorize_normalize_other_with_nufft(P, y, mf, stdf):
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])
    P_fft_tensor = np.zeros((len(P), 3, T, F))

    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        P_length[i] = 600
    P_tensor = mask_normalize(P, mf, stdf)
    for i in range(len(P)):
        for j in range(F):
            idx_not_zero = np.where(P_tensor[i][:, j])
            if len(idx_not_zero[0]) > 1:
                x = P_tensor[i][:, j][idx_not_zero]
                t = P_time[i][idx_not_zero]
                nufft_complex= np.fft.fft(x)
                interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
#                P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
#                P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
#                    np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0
    P_tensor = torch.cat((P_tensor, P_time), dim=2)
#    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
    P_delta_t_tensor = P_delta_t.squeeze(-1)
    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor

def tensorize_normalize_with_nufft_misssing(P, y, mf, stdf, ms, ss, missingtype, missingratio, idx = None):
    origin_T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])
    if missingratio <= 0:
        return tensorize_normalize_with_nufft(P, y, mf, stdf, ms, ss, None)
    else:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                idx = np.sort(
                    np.random.choice(P[i]['length'], math.ceil(P[i]['length'] * (1 - missingratio)), replace=False))
                length = idx.size
                P_tensor[i][:length] = P[i]['arr'][idx, :]
                P_time[i][:length] = P[i]['time'][idx, :] / 60
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = length
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if length < T:
                    P_delta_t[i][length] = 0

            P_tensor = mask_normalize(P_tensor, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
                torch.FloatTensor(P_length), P_time, y_tensor
        else:
            T = origin_T
            P_tensor = np.zeros((len(P), T, F))
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_static_tensor = np.zeros((len(P), D))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                P_tensor[i][:, idx] = P[i]['arr'][:, idx]
                P_time[i] = P[i]['time']
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_static_tensor[i] = P[i]['extended_static']
                P_length[i] = P[i]['length']
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                if P[i]['length'] < T:
                    P_delta_t[i][P[i]['length']] = 0
            P_tensor = mask_normalize(P_tensor, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
            P_tensor = torch.cat((P_tensor, P_time), dim=2)

            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
            P_static_tensor = torch.Tensor(P_static_tensor)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), P_static_tensor, torch.FloatTensor(P_delta_t_tensor), \
                torch.FloatTensor(P_length), P_time, y_tensor

def tensorize_normalize_other_missing(P, y, mf, stdf, missingtype, missingratio, idx=None):
    origin_T, F = P[0].shape
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_new = np.zeros((len(P), T, F))
            for i in range(len(P)):
                idx = np.sort(np.random.choice(origin_T, round(T), replace=False))
                P_new[i] = P[i, idx, :]
                tim = torch.linspace(0, origin_T, origin_T).reshape(-1, 1)[idx]
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1))
                P_length[i] = T
            P_tensor = mask_normalize(P_new, mf, stdf)
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
        elif missingtype == 'variable':
            F = round((1 - missingratio) * F)
            T = origin_T
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])

            for i in range(len(P)):
                tim = torch.linspace(0, T, T).reshape(-1, 1)
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                P_length[i] = 600
            P_tensor = mask_normalize(P[:, :, idx], mf[idx], stdf[idx])
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
    else:
        return tensorize_normalize_other(P, y, mf, stdf)
def tensorize_normalize_misssing_ode(P, y, mf, stdf, ms, ss, missingtype, missingratio, idx = None, combined_tt = None):
    origin_T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            D = P[0]['arr'].shape[1]
            left_time_list = []
            idx_list = []
            left_time = np.zeros((len(P), T))
            for i in range(len(P)):
                idx = np.sort(
                    np.random.choice(P[i]['length'], math.ceil(P[i]['length'] * (1 - missingratio)), replace=False))
                # time_i[:len(idx)] = P[i]['time'][idx]
                left_time[i][:len(idx)] = P[i]['time'][idx][:, 0]
                left_time_list.append(P[i]['time'][idx][:, 0])
                idx_list.append(idx)
            combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex) for ex in left_time_list]), sorted=True,
                                                    return_inverse=True)
            offset = 0
            combined_vals = torch.zeros([len(P), len(combined_tt), D])
            combined_mask = torch.zeros([len(P), len(combined_tt), D])
            combined_labels = torch.squeeze(torch.FloatTensor(y))

            for i in range(len(P)):
                tt = left_time_list[i]
                vals = P[i]['arr'][idx_list[i]]
                mask = np.zeros_like(vals)
                mask[np.where(vals != 0)] = 1
                indices = inverse_indices[offset:offset + len(tt)]
                offset += len(tt)
                combined_vals[i, indices] = torch.FloatTensor(vals)
                combined_mask[i, indices] = torch.FloatTensor(mask)
            Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
            return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels
        elif missingtype == 'variable':
                T = origin_T
                F = round((1 - missingratio) * F)
                P_tensor = np.zeros((len(P), T, F))
                P_time = np.zeros((len(P), T, 1))
                P_delta_t = np.zeros((len(P), T, 1))
                P_length = np.zeros([len(P), 1])
                P_static_tensor = np.zeros((len(P), D))
                for i in range(len(P)):
                    P_tensor[i] = P[i]['arr'][:, idx]
                    P_time[i] = P[i]['time']
                    P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                    P_static_tensor[i] = P[i]['extended_static']
                    P_length[i] = P[i]['length']
                    P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                    if P[i]['length'] < T:
                        P_delta_t[i][P[i]['length']] = 0
                P_tensor = mask_normalize(P_tensor, mf[idx], stdf[idx])
                P_tensor = torch.Tensor(P_tensor)

                P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
                P_tensor = torch.cat((P_tensor, P_time), dim=2)

                P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
                P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
                P_static_tensor = torch.Tensor(P_static_tensor)
                y_tensor = y
                y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
                return P_tensor, P_static_tensor, torch.FloatTensor(P_delta_t_tensor), torch.tensor(
                    P_length), P_time, y_tensor
    else:
        D = P[0]['arr'].shape[1]
        combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex['time']) for ex in P]), sorted=True,
                                                    return_inverse=True)
        offset = 0
        combined_vals = torch.zeros([len(P), len(combined_tt), D])
        combined_mask = torch.zeros([len(P), len(combined_tt), D])

        combined_labels = torch.squeeze(torch.FloatTensor(y))

        for i in range(len(P)):
            tt = P[i]['time']
            vals = P[i]['arr']
            mask = np.zeros_like(vals)
            mask[np.where(vals != 0)] = 1
            indices = inverse_indices[offset:offset + len(tt)]
            offset += len(tt)

            combined_vals[i, indices] = torch.FloatTensor(vals).unsqueeze(1)
            combined_mask[i, indices] = torch.FloatTensor(mask).unsqueeze(1)
        Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
        return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels

def tensorize_normalize_other_missing_ode(P, y, mf, stdf, missingtype, missingratio, idx=None):
    origin_T, F = P[0].shape
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            D = F
            left_time_list = []
            idx_list = []
            left_time = np.zeros((len(P), T))
            time = np.arange(0, origin_T)
            for i in range(len(P)):
                idx = np.sort(np.random.choice(origin_T, round(T), replace=False))
                # time_i[:len(idx)] = P[i]['time'][idx]
                left_time[i][:len(idx)] = time[idx][:]
                left_time_list.append(time[idx][:])
                idx_list.append(idx)
            combined_tt, inverse_indices = torch.unique(torch.cat([torch.FloatTensor(ex) for ex in left_time_list]),
                                                        sorted=True,
                                                        return_inverse=True)
            offset = 0
            combined_vals = torch.zeros([len(P), len(combined_tt), D])
            combined_mask = torch.zeros([len(P), len(combined_tt), D])
            combined_labels = torch.squeeze(torch.FloatTensor(y))

            for i in range(len(P)):
                tt = left_time_list[i]
                vals = P[i][idx_list[i]]
                mask = np.zeros_like(vals)
                mask[np.where(vals != 0)] = 1
                indices = inverse_indices[offset:offset + len(tt)]
                offset += len(tt)
                combined_vals[i, indices] = torch.FloatTensor(vals)
                combined_mask[i, indices] = torch.FloatTensor(mask)
            Ptensor = mask_normalize(combined_vals.numpy(), mf, stdf)
            return torch.FloatTensor(Ptensor), combined_tt / torch.max(combined_tt), combined_labels
        elif missingtype == 'variable':
            F = round((1 - missingratio) * F)
            T = origin_T
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])

            for i in range(len(P)):
                tim = torch.linspace(0, T, T).reshape(-1, 1)
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                P_length[i] = 600
            P_tensor = mask_normalize(P[:, :, idx], mf[idx], stdf[idx])
            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor
    else:
        Ptensor = mask_normalize(P, mf, stdf)
        combined_tt = torch.arange(0, origin_T)
        combined_labels = torch.squeeze(torch.FloatTensor(y))
        return torch.FloatTensor(Ptensor), combined_tt / origin_T, combined_labels
def tensorize_normalize_other_missing_with_nufft(P, y, mf, stdf, missingtype, missingratio, idx=None):
    origin_T, F = P[0].shape
    if missingratio > 0:
        if missingtype == 'time':
            T = int((1 - missingratio) * origin_T)
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_new = np.zeros((len(P), T, F))
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                idx = np.sort(np.random.choice(origin_T, round(T), replace=False))
                P_new[i] = P[i, idx, :]
                tim = torch.linspace(0, origin_T, origin_T).reshape(-1, 1)[idx]
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1))
                P_length[i] = T
            P_tensor = mask_normalize(P_new, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
#                        nufft_complex, f = nufft(x, t)
                        nufft_complex= np.fft.fft(x)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        # P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        # P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                        #     np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
#            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))
            P_delta_t_tensor = P_delta_t.squeeze(-1)
            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor
        elif missingtype == 'variable':
            T = origin_T
            P_time = np.zeros((len(P), T, 1))
            P_delta_t = np.zeros((len(P), T, 1))
            P_length = np.zeros([len(P), 1])
            P_fft_tensor = np.zeros((len(P), 3, T, F))
            for i in range(len(P)):
                tim = torch.linspace(0, T, T).reshape(-1, 1)
                P_time[i] = tim
                P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
                P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
                P_length[i] = 600
            del_idx = [j for j in np.arange(F) if j not in idx]
            P[:, :, del_idx] = 0
            P_tensor= mask_normalize(P, mf, stdf)
            for i in range(len(P)):
                for j in range(F):
                    idx_not_zero = np.where(P_tensor[i][:, j])
                    if len(idx_not_zero[0]) > 1:
                        x = P_tensor[i][:, j][idx_not_zero]
                        t = P_time[i][idx_not_zero]
                        nufft_complex, f = nufft(x, t)
#                        nufft_complex= np.fft.fft(x)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        # P_fft_tensor[i][0][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = f
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)
                        # P_fft_tensor[i][2][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                        #     np.arctan(nufft_complex.imag / (nufft_complex.real + 1e-8))


            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time) / 60.0
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
            P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

            y_tensor = y
            y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
            return P_tensor, torch.FloatTensor(P_fft_tensor), None, torch.FloatTensor(P_delta_t_tensor), torch.FloatTensor(P_length), P_time, y_tensor
    else:
        return tensorize_normalize_other_with_nufft(P, y, mf, stdf)
def mask_normalize_delta(P_delta_tensor):
    # input normalization
    # set missing values to zero after normalization
    idx_missing = np.where(P_delta_tensor == 0)
    idx_existing = np.where(P_delta_tensor != 0)
    max = np.max(P_delta_tensor[idx_existing])
    min = np.min(P_delta_tensor[idx_existing])
    if min == max:
        return P_delta_tensor
    P_delta_tensor = (P_delta_tensor - min) / ((max - min) + 1e-18)
    P_delta_tensor[idx_missing] = 0
    return P_delta_tensor

def get_data_split(base_path='./data/P12data', split_path='', split_type='random', reverse=False, baseline=True, dataset='P12', predictive_label='mortality'):
    # load data
    if dataset == 'P12' or dataset == 'physionet':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''
    elif dataset == 'P19':
        Pdict_list = np.load(base_path + '/processed_data/PT_dict_list_6.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes_6.npy', allow_pickle=True)
        dataset_prefix = 'P19_'
    elif dataset == 'eICU':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = 'eICU_'
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

    show_statistics = False
    if show_statistics:
        idx_under_65 = []
        idx_over_65 = []

        idx_male = []
        idx_female = []

        # variables for statistics
        all_ages = []
        female_count = 0
        male_count = 0
        all_BMI = []

        X_static = np.zeros((len(Pdict_list), len(Pdict_list[0]['extended_static'])))
        for i in range(len(Pdict_list)):
            X_static[i] = Pdict_list[i]['extended_static']
            age, gender_0, gender_1, height, _, _, _, _, weight = X_static[i]
            if age > 0:
                all_ages.append(age)
                if age < 65:
                    idx_under_65.append(i)
                else:
                    idx_over_65.append(i)
            if gender_0 == 1:
                female_count += 1
                idx_female.append(i)
            if gender_1 == 1:
                male_count += 1
                idx_male.append(i)
            if height > 0 and weight > 0:
                all_BMI.append(weight / ((height / 100) ** 2))

        # # plot statistics
        # plt.hist(all_ages, bins=[i * 10 for i in range(12)])
        # plt.xlabel('Years')
        # plt.ylabel('# people')
        # plt.title('Histogram of patients ages, age known in %d samples.\nMean: %.1f, Std: %.1f, Median: %.1f' %
        #           (len(all_ages), np.mean(np.array(all_ages)), np.std(np.array(all_ages)), np.median(np.array(all_ages))))
        # plt.show()
        #
        # plt.hist(all_BMI, bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        # all_BMI = np.array(all_BMI)
        # all_BMI = all_BMI[(all_BMI > 10) & (all_BMI < 65)]
        # plt.xlabel('BMI')
        # plt.ylabel('# people')
        # plt.title('Histogram of patients BMI, height and weight known in %d samples.\nMean: %.1f, Std: %.1f, Median: %.1f' %
        #           (len(all_BMI), np.mean(all_BMI), np.std(all_BMI), np.median(all_BMI)))
        # plt.show()
        # print('\nGender known: %d,  Male count: %d,  Female count: %d\n' % (male_count + female_count, male_count, female_count))

    # np.save('saved/idx_under_65.npy', np.array(idx_under_65), allow_pickle=True)
    # np.save('saved/idx_over_65.npy', np.array(idx_over_65), allow_pickle=True)
    # np.save('saved/idx_male.npy', np.array(idx_male), allow_pickle=True)
    # np.save('saved/idx_female.npy', np.array(idx_female), allow_pickle=True)

    if baseline==True:
        BL_path = ''
    else:
        BL_path = 'baselines/'

    if split_type == 'random':
        # load random indices from a split
        idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)
    elif split_type == 'age':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_over_65.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_under_65.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]
    elif split_type == 'gender':
        if reverse == False:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
        elif reverse == True:
            idx_train = np.load(BL_path+'saved/' + dataset_prefix + 'idx_female.npy', allow_pickle=True)
            idx_vt = np.load(BL_path+'saved/' + dataset_prefix + 'idx_male.npy', allow_pickle=True)

        np.random.shuffle(idx_vt)
        idx_val = idx_vt[:round(len(idx_vt) / 2)]
        idx_test = idx_vt[round(len(idx_vt) / 2):]

    # length = {
    #           '1-20': 0, '21-40': 0, '41-60': 0,
    #           # '61-80': 0, '81-100': 0, '101-120': 0, '121-140': 0, '141-160': 0, '161-180': 0,
    #           # '181-200': 0, '201-220': 0
    #           }
    # for i in range(len(Pdict_list)):
    #     if Pdict_list[i]['length'] >= 1 and Pdict_list[i]['length'] <= 20:
    #         length['1-20'] = length['1-20'] + 1
    #     elif Pdict_list[i]['length'] >= 21 and Pdict_list[i]['length'] <= 40:
    #         length['21-40'] = length['21-40'] + 1
    #     elif Pdict_list[i]['length'] >= 41 and Pdict_list[i]['length'] <= 60:
    #         length['41-60'] = length['41-60'] + 1
        # elif Pdict_list[i]['length'] >= 61 and Pdict_list[i]['length'] <= 80:
        #     length['61-80'] = length['61-80'] + 1
        # elif Pdict_list[i]['length'] >= 81 and Pdict_list[i]['length'] <= 100:
        #     length['81-100'] = length['81-100'] + 1
        # elif Pdict_list[i]['length'] >= 101 and Pdict_list[i]['length'] <= 120:
        #     length['101-120'] = length['101-120'] + 1
        # elif Pdict_list[i]['length'] >= 121 and Pdict_list[i]['length'] <= 140:
        #     length['121-140'] = length['121-140'] + 1
        # elif Pdict_list[i]['length'] >= 141 and Pdict_list[i]['length'] <= 160:
        #     length['141-160'] = length['141-160'] + 1
        # elif Pdict_list[i]['length'] >= 161 and Pdict_list[i]['length'] <= 180:
        #     length['161-180'] = length['161-180'] + 1
        # elif Pdict_list[i]['length'] >= 181 and Pdict_list[i]['length'] <= 200:
        #     length['181-200'] = length['181-200'] + 1
        # elif Pdict_list[i]['length'] >= 201 and Pdict_list[i]['length'] <= 220:
        #     length['201-220'] = length['201-220'] + 1

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20, 10))
    # plt.rcParams["font.sans-serif"] = ['Times New Roman']
    # plt.rcParams["axes.unicode_minus"] = False
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=30)
    # plt.bar(list(length.keys()), list(length.values()))
    # for a, b in zip(list(length.keys()), list(length.values())):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    # plt.title("Sample observation length distribution of P19(maxlen=60, total_num=38,803)", fontsize=30)
    # plt.xlabel("Observation length interval", fontsize=30)
    # plt.ylabel("Number of samples", fontsize=30)
    #
    # plt.show()

    # extract train/val/test examples
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]

    if dataset == 'P12' or dataset == 'P19' or dataset == 'PAM' or dataset == 'physionet':
        if predictive_label == 'mortality':
            y = arr_outcomes[:, -1].reshape((-1, 1))
        elif predictive_label == 'LoS':  # for P12 only
            y = arr_outcomes[:, 3].reshape((-1, 1))
            y = np.array(list(map(lambda los: 0 if los <= 3 else 1, y)))[..., np.newaxis]
    elif dataset == 'eICU':
        y = arr_outcomes[..., np.newaxis]
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest


def evaluate(model, P_tensor, P_time_tensor, P_static_tensor, P_delta_t_tensor, P_length_tensor, batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()
        N, Fs = P_static_tensor.shape

    T, N, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        P_delta_t = P_delta_t_tensor[start:start + batch_size, :]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + batch_size])
        middleoutput = model.forward(P.permute(1, 0, 2), P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Ptime = P_time_tensor[:, start:start + rem]
        P_delta_t = P_delta_t_tensor[start:start + rem, :].cuda()
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + rem])
        whatever = model.forward(P.permute(1, 0, 2), P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + rem] = whatever.detach().cpu()
    return out


def evaluate_nufft(model, P_tensor, P_fft_tensor, P_time_tensor, P_static_tensor, P_delta_t_tensor, P_length_tensor, batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_fft_tensor = P_fft_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()
        N, Fs = P_static_tensor.shape

    T, N, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        P_fft = P_fft_tensor[start: start + batch_size]
        Ptime = P_time_tensor[:, start:start + batch_size]
        P_delta_t = P_delta_t_tensor[start:start + batch_size, :]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + batch_size])
        middleoutput = model.forward(P.permute(1, 0, 2), P_fft, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        P_fft = P_fft_tensor[start:start + rem]
        Ptime = P_time_tensor[:, start:start + rem]
        P_delta_t = P_delta_t_tensor[start:start + rem, :].cuda()
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + rem])
        whatever = model.forward(P.permute(1, 0, 2), P_fft, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + rem] = whatever.detach().cpu()
    return out
def evaluate_var_length(model, P_tensor, Pvar_length_tensor, Pvar_last_obs_tp_tensor,
                        P_time_tensor, P_static_tensor, P_delta_t_tensor, P_length_tensor,
                        batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    Pvar_length_tensor = Pvar_length_tensor.cuda()
    Pvar_last_obs_tp_tensor = Pvar_last_obs_tp_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()
        N, Fs = P_static_tensor.shape

    T, N, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        Pvar_length = Pvar_length_tensor[start:start + batch_size]
        Pvar_last_obs_tp = Pvar_last_obs_tp_tensor[start:start + batch_size]
        Ptime = P_time_tensor[:, start:start + batch_size]
        P_delta_t = P_delta_t_tensor[start:start + batch_size, :]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + batch_size])
        middleoutput = model.forward(P.permute(1, 0, 2), Pvar_length, Pvar_last_obs_tp, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Pvar_length = Pvar_length_tensor[start:start + rem]
        Pvar_last_obs_tp = Pvar_last_obs_tp_tensor[start:start + rem]
        Ptime = P_time_tensor[:, start:start + rem]
        P_delta_t = P_delta_t_tensor[start:start + rem, :].cuda()
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + rem])
        whatever = model.forward(P.permute(1, 0, 2), Pvar_length, Pvar_last_obs_tp, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + rem] = whatever.detach().cpu()
    return out
def evaluate_density(model, P_tensor, P_density_tensor, P_time_tensor, P_static_tensor, P_delta_t_tensor, P_length_tensor, batch_size=100, n_classes=2, static=None):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_density_tensor = P_density_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    P_length_tensor = P_length_tensor.cuda()
    if static is None:
        Pstatic = None
    else:
        P_static_tensor = P_static_tensor.cuda()
        N, Fs = P_static_tensor.shape

    T, N, Ff = P_tensor.shape
    n_batches, rem = N // batch_size, N % batch_size
    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        P_density = P_density_tensor[start: start + batch_size]
        Ptime = P_time_tensor[:, start:start + batch_size]
        P_delta_t = P_delta_t_tensor[start:start + batch_size, :]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + batch_size])
        middleoutput = model.forward(P.permute(1, 0, 2), P_density, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + batch_size] = middleoutput.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        P_density = P_density_tensor[start:start + rem]
        Ptime = P_time_tensor[:, start:start + rem]
        P_delta_t = P_delta_t_tensor[start:start + rem, :].cuda()
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size].cuda()
        # lengths = torch.sum(Ptime > 0, dim=0) + 1
        lengths = torch.squeeze(P_length_tensor[start: start + rem])
        whatever = model.forward(P.permute(1, 0, 2), P_density, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + rem] = whatever.detach().cpu()
    return out


def tensorize_normalize_other(P, y, mf, stdf):
    T, F = P[0].shape
    P_time = np.zeros((len(P), T, 1))
    P_delta_t = np.zeros((len(P), T, 1))
    P_length = np.zeros([len(P), 1])

    for i in range(len(P)):
        tim = torch.linspace(0, T, T).reshape(-1, 1)
        P_time[i] = tim
        P_time_right_shifting = np.insert(P_time[i][:-1], 0, values=P_time[i][0])
        P_delta_t[i] = (P_time[i] - np.expand_dims(P_time_right_shifting, -1)) / 60.0
        P_length[i] = 600
    P_tensor = mask_normalize(P, mf, stdf)
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0
    P_tensor = torch.cat((P_tensor, P_time), dim=2)
    P_delta_t_tensor = mask_normalize_delta(P_delta_t.squeeze(-1))

    y_tensor = y
    y_tensor = torch.Tensor(y_tensor[:, 0]).type(torch.LongTensor)
    return P_tensor, None, torch.FloatTensor(P_delta_t_tensor), torch.tensor(P_length), P_time, y_tensor

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = 1 * (P_tensor > 0) + 0 * (P_tensor <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        # Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
        # Pf[f] = (Pf[f] - np.min(Pf[f])) / np.max(Pf[f]) - np.min(Pf[f])
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor

def mask_normalize_IP_Net(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = 1 * (P_tensor > 0) + 0 * (P_tensor <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    # for f in range(F):
    #     # Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    #     Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    #     # Pf[f] = (Pf[f] - np.min(Pf[f])) / np.max(Pf[f]) - np.min(Pf[f])
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor


def normalize(P_tensor, mf, stdf):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return Pnorm_tensor

def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor



def variable_time_collate_fn_physionet(batch, device=torch.device("cuda:0"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1] - 5
    # number of labels
    N = batch[0][-1].shape[1] if activity else 1
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = 203
    enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)

    combined_labels = torch.zeros([len(batch)]).to(device)


    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals[:, 5:].to(device)
        enc_combined_mask[b, :currlen] = mask[:, 5:].to(device)
        if classify:
            if activity:
                combined_labels[b] = torch.argmax(labels[0], dim=-1).to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data

def get_activity_data(args, device):
    n_samples = 10000
    dataset_obj = PersonActivity('data/PersonActivity',
                                 download=True, n_samples=n_samples, device=device)

    # print(dataset_obj)
    np_dataset = np.array(dataset_obj.data)
    k_fold = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
    idx_list = [(train_idx, test_idx) for train_idx, test_idx in k_fold.split(np.zeros(len(dataset_obj)), np.zeros(len(dataset_obj)))]

    data_objects = []

    for i in range(len(idx_list)):
        idx_train, idx_test = idx_list[i]
        idx_val = idx_test[: len(idx_test) // 2]
        idx_test = idx_test[len(idx_test) // 2:]
        train_data = np_dataset[idx_train]
        test_data = np_dataset[idx_test]
        val_data = np_dataset[idx_val]

        record_id, tt, vals, mask, labels = train_data[0]
        input_dim = vals.size(-1)

        batch_size = min(min(len(dataset_obj), args.batch_size), 10000)
        test_data_combined = variable_time_collate_fn(test_data, device, classify=True,
                                                      activity=True)

        train_data_combined = variable_time_collate_fn(
            train_data, device, classify=True, activity=True)
        val_data_combined = variable_time_collate_fn(
            val_data, device, classify=True, activity=True)
        print(train_data_combined[1].sum(
        ), val_data_combined[1].sum(), test_data_combined[1].sum())
        print(train_data_combined[0].size(), train_data_combined[1].size(),
              val_data_combined[0].size(), val_data_combined[1].size(),
              test_data_combined[0].size(), test_data_combined[1].size())

        train_data_combined = TensorDataset(
            train_data_combined[0], train_data_combined[1].long())
        val_data_combined = TensorDataset(
            val_data_combined[0], val_data_combined[1].long())
        test_data_combined = TensorDataset(
            test_data_combined[0], test_data_combined[1].long())

        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(
            val_data_combined, batch_size=batch_size, shuffle=False)

        #attr_names = train_dataset_obj.params
        data_object = {"train_dataloader": train_dataloader,
                        "test_dataloader": test_dataloader,
                        "val_dataloader": val_dataloader,
                        "input_dim": input_dim,
                        }
        data_objects.append(data_object)
    return data_objects
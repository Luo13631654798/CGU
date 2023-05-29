# pylint: disable=E1101
import torch
import torch.nn as nn
import numpy as np
import math

def nufft(x, t):
    N = len(x)
    f = np.array([i / N for i in range(N)])
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * t * f[k]))
    return X, f

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def getStats(P_tensor):
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

def getStats_static(P_tensor, dataset='P12'):
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
                P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                    np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)

    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0
    P_tensor = torch.cat((P_tensor, P_time), dim=2)
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
                        nufft_complex= np.fft.fft(x)
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)

            P_tensor = torch.Tensor(P_tensor)

            P_time = torch.Tensor(P_time)
            P_tensor = torch.cat((P_tensor, P_time), dim=2)
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
                        interval = math.floor((T - 1) / (len(idx_not_zero[0]) - 1))
                        P_fft_tensor[i][1][np.arange(0, interval * (len(idx_not_zero[0]) - 1) + 1, interval).tolist(), j] = \
                            np.sqrt(nufft_complex.real * nufft_complex.real + nufft_complex.imag * nufft_complex.imag)


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

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def create_net(n_inputs, n_outputs, n_layers=0, n_units=10, nonlinear=nn.Tanh, add_softmax=False, dropout=0.0):
    if n_layers >= 0:
        layers = [nn.Linear(n_inputs, n_units)]
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(n_units, n_units))
            layers.append(nn.Dropout(p=dropout))

        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_outputs))
        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    else:
        layers = [nn.Linear(n_inputs, n_outputs)]

        if add_softmax:
            layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)

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
    elif dataset == 'PAM':
        Pdict_list = np.load(base_path + '/processed_data/PTdict_list.npy', allow_pickle=True)
        arr_outcomes = np.load(base_path + '/processed_data/arr_outcomes.npy', allow_pickle=True)
        dataset_prefix = ''  # not applicable

    idx_train, idx_val, idx_test = np.load(base_path + split_path, allow_pickle=True)

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
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]

    return Ptrain, Pval, Ptest, ytrain, yval, ytest

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
        lengths = torch.squeeze(P_length_tensor[start: start + rem])
        whatever = model.forward(P.permute(1, 0, 2), P_fft, P_delta_t.cuda(), Pstatic, lengths)
        out[start:start + rem] = whatever.detach().cpu()
    return out

def mask_normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M = 1 * (P_tensor > 0) + 0 * (P_tensor <= 0)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pf = Pf * M_3D
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    return Pfinal_tensor

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
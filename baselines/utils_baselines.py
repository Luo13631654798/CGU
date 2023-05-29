import pickle

import torch
import torch.nn as nn
import numpy as np

def evaluate(model, P_tensor, P_time_tensor, P_static_tensor, Plength_tensor, batch_size=100, n_classes=2, static=1):
    model.eval()
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    Plength_tensor = Plength_tensor.cuda()
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
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        # lengths = torch.sum(Ptime > 0, dim=0)
        lengths = torch.squeeze(Plength_tensor[start: start + batch_size])
        output = model.forward(P, Pstatic, Ptime, lengths)
        if isinstance(output, tuple):
            output = output[0]
        out[start:start + batch_size] = output.detach().cpu()
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Ptime = P_time_tensor[:, start:start + rem]
        if P_static_tensor is not None:
            Pstatic = P_static_tensor[start:start + batch_size]
        lengths = torch.squeeze(Plength_tensor[start: start + rem])
        output = model.forward(P, Pstatic, Ptime, lengths)
        if isinstance(output, tuple):
            output = output[0]
        out[start:start + rem] = output.detach().cpu()
    return out

def evaluate_standard(model, P_tensor, P_time_tensor, P_static_tensor, P_length, Pbatch_size=100, n_classes=2, static=1):
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_time_tensor.cuda()
    if static is None:
        P_static_tensor = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    lengths = torch.squeeze(P_length.cuda())
    out = model.forward(P_tensor, P_static_tensor, P_time_tensor, lengths)
    return out

def evaluate_mTAND(model, P_tensor, batch_size=100, n_classes=2):
    P_tensor = P_tensor.cuda()
    P_time_tensor = P_tensor[:, :, -1].cuda()
    T, N, Ff = P_tensor.shape

    n_batches, rem = N // batch_size, N % batch_size

    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[:, start:start + batch_size, :]
        Ptime = P_time_tensor[:, start:start + batch_size]
        output = model.forward(P[:, :, :-1].permute(1, 0, 2), Ptime.permute(1, 0)).detach().cpu()
        out[start:start + batch_size] = output
        start += batch_size
    if rem > 0:
        P = P_tensor[:, start:start + rem, :]
        Ptime = P_time_tensor[:, start:start + rem]
        output = model.forward(P[:, :, :-1].permute(1, 0, 2), Ptime.permute(1, 0)).detach().cpu()
        out[start:start + rem] = output
    return out

def evaluate_ode(model, Ptensor, Ptime_tensor, batch_size=100, n_classes=2):
    P_tensor = Ptensor.cuda()
    P_time_tensor = Ptime_tensor.cuda()
    N, T, Ff = P_tensor.shape

    n_batches, rem = N // batch_size, N % batch_size

    out = torch.zeros(N, n_classes)
    start = 0
    for i in range(n_batches):
        P = P_tensor[start:start + batch_size]
        Ptime = P_time_tensor
        output = model.forward(P, Ptime)
        out[start:start + batch_size] = output
        start += batch_size
    if rem > 0:
        P = P_tensor[start:start + rem]
        Ptime = P_time_tensor
        output = model.forward(P, Ptime)
        out[start:start + rem] = output
    return out
def evaluate_GRUD(model, x, mask, batch_size=100, n_classes=2):
    x = x.cuda()
    mask = mask.cuda()
    delta_t = torch.ones_like(x)
    for i in range(0, x.shape[0]):
        delta_t[i, :, 0] = 0
        for j in range(1, x.shape[-1]):
            delta_t[i, :, j] = delta_t[i, :, j] + (1 - mask[i, :, j]) * delta_t[i, :, j - 1]
    # outputs = model.forward(torch.cat([x.unsqueeze(0), mask.unsqueeze(0), delta_t.unsqueeze(0)], dim=0))
    N = x.shape[0]
    # T, N, Ff = P_tensor.shape

    n_batches, rem = N // batch_size, N % batch_size

    out = torch.zeros(N, n_classes)
    start = 0
    P = torch.cat([x.unsqueeze(0), mask.unsqueeze(0), delta_t.unsqueeze(0)], dim=0)
    for i in range(n_batches):
        output = model.forward(P[:, start:start+batch_size, :, :]).detach().cpu()
        out[start:start + batch_size] = output
        start += batch_size
    if rem > 0:
        output = model.forward(P[:, start:start+batch_size, :, :]).detach().cpu()
        out[start:start + rem] = output
    return out

def evaluate_MTGNN(model, P_tensor, P_static_tensor, static=1):
    P_tensor = P_tensor.cuda()

    P_tensor = torch.permute(P_tensor, (1, 0, 2))
    P_tensor = torch.unsqueeze(P_tensor, dim=1)
    P_tensor = P_tensor.transpose(2, 3)

    if static is None:
        P_static_tensor = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    out = model.forward(P_tensor, P_static_tensor)
    return out

def evaluate_DGM2(model, P_tensor, P_static_tensor, static=1):
    # suppose P_time is equal in all patients
    P_time = torch.arange(P_tensor.size()[0])

    P_tensor = P_tensor.cuda()
    P_tensor = torch.permute(P_tensor, (1, 0, 2))

    if static is None:
        P_static_tensor = None
    else:
        P_static_tensor = P_static_tensor.cuda()

    out = model.forward(P_tensor, P_time, P_static_tensor)
    return out

def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert(start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        filecontent = pickle.load(pkl_file)
    return filecontent

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2

    if len(data.size()) == 3:
        res = data[:, :, :last_dim], data[:, :, last_dim:]

    if len(data.size()) == 2:
        res = data[:, :last_dim], data[:, last_dim:]
    return res


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def subsample_timepoints(data, time_steps, mask, n_tp_to_sample=None):
    # n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
    if n_tp_to_sample is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)

    if n_tp_to_sample > 1:
        # Subsample exact number of points
        assert (n_tp_to_sample <= n_tp_in_batch)
        n_tp_to_sample = int(n_tp_to_sample)

        for i in range(data.size(0)):
            missing_idx = sorted(
                np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace=False))

            data[i, missing_idx] = 0.
            if mask is not None:
                mask[i, missing_idx] = 0.

    elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
        # Subsample percentage of points from each time series
        percentage_tp_to_sample = n_tp_to_sample
        for i in range(data.size(0)):
            # take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
            current_mask = mask[i].sum(-1).cpu()
            non_missing_tp = np.where(current_mask > 0)[0]
            n_tp_current = len(non_missing_tp)
            n_to_sample = int(n_tp_current * percentage_tp_to_sample)
            subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace=False))
            tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

            data[i, tp_to_set_to_zero] = 0.
            if mask is not None:
                mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask

def cut_out_timepoints(data, time_steps, mask, n_points_to_cut=None):
    # n_points_to_cut: number of consecutive time points to cut out
    if n_points_to_cut is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)

    if n_points_to_cut < 1:
        raise Exception("Number of time points to cut out must be > 1")

    assert (n_points_to_cut <= n_tp_in_batch)
    n_points_to_cut = int(n_points_to_cut)

    for i in range(data.size(0)):
        start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut - 5), replace=False)

        data[i, start: (start + n_points_to_cut)] = 0.
        if mask is not None:
            mask[i, start: (start + n_points_to_cut)] = 0.

    return data, time_steps, mask


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()

    batch_dict = get_dict_template()

    # remove the time points where there are no observations in this batch
    non_missing_tp = torch.sum(data_dict["observed_data"], (0, 2)) != 0.
    batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
    batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

    # print("observed data")
    # print(batch_dict["observed_data"].size())

    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

    batch_dict["data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

    non_missing_tp = torch.sum(data_dict["data_to_predict"], (0, 2)) != 0.
    batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

    # print("data_to_predict")
    # print(batch_dict["data_to_predict"].size())

    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    batch_dict["mode"] = data_dict["mode"]
    return batch_dict

def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert (start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def create_net(n_inputs, n_outputs, n_layers=1,
               n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "labels": None
            }
def split_data_interp(data_dict):
    device = get_device(data_dict["data"])

    split_dict = {"observed_data": data_dict["data"].clone(),
                  "observed_tp": data_dict["time_steps"].clone(),
                  "data_to_predict": data_dict["data"].clone(),
                  "tp_to_predict": data_dict["time_steps"].clone()}

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "interp"
    return split_dict

def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict

def subsample_observed_data(data_dict, n_tp_to_sample=None, n_points_to_cut=None):
    # n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
    # n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points

    if n_tp_to_sample is not None:
        # Randomly subsample time points
        data, time_steps, mask = subsample_timepoints(
            data_dict["observed_data"].clone(),
            time_steps=data_dict["observed_tp"].clone(),
            mask=(data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_tp_to_sample=n_tp_to_sample)

    if n_points_to_cut is not None:
        # Remove consecutive time points
        data, time_steps, mask = cut_out_timepoints(
            data_dict["observed_data"].clone(),
            time_steps=data_dict["observed_tp"].clone(),
            mask=(data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_points_to_cut=n_points_to_cut)

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict["observed_data"] = data.clone()
    new_data_dict["observed_tp"] = time_steps.clone()
    new_data_dict["observed_mask"] = mask.clone()

    if n_points_to_cut is not None:
        # Cut the section in the data to predict as well
        # Used only for the demo on the periodic function
        new_data_dict["data_to_predict"] = data.clone()
        new_data_dict["tp_to_predict"] = time_steps.clone()
        new_data_dict["mask_predicted_data"] = mask.clone()

    return new_data_dict

def check_mask(data, mask):
    # check that "mask" argument indeed contains a mask for data
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()

    # mask should contain only zeros and ones
    assert ((n_zeros + n_ones) == np.prod(list(mask.size())))

    # all masked out elements should be zeros
    assert (torch.sum(data[mask == 0.] != 0.) == 0)
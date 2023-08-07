import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PAM', choices=['P12', 'P19', 'PAM', 'physionet', 'mimic3']) #
parser.add_argument('--cuda', type=str, default='0') #
parser.add_argument('--epochs', type=int, default=20) #
parser.add_argument('--batch_size', type=int, default=128) #
parser.add_argument('--lr', type=float, default=1e-3) #
parser.add_argument('--baseline', type=str, default='mTAND', choices=['GRUD', 'SEFT', 'mTAND',
                                                                    'DGM2O', 'MTGNN', 'RAINDROP'])
args, unknown = parser.parse_known_args()
print(args)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
print(os.environ['CUDA_VISIBLE_DEVICES'])
from baseline_models import *
import sys
sys.path.append("..")
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
import warnings
from utils import *
from utils_baselines import evaluate_standard, evaluate, evaluate_MTGNN, evaluate_DGM2, evaluate_mTAND, evaluate_GRUD
warnings.filterwarnings("ignore")
wandb = False
device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.cuda.is_available())
sign = 888

arch = args.baseline
model_path = 'models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
dataset = args.dataset
baseline = args.baseline
print('Dataset used: ', dataset)


if dataset == 'P12':
    base_path = '../data/P12data'
elif dataset == 'physionet':
    base_path = '../data/physionet/PhysioNet'
elif dataset == 'P19':
    base_path = '../data/P19data'
elif dataset == 'PAM':
    base_path = '../data/PAMdata'
elif dataset == 'mimic3':
    base_path = '../data/mimic3'

batch_size = args.batch_size
num_epochs = args.epochs
learning_rate = args.lr
if args.dataset == 'P12' or args.dataset == 'physionet':
    variables_num = 36
    # timestamp_num = 160
    d_static = 9
    timestamp_num = 215
    n_class = 2
elif args.dataset == 'P19':
    d_static = 6
    variables_num = 34
    timestamp_num = 60
    n_class = 2
elif args.dataset == 'PAM':
    d_static = 0
    variables_num = 17
    timestamp_num = 600
    n_class = 8
elif args.dataset == 'mimic3':
    d_static = 0
    variables_num = 16
    timestamp_num = 292
    n_class = 2
aggreg = 'mean'

n_runs = 1
n_splits = 5
subset = False
early_stop_epochs = 100
total_time = 0
acc_arr = np.zeros((n_splits, n_runs))
auprc_arr = np.zeros((n_splits, n_runs))
auroc_arr = np.zeros((n_splits, n_runs))
precision_arr = np.zeros((n_splits, n_runs))
recall_arr = np.zeros((n_splits, n_runs))
F1_arr = np.zeros((n_splits, n_runs))
for k in range(0, n_splits):
    torch.manual_seed(k)
    torch.cuda.manual_seed(k)
    np.random.seed(k)
    split_idx = k + 1
    print('Split id: %d' % split_idx)
    if dataset == 'P12':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif dataset == 'physionet':
        split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    elif dataset == 'P19':
        split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
    elif dataset == 'PAM':
        split_path = '/splits/PAMAP2_split_' + str(split_idx) + '.npy'
    else:
        split_path = ''

    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset)
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']
        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

        Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor = \
            tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss, None)
        Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor = \
            tensorize_normalize(Pval, yval, mf, stdf, ms, ss, None)
        Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor = \
            tensorize_normalize(Ptest, ytest, mf, stdf, ms, ss, None)

    elif dataset == 'PAM':
        T, F = Ptrain[0].shape
        D = 1

        Ptrain_tensor = Ptrain
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        mf, stdf = getStats(Ptrain)
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_other(Ptrain, ytrain, mf, stdf)
        Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_other(Pval, yval, mf, stdf)
        Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_other(Ptest, ytest, mf, stdf)
    elif dataset == 'mimic3':
        T, F = timestamp_num, variables_num

        Ptrain_tensor = np.zeros((len(Ptrain), T, F))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i][:Ptrain[i][4]] = Ptrain[i][2]

        mf, stdf = getStats(Ptrain_tensor)

        Ptrain_tensor, Ptrain_fft_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
            = tensorize_normalize_with_nufft_mimic3(Ptrain, ytrain, mf, stdf)
        Pval_tensor, Pval_fft_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
            = tensorize_normalize_with_nufft_mimic3(Pval, yval, mf, stdf)
        Ptest_tensor, Ptest_fft_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
            = tensorize_normalize_with_nufft_mimic3(Ptest, ytest, mf, stdf)

    nhid = 2 * variables_num

    if baseline == 'SEFT':
        nhead = 2
        nlayers = 2
        dropout = 0.2
        MAX = 100
        model = SEFT(variables_num, variables_num, nhead, nhid, nlayers, dropout, timestamp_num, d_static, MAX, 0.5, aggreg,
                     n_class, static=False if dataset == 'PAM' or dataset == 'mimic3' else True).to(device)
    elif baseline == 'MTGNN':
        model = MTGNN(True, True, 2, variables_num * 2, torch.device('cuda:0'),
                      num_static_features=d_static,
                      node_dim=timestamp_num, dilation_exponential=2, conv_channels=16,
                      residual_channels=16, skip_channels=32,
                      end_channels=64, seq_length=timestamp_num, in_dim=1, out_dim=1, layers=5,
                      layer_norm_affline=False).to(device)
    elif baseline == 'DGM2O':
        rec_ode_func = ODEFunc(
            input_dim=10,
            latent_dim=10,
            ode_func_net=create_net(10, 10),
            device=device).to(device)

        z0_diffeq_solver = DiffeqSolver(10, rec_ode_func, "euler", 10, odeint_rtol=1e-3, odeint_atol=1e-4,
                                        device=device)

        gru_update = GRU_unit_cluster(10, variables_num * 2, n_units=10, device=device, use_mask=False, dropout=0.0)

        model = DGM2_O(10, variables_num * 2, 20, z0_diffeq_solver, z0_dim=10, n_gru_units=10, GRU_update=gru_update,
                       device=device, use_mask=False, dropout=0.0, use_static=False if d_static == 0 else True,
                       num_time_steps_and_static=(timestamp_num, 0 if d_static==0 else d_static), n_classes=n_class).to(device)
    elif baseline == 'RAINDROP':
        d_ob = 4
        d_model = variables_num * d_ob
        nhid = 2 * d_model
        nlayers = 2
        nhead = 2
        dropout = 0.2
        global_structure = torch.ones(variables_num, variables_num)
        model = Raindrop_v2(variables_num, d_model, nhead, nhid, nlayers, dropout, timestamp_num,
                            d_static, 100, 0.5, aggreg, n_class, global_structure,
                            sensor_wise_mask=False, static=False if d_static == 0 else True).to(device)
    elif baseline == 'mTAND':
        model = enc_mtan_classif(
            variables_num, torch.linspace(0, 1., 128), 32, 128, 1,
            True, 10., device=device, n_classes=n_class).to(device)
    elif baseline == 'GRUD':
        x_mean = (torch.sum(Ptrain_tensor[:, :, :variables_num].view(-1, variables_num), dim=0) \
        / torch.sum(Ptrain_tensor[:, :, variables_num:-1].view(-1, variables_num), dim=0))
        model = grud_model(input_size=variables_num, hidden_size=49,
                     output_size=n_class, x_mean=x_mean, num_layers=1).to(device)

    params = (list(model.parameters()))

    print('model', model)
    print('parameters:', count_parameters(model))

    Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
    Pval_tensor = Pval_tensor.permute(1, 0, 2)
    Ptest_tensor = Ptest_tensor.permute(1, 0, 2)

    Ptrain_delta_t_tensor = Ptrain_delta_t_tensor
    Pval_delta_t_tensor = Pval_delta_t_tensor
    Ptest_delta_t_tensor = Ptest_delta_t_tensor

    Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)
    Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)
    Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    idx_0 = np.where(ytrain == 0)[0]
    idx_1 = np.where(ytrain == 1)[0]

    n0, n1 = len(idx_0), len(idx_1)
    expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
    expanded_n1 = len(expanded_idx_1)

    batch_size = args.batch_size
    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
        K0 = n0 // int(batch_size / 2)
        K1 = expanded_n1 // int(batch_size / 2)
        n_batches = np.min([K0, K1])
    elif dataset == 'PAM':
        n_batches = 30

    best_val_epoch = 0
    best_aupr_val = best_auc_val = 0.0
    best_loss_val = 100.0

    print('Stop epochs: %d, Batches/epoch: %d, Total batches: %d' % (num_epochs, n_batches, num_epochs * n_batches))

    start = time.time()
    for epoch in range(num_epochs):
        if epoch - best_val_epoch > early_stop_epochs:
            break
        model.train()

        if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
            np.random.shuffle(expanded_idx_1)
            I1 = expanded_idx_1
            np.random.shuffle(idx_0)
            I0 = idx_0

        for n in range(n_batches):
            if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
                idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                P, Ptime, Pdelta_t, Plength, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:,idx].cuda(), \
                    Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                    Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
            elif dataset == 'PAM':
                idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                P, Ptime, Pdelta_t, Plength, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                    Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                    None, ytrain_tensor[idx].cuda()

            if baseline == 'SEFT':
                outputs = evaluate_standard(model, P[:, :, :-1], Ptime, Pstatic, Plength, static=None if d_static == 0 else 1)
            elif baseline == 'RAINDROP':
                lengths = torch.sum(Ptime > 0, dim=0)
                outputs, local_structure_regularization, _ = model.forward(P[:, :, :-1], Pstatic, Ptime, lengths)
            elif baseline == 'MTGNN':
                outputs = evaluate_MTGNN(model, P[:, :, :-1], Pstatic, static=None if d_static == 0 else 1)
            elif baseline == 'DGM2O':
                outputs = evaluate_DGM2(model, P[:, :, :-1], Pstatic, static=None if d_static == 0 else 1)
            elif baseline == 'mTAND':
                outputs = model.forward(P[:, :, :-1].permute(1, 0, 2), Ptime.permute(1, 0))
            elif baseline == 'GRUD':
                x = P[:, :, :variables_num].permute(1, 2, 0)
                mask = P[:, :, variables_num:2 * variables_num].permute(1, 2, 0)
                delta_t = torch.ones_like(x)
                for i in range(0, x.shape[0]):
                    delta_t[i, :, 0] = 0
                    for j in range(1, x.shape[-1]):
                        delta_t[i, :, j] = delta_t[i, :, j] + (1 - mask[i, :, j]) * delta_t[i, :, j - 1]
                outputs = model.forward(torch.cat([x.unsqueeze(0), mask.unsqueeze(0), delta_t.unsqueeze(0)], dim=0))
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
            train_probs = torch.squeeze(torch.sigmoid(outputs))  # 128 * 2
            train_probs = train_probs.cpu().detach().numpy()  # 128 * 2
            train_y = y.cpu().detach().numpy()  # 128
            train_auroc = roc_auc_score(train_y, train_probs[:, 1])
            train_auprc = average_precision_score(train_y, train_probs[:, 1])
        elif dataset == 'PAM':
            train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))
            train_probs = train_probs.cpu().detach().numpy()
            train_y = y.cpu().detach().numpy()

        # if epoch == 0 or epoch == num_epochs - 1:
        #     print(confusion_matrix(train_y, np.argmax(train_probs, axis=1), labels=[0, 1]))

        """Validation"""
        model.eval()
        with torch.no_grad():
            if baseline == 'SEFT' or baseline == 'RAINDROP':
                out_val = evaluate_standard(model, Pval_tensor[:, :, :-1], Pval_time_tensor, Pval_static_tensor, Pval_length_tensor,
                                            static=None if d_static == 0 else 1)
                out_val = out_val[0] if baseline == 'RAINDROP' else out_val
            elif baseline == 'mTAND':
                out_val = evaluate_mTAND(model, Pval_tensor, n_classes=n_class)
            elif baseline == 'MTGNN':
                n_batches = math.ceil(Pval_tensor.size()[1] / batch_size)
                out_val_tensors = []
                for n in range(n_batches):
                    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
                        out_val_tensors.append(evaluate_MTGNN(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                           Pval_static_tensor[n * batch_size: (n + 1) * batch_size, :] if dataset != 'mimic3' else None,
                                                              static=None if d_static == 0 else 1))
                    elif dataset == 'PAM':
                        out_val_tensors.append(evaluate_MTGNN(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                                              None, static=None if d_static == 0 else 1))
                out_val = torch.cat(out_val_tensors, dim=0)
            elif baseline == 'DGM2O':
                n_batches = math.ceil(Pval_tensor.size()[1] / batch_size)
                out_val_tensors = []
                for n in range(n_batches):
                    if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
                        out_val_tensors.append(evaluate_DGM2(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                           Pval_static_tensor[n * batch_size: (n + 1) * batch_size, :],
                                                              static=None if d_static == 0 else 1))
                    elif dataset == 'PAM':
                        out_val_tensors.append(evaluate_DGM2(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                                              None, static=None if d_static == 0 else 1))
                out_val = torch.cat(out_val_tensors, dim=0)
            elif baseline == 'GRUD':
                x = Pval_tensor[:, :, :variables_num].permute(1, 2, 0)
                mask = Pval_tensor[:, :, variables_num:2 * variables_num].permute(1, 2, 0)
                out_val = evaluate_GRUD(model, x, mask, n_classes=n_class)

            out_val = torch.squeeze(torch.sigmoid(out_val))
            out_val = out_val.detach().cpu().numpy()
            y_val_pred = np.argmax(out_val, axis=1)
            acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]
            val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

            if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
                auc_val = roc_auc_score(yval, out_val[:, 1])
                aupr_val = average_precision_score(yval, out_val[:, 1])
                print(
                    "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                    (epoch, loss.item(), train_auprc * 100, train_auroc * 100,
                     val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))
            elif dataset == 'PAM':
                auc_val = roc_auc_score(one_hot(yval), out_val)
                aupr_val = average_precision_score(one_hot(yval), out_val)
                precision = precision_score(yval, y_val_pred, average='macro', )
                recall = recall_score(yval, y_val_pred, average='macro', )
                F1 = f1_score(yval, y_val_pred, average='macro', )
                print(
                    "Validation: Epoch %d,  val_loss:%.2f, acc_val: %2f, aupr_val: %.2f, auc_val: %.2f"
                    ", precision_val: %.2f, recall_val: %.2f, F1-score_val: %.2f" %
                    (
                    epoch, val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100, precision * 100, recall * 100,
                    F1 * 100))

        if (dataset == 'PAM' and auc_val > best_auc_val) or (dataset != 'PAM' and aupr_val > best_aupr_val):
                best_val_epoch = epoch
                best_auc_val = auc_val
                best_aupr_val = aupr_val
                torch.save(model.state_dict(), model_path + arch + '_' + dataset + '_' + str(sign) + '_' + str(split_idx) + '.pt')

    end = time.time()
    time_elapsed = end - start
    total_time = total_time + time_elapsed
    print('Total epochs : %d , Time elapsed: %.3f mins' % (epoch, time_elapsed / 60.0))

    """testing"""
    model.load_state_dict(torch.load(model_path + arch + '_' + dataset + '_' + str(sign) + '_' + str(split_idx) + '.pt'))
    model.eval()

    with torch.no_grad():
        if baseline == 'SEFT' or baseline == 'RAINDROP':
            out_test = evaluate(model, Ptest_tensor[:, :, :-1], Ptest_time_tensor, Ptest_static_tensor, Ptest_length_tensor,
                                n_classes=n_class, static=None if d_static == 0 else 1).numpy()
        elif baseline == 'mTAND':
            out_test = evaluate_mTAND(model, Ptest_tensor, n_classes=n_class).numpy()
        elif baseline == 'MTGNN':
            n_batches = math.ceil(Ptest_tensor.size()[1] / batch_size)
            out_test_tensors = []
            for n in range(n_batches):
                if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
                    out_test_tensors.append(
                        evaluate_MTGNN(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                       Ptest_static_tensor[n * batch_size: (n + 1) * batch_size, :] if dataset != 'mimic3' else None,
                                       static=None if d_static == 0 else 1))
                elif dataset == 'PAM':
                    out_test_tensors.append(
                        evaluate_MTGNN(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                       None, static=None if d_static == 0 else 1))
            out_test = np.array(torch.cat(out_test_tensors, dim=0).detach().cpu())
        elif baseline == 'DGM2O':
            n_batches = math.ceil(Ptest_tensor.size()[1] / batch_size)
            out_test_tensors = []
            for n in range(n_batches):
                if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
                    out_test_tensors.append(
                        evaluate_DGM2(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                       Ptest_static_tensor[n * batch_size: (n + 1) * batch_size, :],
                                       static=None if d_static == 0 else 1))
                elif dataset == 'PAM':
                    out_test_tensors.append(
                        evaluate_DGM2(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                       None, static=None if d_static == 0 else 1))
            out_test = np.array(torch.cat(out_test_tensors, dim=0).detach().cpu())
        elif baseline == 'GRUD':
            x = Ptest_tensor[:, :, :variables_num].permute(1, 2, 0)
            mask = Ptest_tensor[:, :, variables_num:2 * variables_num].permute(1, 2, 0)
            out_test = evaluate_GRUD(model, x, mask, n_classes=n_class).numpy()

        denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
        y_test = ytest.copy()
        probs = np.exp(out_test.astype(np.float64)) / denoms
        ypred = np.argmax(out_test, axis=1)
        acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]

        if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet' or dataset == 'mimic3':
            auc = roc_auc_score(y_test, probs[:, 1])
            aupr = average_precision_score(y_test, probs[:, 1])
        elif dataset == 'PAM':
            auc = roc_auc_score(one_hot(y_test), probs)
            aupr = average_precision_score(one_hot(y_test), probs)
            precision = precision_score(y_test, ypred, average='macro', )
            recall = recall_score(y_test, ypred, average='macro', )
            F1 = f1_score(y_test, ypred, average='macro', )
            print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' % (precision * 100, recall * 100, F1 * 100))

        print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
        print('classification report', classification_report(y_test, ypred))
        print(confusion_matrix(y_test, ypred, labels=list(range(n_class))))

    # store
    acc_arr[k] = acc * 100
    auprc_arr[k] = aupr * 100
    auroc_arr[k] = auc * 100
    if dataset == 'PAM':
        precision_arr[k] = precision * 100
        recall_arr[k] = recall * 100
        F1_arr[k] = F1 * 100

# display mean and standard deviation
mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
print('------------------------------------------')
print('total_time : %d' % total_time)
print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
print('AUPRC    = %.1f +/- %.1f' % (mean_auprc, std_auprc))
print('AUROC    = %.1f +/- %.1f' % (mean_auroc, std_auroc))
if dataset == 'PAM':
    mean_precision, std_precision = np.mean(precision_arr), np.std(precision_arr)
    mean_recall, std_recall = np.mean(recall_arr), np.std(recall_arr)
    mean_F1, std_F1 = np.mean(F1_arr), np.std(F1_arr)
    print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
    print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
    print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))

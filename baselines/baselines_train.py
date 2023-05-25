
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PAM', choices=['P12', 'P19', 'PAM', 'physionet', 'mimic3']) #
parser.add_argument('--cuda', type=str, default='0') #
parser.add_argument('--epochs', type=int, default=20) #
parser.add_argument('--batch_size', type=int, default=32) #
parser.add_argument('--lr', type=float, default=1e-3) #
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--baseline', type=str, default='mTAND', choices=['GRUD', 'SEFT', 'mTAND',
                                                                    'DGM2O', 'MTGNN', 'RAINDROP'])
parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training') #
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal') #
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')
args, unknown = parser.parse_known_args()
print(args)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
print(os.environ['CUDA_VISIBLE_DEVICES'])
from baseline_models import *
import sys
sys.path.append("..")
import torch
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
import warnings
from utils import *
from utils_phy12 import evaluate_standard, evaluate, evaluate_MTGNN, evaluate_DGM2, evaluate_mTAND, evaluate_GRUD
warnings.filterwarnings("ignore")
wandb = False
device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.cuda.is_available())
sign = 888
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
arch = args.baseline
model_path = 'models/'

dataset = args.dataset
print('Dataset used: ', dataset)
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

if dataset == 'P12':
    base_path = '../data/P12data'
elif dataset == 'physionet':
    base_path = '../data/physionet/PhysioNet'
elif dataset == 'P19':
    base_path = '../data/P19data'
elif dataset == 'mimiciii':
    base_path = '../data/MIMIC-III'
elif dataset == 'PAM':
    base_path = '../data/PAMdata'


baseline = args.baseline  # always False for Raindrop
split = args.splittype  # possible values: 'random', 'age', 'gender'
reverse = args.reverse  # False or True
feature_removal_level = args.feature_removal_level  # 'set', 'sample'
batch_size = args.batch_size
print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level')
print(args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level)

"""While missing_ratio >0, feature_removal_level is automatically used"""
if args.withmissingratio == 'True':
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    missing_ratios = [0]
print('missing ratio list', missing_ratios)

sensor_wise_mask = False

for missing_ratio in missing_ratios:
    num_epochs = args.epochs
    learning_rate = args.lr  # 0.001 works slightly better, sometimes 0.0001 better, depends on settings and datasets
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
        variables_num = 12
        timestamp_num = 200
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
        split_idx = k + 1
        print('Split id: %d' % split_idx)
        if dataset == 'P12':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'physionet':
            if subset == True:
                split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'eICU':
            split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAMAP2_split_' + str(split_idx) + '.npy'

        # prepare the data:
        if dataset != 'mimic3':
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split, reverse=reverse,
                                                                      baseline=baseline, dataset=dataset,
                                                                      predictive_label=args.predictive_label)
            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
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

        nhid = 2 * variables_num
        if baseline == 'SEFT':
            nhead = 2
            nlayers = 2
            dropout = 0.2
            MAX = 100
            model = SEFT(variables_num, variables_num, nhead, nhid, nlayers, dropout, timestamp_num, d_static, MAX, 0.5, aggreg,
                         n_class, static=False if dataset == 'PAM' else True).to(device)
        elif baseline == 'Transformer':
            nhead = 1
            nlayers = 2
            dropout = 0.2
            MAX = 100
            model = TransformerModel2(variables_num, variables_num, nhead, nhid, nlayers, dropout, timestamp_num, d_static, MAX,
                                      0.5, aggreg, n_class, static=False if dataset == 'PAM' else True).to(device)
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
        # remove part of variables in validation and test set
        if missing_ratio > 0:
            num_all_features =int(Pval_tensor.shape[2] / 2)
            num_missing_features = round(missing_ratio * num_all_features)
            if feature_removal_level == 'sample':
                for i, patient in enumerate(Pval_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Pval_tensor.shape[1], num_missing_features)
                    Pval_tensor[i] = patient
                for i, patient in enumerate(Ptest_tensor):
                    idx = np.random.choice(num_all_features, num_missing_features, replace=False)
                    patient[:, idx] = torch.zeros(Ptest_tensor.shape[1], num_missing_features)
                    Ptest_tensor[i] = patient
            elif feature_removal_level == 'set':
                density_score_indices = np.load('./baselines/saved/IG_density_scores_' + dataset + '.npy', allow_pickle=True)[:, 0]
                idx = density_score_indices[:num_missing_features].astype(int)
                Pval_tensor[:, :, idx] = torch.zeros(Pval_tensor.shape[0], Pval_tensor.shape[1], num_missing_features)
                Ptest_tensor[:, :, idx] = torch.zeros(Ptest_tensor.shape[0], Ptest_tensor.shape[1], num_missing_features)

        Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)[:timestamp_num, :, :]
        Pval_tensor = Pval_tensor.permute(1, 0, 2)[:timestamp_num, :, :]
        Ptest_tensor = Ptest_tensor.permute(1, 0, 2)[:timestamp_num, :, :]

        Ptrain_delta_t_tensor = Ptrain_delta_t_tensor[:, :timestamp_num]
        Pval_delta_t_tensor = Pval_delta_t_tensor[:, :timestamp_num]
        Ptest_delta_t_tensor = Ptest_delta_t_tensor[:, :timestamp_num]

        Ptrain_time_tensor = Ptrain_time_tensor.squeeze(2).permute(1, 0)[:timestamp_num, :]
        Pval_time_tensor = Pval_time_tensor.squeeze(2).permute(1, 0)[:timestamp_num, :]
        Ptest_time_tensor = Ptest_time_tensor.squeeze(2).permute(1, 0)[:timestamp_num, :]


        for m in range(n_runs):
            print('- - Run %d - -' % (m + 1))


            criterion = torch.nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
            #                                                        patience=0, threshold=0.0001, threshold_mode='rel',
            #                                                        cooldown=5, min_lr=1e-8, eps=1e-08, verbose=True)
            # scheduler = torch.optim.lr_scheduler.WarmupLinearSchedule(optimizer,
            #                                                           warmup_steps=5,
            #                                                           t_total=num_epochs)
            idx_0 = np.where(ytrain == 0)[0]
            idx_1 = np.where(ytrain == 1)[0]
            # idx_0 = idx_0[torch.sort(Ptrain_length_tensor[idx_0, 0], descending=True).indices.numpy()]
            # idx_1 = idx_1[torch.sort(Ptrain_length_tensor[idx_1, 0], descending=True).indices.numpy()]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                strategy = 2
            elif dataset == 'PAM':
                strategy = 3

            n0, n1 = len(idx_0), len(idx_1)
            expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
            # expanded_idx_1 = np.repeat(idx_1, 3, axis=0)
            # expanded_idx_1 = idx_1
            #expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1, idx_1, idx_1, idx_1], axis=0)
            expanded_n1 = len(expanded_idx_1)

            batch_size = args.batch_size
            if strategy == 1:
                n_batches = 10
            elif strategy == 2:
                K0 = n0 // int(batch_size / 2)
                K1 = expanded_n1 // int(batch_size / 2)
                n_batches = np.min([K0, K1])
            elif strategy == 3:
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

                if strategy == 2:
                    np.random.shuffle(expanded_idx_1)
                    I1 = expanded_idx_1
                    np.random.shuffle(idx_0)
                    I0 = idx_0

                for n in range(n_batches):
                    if strategy == 1:
                        idx = random_sample(idx_0, idx_1, batch_size)
                    elif strategy == 2:
                        idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                        idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                    elif strategy == 3:
                        idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                        # idx = random_sample_8(ytrain, batch_size)   # to balance dataset

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                        P, Ptime, Pdelta_t, Plength, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(),\
                            Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                            Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                    elif dataset == 'PAM':
                        P, Ptime, Pdelta_t, Plength, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_time_tensor[:, idx].cuda(), \
                            Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                            None, ytrain_tensor[idx].cuda()


                    # outputs = model.forward(P.permute(1, 0, 2), Pdelta_t, Pstatic, Plength.squeeze(-1).to(device))
                    if baseline == 'Transformer' or baseline == 'SEFT':
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

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                    train_probs = torch.squeeze(torch.sigmoid(outputs))  # 128 * 2
                    train_probs = train_probs.cpu().detach().numpy()  # 128 * 2
                    train_y = y.cpu().detach().numpy()  # 128
                    train_auroc = roc_auc_score(train_y, train_probs[:, 1])
                    train_auprc = average_precision_score(train_y, train_probs[:, 1])
                elif dataset == 'PAM':
                    train_probs = torch.squeeze(nn.functional.softmax(outputs, dim=1))
                    train_probs = train_probs.cpu().detach().numpy()
                    train_y = y.cpu().detach().numpy()

                if epoch == 0 or epoch == num_epochs - 1:
                    print(confusion_matrix(train_y, np.argmax(train_probs, axis=1), labels=[0, 1]))

                """Validation"""
                model.eval()
                with torch.no_grad():
                    if baseline == 'Transformer' or baseline == 'SEFT' or baseline == 'RAINDROP':
                        out_val = evaluate_standard(model, Pval_tensor[:, :, :-1], Pval_time_tensor, Pval_static_tensor, Pval_length_tensor,
                                                    static=None if d_static == 0 else 1)
                        out_val = out_val[0] if baseline == 'RAINDROP' else out_val
                    elif baseline == 'mTAND':
                        out_val = evaluate_mTAND(model, Pval_tensor, n_classes=n_class)
                    elif baseline == 'MTGNN':
                        n_batches = math.ceil(Pval_tensor.size()[1] / batch_size)
                        out_val_tensors = []
                        for n in range(n_batches):
                            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                                out_val_tensors.append(evaluate_MTGNN(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                                   Pval_static_tensor[n * batch_size: (n + 1) * batch_size, :],
                                                                      static=None if d_static == 0 else 1))
                            elif dataset == 'PAM':
                                out_val_tensors.append(evaluate_MTGNN(model, Pval_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                                                      None, static=None if d_static == 0 else 1))
                        out_val = torch.cat(out_val_tensors, dim=0)
                    elif baseline == 'DGM2O':
                        n_batches = math.ceil(Pval_tensor.size()[1] / batch_size)
                        out_val_tensors = []
                        for n in range(n_batches):
                            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
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

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                        auc_val = roc_auc_score(yval, out_val[:, 1])
                        aupr_val = average_precision_score(yval, out_val[:, 1])
                    elif dataset == 'PAM':
                        auc_val = roc_auc_score(one_hot(yval), out_val)
                        aupr_val = average_precision_score(one_hot(yval), out_val)

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                        print(
                            "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                            (epoch, loss.item(), train_auprc * 100, train_auroc * 100,
                             val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))
                    elif dataset == 'PAM':
                        print(
                            "Validation: Epoch %d,  val_loss:%.2f, acc_val: %2f, aupr_val: %.2f, auc_val: %.2f" %
                            (epoch, val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))

                    #scheduler.step(auc_val)
                    if baseline == 'Transformer' or baseline == 'SEFT' or baseline == 'RAINDROP':
                        out_test = evaluate(model, Ptest_tensor[:, :, :-1], Ptest_time_tensor, Ptest_static_tensor, Ptest_length_tensor,
                                            n_classes=n_class, static=None if d_static == 0 else 1).numpy()
                    elif baseline == 'mTAND':
                        out_test = evaluate_mTAND(model, Ptest_tensor, n_classes=n_class).numpy()
                    elif baseline == 'MTGNN':
                        n_batches = math.ceil(Ptest_tensor.size()[1] / batch_size)
                        out_test_tensors = []
                        for n in range(n_batches):
                            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                                out_test_tensors.append(
                                    evaluate_MTGNN(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                                   Ptest_static_tensor[n * batch_size: (n + 1) * batch_size, :],
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
                            if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
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

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                        auc = roc_auc_score(y_test, probs[:, 1])
                        aupr = average_precision_score(y_test, probs[:, 1])
                    elif dataset == 'PAM':
                        auc = roc_auc_score(one_hot(y_test), probs)
                        aupr = average_precision_score(one_hot(y_test), probs)
                        precision = precision_score(y_test, ypred, average='macro', )
                        recall = recall_score(y_test, ypred, average='macro', )
                        F1 = f1_score(y_test, ypred, average='macro', )
                        print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' % (
                        precision * 100, recall * 100, F1 * 100))

                    print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % (auc * 100, aupr * 100, acc * 100))
#                    if auc_val > best_auc_val:
                    if (dataset == 'PAM' and auc_val > best_auc_val) or (dataset != 'PAM' and aupr_val > best_aupr_val):
                        best_val_epoch = epoch
                        best_auc_val = auc_val
                        best_aupr_val = aupr_val
                        print(
                            "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (
                            epoch, aupr_val * 100, auc_val * 100))
                        torch.save(model.state_dict(), model_path + arch + '_' + dataset + '_' + str(sign) + '_' + str(split_idx) + '.pt')

            end = time.time()
            time_elapsed = end - start
            total_time = total_time + time_elapsed
            print('Total epochs : %d , Time elapsed: %.3f mins' % (epoch, time_elapsed / 60.0))

            """testing"""
            model.load_state_dict(torch.load(model_path + arch + '_' + dataset + '_' + str(sign) + '_' + str(split_idx) + '.pt'))
            model.eval()

            with torch.no_grad():
                if baseline == 'Transformer' or baseline == 'SEFT' or baseline == 'RAINDROP':
                    out_test = evaluate(model, Ptest_tensor[:, :, :-1], Ptest_time_tensor, Ptest_static_tensor, Ptest_length_tensor,
                                        n_classes=n_class, static=None if d_static == 0 else 1).numpy()
                elif baseline == 'mTAND':
                    out_test = evaluate_mTAND(model, Ptest_tensor, n_classes=n_class).numpy()
                elif baseline == 'MTGNN':
                    n_batches = math.ceil(Ptest_tensor.size()[1] / batch_size)
                    out_test_tensors = []
                    for n in range(n_batches):
                        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                            out_test_tensors.append(
                                evaluate_MTGNN(model, Ptest_tensor[:, n * batch_size: (n + 1) * batch_size, :-1],
                                               Ptest_static_tensor[n * batch_size: (n + 1) * batch_size, :],
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
                        if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
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

                if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
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
            acc_arr[k, m] = acc * 100
            auprc_arr[k, m] = aupr * 100
            auroc_arr[k, m] = auc * 100
            if dataset == 'PAM':
                precision_arr[k, m] = precision * 100
                recall_arr[k, m] = recall * 100
                F1_arr[k, m] = F1 * 100

    # pick best performer for each split based on max AUPRC
    idx_max = np.argmax(auprc_arr, axis=1)
    acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
    auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
    auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
    if dataset == 'PAM':
        precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
        recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
        F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]

    print("missing ratio:{}, split type:{}, reverse:{}, using baseline:{}".format(missing_ratio, split, reverse,
                                                                                  baseline))
    print('args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level',
          args.dataset, args.splittype, args.reverse, args.withmissingratio, args.feature_removal_level)

    # display mean and standard deviation
    mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
    mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
    mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
    print('------------------------------------------')
    print('total_time : %d' % total_time)
    print('Accuracy = %.1f +/- %.1f' % (mean_acc, std_acc))
    print('AUPRC    = %.1f +/- %.1f' % (mean_auprc, std_auprc))
    print('AUROC    = %.1f +/- %.1f' % (mean_auroc, std_auroc))
    if dataset == 'PAM':
        mean_precision, std_precision = np.mean(precision_vec), np.std(precision_vec)
        mean_recall, std_recall = np.mean(recall_vec), np.std(recall_vec)
        mean_F1, std_F1 = np.mean(F1_vec), np.std(F1_vec)
        print('Precision = %.1f +/- %.1f' % (mean_precision, std_precision))
        print('Recall    = %.1f +/- %.1f' % (mean_recall, std_recall))
        print('F1        = %.1f +/- %.1f' % (mean_F1, std_F1))

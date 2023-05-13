import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='P12', choices=['P12', 'P19', 'PAM', 'physionet', 'mimic3']) #
parser.add_argument('--cuda', type=str, default='0') #
parser.add_argument('--epochs', type=int, default=10) #
parser.add_argument('--batch_size', type=int, default=96) #
parser.add_argument('--lr', type=float, default=1e-3) #
parser.add_argument('--attention_d_model', type=int, default=128) #
parser.add_argument('--graph_node_d_model', type=int, default=64) #
parser.add_argument('--kernel_size', type=int, default=3) #
parser.add_argument('--n_layer', type=int, default=1) #
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'], help='only use for P12 and P19')
parser.add_argument('--ablation', type=str, default='full', choices=['full', 'wo_temporal_attention', 'wo_frequency_variable_attention',
                                                                     'wo_variable_attention', 'wo_time_interval_modeling'])

parser.add_argument('--reverse', default=False, help='if True,use female, older for tarining; if False, use female or younger for training') #
parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                    help='use this only when splittype==random; otherwise, set as no_removal') #
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                    help='use this only with P12 dataset (mortality or length of stay)')
args, unknown = parser.parse_known_args()
print(args)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import torch
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
# from utils_rd import *
from ASTGCN_r import *
import warnings
from utils import *
warnings.filterwarnings("ignore")
wandb = False
device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.cuda.is_available())
sign = 88
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
arch = 'mymodelnufft0506'
model_path = 'models/'

dataset = args.dataset
print('Dataset used: ', dataset)
def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
if dataset == 'P12':
    base_path = 'data/P12data'
elif dataset == 'physionet':
    base_path = 'data/physionet/PhysioNet'
elif dataset == 'P19':
    base_path = 'data/P19data'
elif dataset == 'mimiciii':
    base_path = 'data/MIMIC-III'
elif dataset == 'PAM':
    base_path = 'data/PAMdata'


baseline = False  # always False for Raindrop
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
    learning_rate = args.lr  # 0.001 works slightly better, sometimes 0.0001 better, depends on settings and datasets
    warm_up_epochs = -1
    at = 0
    bt = 0
    varatt_dim = 0
    if args.dataset == 'physionet':
        variables_num = 36
        d_static = 9
        timestamp_num = 215
        n_class = 2
        num_epochs = 10
        early_stop_epochs = 5
        attention_d_model = 128
        graph_node_d_model = 64
    elif args.dataset == 'P12':
        variables_num = 36
        d_static = 9
        timestamp_num = 215
        n_class = 2
        num_epochs = 10
        early_stop_epochs = 5
        attention_d_model = 128
        graph_node_d_model = 64
    elif args.dataset == 'P19':
        d_static = 6
        variables_num = 34
        timestamp_num = 60
        n_class = 2
        num_epochs = 10
        early_stop_epochs = 5
        varatt_dim = 256
        attention_d_model = 128
        graph_node_d_model = 64
    elif args.dataset == 'PAM':
        d_static = 0
        variables_num = 17
        timestamp_num = 600
        n_class = 8
        num_epochs = 20
        early_stop_epochs = 20
        at = 1- 1e-6
        bt = 1e-3
        attention_d_model = 64
        graph_node_d_model = 32
        learning_rate = 0.005
        batch_size = 128
    elif args.dataset == 'mimic3':
        variables_num = 12
        timestamp_num = 200
        n_class = 2


    aggreg = 'mean'

    n_runs = 1
    n_splits = 5
    subset = False

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

            Ptrain_tensor, Ptrain_fft_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                = tensorize_normalize_with_nufft(Ptrain, ytrain, mf, stdf, ms, ss, None)
            Pval_tensor, Pval_fft_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                = tensorize_normalize_with_nufft(Pval, yval, mf, stdf, ms, ss, None)
            Ptest_tensor, Ptest_fft_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                = tensorize_normalize_with_nufft(Ptest, ytest, mf, stdf, ms, ss, None)
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1

            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            Ptrain_tensor, Ptrain_fft_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                = tensorize_normalize_other_with_nufft(Ptrain, ytrain, mf, stdf)
            Pval_tensor, Pval_fft_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                = tensorize_normalize_other_with_nufft(Pval, yval, mf, stdf)
            Ptest_tensor, Ptest_fft_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                = tensorize_normalize_other_with_nufft(Ptest, ytest, mf, stdf)
        elif dataset == 'mimic3':
            data_objects = get_mimiciii_data(batch_size=batch_size, split=k)
            train_dataloader = data_objects['train_dataloader']
            val_dataloader = data_objects['val_dataloader']
            test_dataloader = data_objects['test_dataloader']


        # model = make_model(DEVICE=device, attention_d_model=args.attention_d_model, graph_node_d_model=args.graph_node_d_model,
        #                    num_of_vertices=variables_num, num_of_timesteps=timestamp_num, n_layer=args.n_layer,
        #                    kernel_size=args.kernel_size, d_static=d_static, n_class=n_class, ablation=args.ablation,
#        #                    at=at, bt=bt, varatt_dim=varatt_dim)
#        model = CGU_wovaratt(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
#                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
        model = CGU(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
#        model = CGU_small(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
#                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
#        model = CGU_wo_timeatt(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
#                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
#        model = CGU_wovaratt(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
#                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
#        model = CGU_timevaratt(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
#                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
                                         
#        model = CGU_wo_interval(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
#                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim)
        params = (list(model.parameters()))

        print('model', model)
        print('parameters:', count_parameters(model))

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

#            batch_size = args.batch_size
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
                        P, P_fft, Ptime, Pdelta_t, Plength, Pstatic, y = \
                            Ptrain_tensor[:, idx, :].cuda(), Ptrain_fft_tensor[idx].cuda(), \
                            Ptrain_time_tensor[:, idx].cuda(),\
                            Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                            Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                    elif dataset == 'PAM':
                        P, P_fft, Ptime, Pdelta_t, Plength, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_fft_tensor[idx].cuda(), \
                            Ptrain_time_tensor[:, idx].cuda(), \
                            Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                            None, ytrain_tensor[idx].cuda()

                    outputs = model.forward(P.permute(1, 0, 2), P_fft, Pdelta_t, Pstatic, Plength.squeeze(-1).to(device))

                    optimizer.zero_grad()
                    loss = criterion(outputs, y)
#                    print(n, ":", loss)
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
                    out_val = evaluate_nufft(model, Pval_tensor, Pval_fft_tensor, Pval_time_tensor, Pval_static_tensor,
                                       Pval_delta_t_tensor, Pval_length_tensor,
                                       n_classes=n_class, batch_size=64)
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


                    out_test = evaluate_nufft(model, Ptest_tensor, Ptest_fft_tensor, Ptest_time_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor,
                                        n_classes=n_class, batch_size=64).numpy()

                    denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
                    y_test = ytest.copy()
                    probs = np.exp(out_test.astype(np.float64)) / denoms
                    ypred = np.argmax(out_test, axis=1)

                    acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]

                    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU' or dataset == 'physionet':
                        auc = roc_auc_score(y_test, probs[:, 1])
                        aupr = average_precision_score(y_test, probs[:, 1])
                        print(
                            "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f test_acc: %.2f test_auc: %.2f test_aupr: %.2f" %
                            (epoch, loss.item(), train_auprc * 100, train_auroc * 100,
                             val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100, acc * 100, auc * 100,
                             aupr * 100))
                    elif dataset == 'PAM':
                        auc = roc_auc_score(one_hot(y_test), probs)
                        aupr = average_precision_score(one_hot(y_test), probs)
                        precision = precision_score(y_test, ypred, average='macro', )
                        recall = recall_score(y_test, ypred, average='macro', )
                        F1 = f1_score(y_test, ypred, average='macro', )
                        print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' % (
                        precision * 100, recall * 100, F1 * 100))
                        print(
                            "Validation: Epoch %d,  val_loss:%.2f, acc_val: %2f, aupr_val: %.2f, auc_val: %.2f" %
                            (epoch, val_loss.item(), acc_val * 100, aupr_val * 100, auc_val * 100))


                    #scheduler.step(auc_val)
                    if (dataset == 'PAM' and auc_val > best_auc_val) or (dataset != 'PAM' and aupr_val > best_aupr_val):
                        best_auc_val = auc_val
                        best_aupr_val = aupr_val
                        best_val_epoch = epoch
                        print(
                            "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (
                            epoch, aupr_val * 100, auc_val * 100))
                        torch.save(model.state_dict(), model_path + arch + '_' + dataset + '_' + str(sign) + '_' + str(split_idx) + '.pt')

            end = time.time()
            time_elapsed = end - start
            print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

            """testing"""
            model.load_state_dict(torch.load(model_path + arch + '_' + dataset + '_' + str(sign) + '_' + str(split_idx) + '.pt'))
            model.eval()

            with torch.no_grad():
                out_test = evaluate_nufft(model, Ptest_tensor, Ptest_fft_tensor, Ptest_time_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor,
                                    n_classes=n_class, batch_size=64).numpy()

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

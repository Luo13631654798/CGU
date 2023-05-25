import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='PAM', choices=['P12', 'P19', 'PAM', 'physionet', 'mimic3']) #
parser.add_argument('--cuda', type=str, default='0') #
parser.add_argument('--epochs', type=int, default=10) #
parser.add_argument('--batch_size', type=int, default=96) #
parser.add_argument('--lr', type=float, default=1e-3) #
parser.add_argument('--attention_d_model', type=int, default=128) #
parser.add_argument('--graph_node_d_model', type=int, default=64) #
parser.add_argument('--early_stop_epochs', type=int, default=5) #
parser.add_argument('--varatt_dim', type=int, default=0) #
parser.add_argument('--at', type=float, default=0)  #
parser.add_argument('--bt', type=float, default=0)  #
parser.add_argument('--beta_start', type=float, default=1e-5)  #
parser.add_argument('--beta_end', type=float, default=2e-5)  #
parser.add_argument('--kernel_size', type=int, default=3) #
parser.add_argument('--n_layer', type=int, default=1) #
# parser.add_argument('--ablation', type=str, default='full', choices=['full', 'wo_temporal_attention', 'wo_frequency_variable_attention',
#                                                                      'wo_variable_attention', 'wo_time_interval_modeling'])
parser.add_argument('--missingtype', type=str, default='nomissing', choices=['nomissing', 'time', 'variable'])
args, unknown = parser.parse_known_args()
print(args)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import time
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score, precision_score, recall_score, f1_score
from cgu_model import *
import warnings
warnings.filterwarnings("ignore")
from utils import *
import sys
device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.cuda.is_available())
sign = 8
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
arch = 'mymodelnufftmissing'
model_path = 'models/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
dataset = args.dataset
print('Dataset used: ', dataset)

if dataset == 'P12':
    base_path = 'data/P12data'
elif dataset == 'physionet':
    base_path = 'data/physionet/PhysioNet'
elif dataset == 'P19':
    base_path = 'data/P19data'
elif dataset == 'PAM':
    base_path = 'data/PAMdata'

batch_size = args.batch_size

if args.missingtype != 'nomissing':
    missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
else:
    missing_ratios = [0]
print('missing ratio list', missing_ratios)

for missing_ratio in missing_ratios:
    learning_rate = args.lr  # 0.001 works slightly better, sometimes 0.0001 better, depends on settings and datasets
    at = args.at
    bt = args.bt
    varatt_dim = args.varatt_dim
    num_epochs = args.epochs
    early_stop_epochs = args.early_stop_epochs
    attention_d_model = args.attention_d_model
    graph_node_d_model = args.graph_node_d_model
    beta_start = args.beta_start
    beta_end = args.beta_end
    if args.dataset == 'physionet':
        variables_num = 36
        d_static = 9
        timestamp_num = 215
        n_class = 2
    elif args.dataset == 'P12':
        variables_num = 36
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

    n_splits = 5
    subset = False

    acc_arr = np.zeros(n_splits)
    auprc_arr = np.zeros(n_splits)
    auroc_arr = np.zeros(n_splits)
    precision_arr = np.zeros(n_splits)
    recall_arr = np.zeros(n_splits)
    F1_arr = np.zeros(n_splits)
    if args.missingtype == 'time':
        timestamp_num = int(timestamp_num * (1 - missing_ratio))
    for k in range(0, n_splits):
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

        # prepare the data:
        if dataset != 'mimic3':
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset)
            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])
            if args.missingtype == 'variable':
                idx = np.sort(np.random.choice(F, round((1 - missing_ratio) * F), replace=False))
            else:
                idx = None
            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            for i in range(len(Ptrain)):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            mf, stdf = getStats(Ptrain_tensor)
            ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

            Ptrain_tensor, Ptrain_fft_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                = tensorize_normalize_with_nufft_misssing(Ptrain, ytrain, mf, stdf, ms, ss, args.missingtype, missing_ratio, idx = None)
            Pval_tensor, Pval_fft_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                = tensorize_normalize_with_nufft_misssing(Pval, yval, mf, stdf, ms, ss, args.missingtype, missing_ratio, idx = None)
            Ptest_tensor, Ptest_fft_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                = tensorize_normalize_with_nufft_misssing(Ptest, ytest, mf, stdf, ms, ss, args.missingtype, missing_ratio, idx = None)
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1
            if args.missingtype == 'variable':
                idx = np.sort(np.random.choice(F, round((1 - missing_ratio) * F), replace=False))
            else:
                idx = None
            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            # if args.missingtype == 'time':
            Ptrain_tensor, Ptrain_fft_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                = tensorize_normalize_other_missing_with_nufft(Ptrain, ytrain, mf, stdf, missingtype=args.missingtype,
                                                    missingratio=missing_ratio, idx=idx)
            Pval_tensor, Pval_fft_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                = tensorize_normalize_other_missing_with_nufft(Pval, yval, mf, stdf, missingtype=args.missingtype,
                                                    missingratio=missing_ratio, idx=idx)
            Ptest_tensor, Ptest_fft_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                = tensorize_normalize_other_missing_with_nufft(Ptest, ytest, mf, stdf, missingtype=args.missingtype,
                                                    missingratio=missing_ratio, idx=idx)

        model = CGU(device, attention_d_model, graph_node_d_model, variables_num, timestamp_num,
                                         d_static, n_class, args.n_layer, args.kernel_size, at=at, bt=bt, varatt_dim=varatt_dim, beta_start=beta_start, beta_end=beta_end)

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
        # expanded_idx_1 = np.repeat(idx_1, 3, axis=0)
        # expanded_idx_1 = idx_1
        #expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1, idx_1, idx_1, idx_1], axis=0)
        expanded_n1 = len(expanded_idx_1)

        if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
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

            if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
                np.random.shuffle(expanded_idx_1)
                I1 = expanded_idx_1
                np.random.shuffle(idx_0)
                I0 = idx_0

            for n in range(n_batches):
                if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
                    idx0_batch = I0[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                    idx1_batch = I1[n * int(batch_size / 2):(n + 1) * int(batch_size / 2)]
                    idx = np.concatenate([idx0_batch, idx1_batch], axis=0)
                    P, P_fft, Ptime, Pdelta_t, Plength, Pstatic, y = \
                        Ptrain_tensor[:, idx, :].cuda(), Ptrain_fft_tensor[idx].cuda(), \
                            Ptrain_time_tensor[:, idx].cuda(), \
                            Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                            Ptrain_static_tensor[idx].cuda(), ytrain_tensor[idx].cuda()
                elif dataset == 'PAM':
                    idx = np.random.choice(list(range(Ptrain_tensor.shape[1])), size=int(batch_size), replace=False)
                    P, P_fft, Ptime, Pdelta_t, Plength, Pstatic, y = Ptrain_tensor[:, idx, :].cuda(), Ptrain_fft_tensor[
                        idx].cuda(), \
                        Ptrain_time_tensor[:, idx].cuda(), \
                        Ptrain_delta_t_tensor[idx, :].cuda(), Ptrain_length_tensor[idx], \
                        None, ytrain_tensor[idx].cuda()

                outputs = model.forward(P.permute(1, 0, 2), P_fft, Pdelta_t, Pstatic, Plength.squeeze(-1).to(device))

                optimizer.zero_grad()
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
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
                out_val = evaluate_nufft(model, Pval_tensor, Pval_fft_tensor, Pval_time_tensor, Pval_static_tensor,
                                   Pval_delta_t_tensor, Pval_length_tensor,
                                   n_classes=n_class, batch_size=64)
                out_val = torch.squeeze(torch.sigmoid(out_val))
                out_val = out_val.detach().cpu().numpy()
                y_val_pred = np.argmax(out_val, axis=1)
                acc_val = np.sum(yval.ravel() == y_val_pred.ravel()) / yval.shape[0]
                val_loss = criterion(torch.from_numpy(out_val), torch.from_numpy(yval.squeeze(1)).long())

                if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
                    auc_val = roc_auc_score(yval, out_val[:, 1])
                    aupr_val = average_precision_score(yval, out_val[:, 1])
                    print(
                        "Validation: Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, "
                        "aupr_val: %.2f, auc_val: %.2f" %
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
                    best_auc_val = auc_val
                    best_aupr_val = aupr_val
                    best_val_epoch = epoch
                    print(
                        "**[S] Epoch %d, aupr_val: %.4f, auc_val: %.4f **" % (
                        epoch, aupr_val * 100, auc_val * 100))
                    torch.save(model.state_dict(),
                               model_path + arch + '_' + dataset + '_' + args.missingtype + '_' + str(split_idx) + '.pt')

        end = time.time()
        time_elapsed = end - start
        print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

        """testing"""
        model.load_state_dict(torch.load(model_path + arch + '_' + dataset + '_' + args.missingtype + '_' + str(split_idx) + '.pt'))
        model.eval()

        with torch.no_grad():
            out_test = evaluate_nufft(model, Ptest_tensor, Ptest_fft_tensor, Ptest_time_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor,
                                n_classes=n_class, batch_size=64).numpy()

            denoms = np.sum(np.exp(out_test.astype(np.float64)), axis=1).reshape((-1, 1))
            y_test = ytest.copy()
            probs = np.exp(out_test.astype(np.float64)) / denoms
            ypred = np.argmax(out_test, axis=1)
            acc = np.sum(y_test.ravel() == ypred.ravel()) / y_test.shape[0]

            if dataset == 'P12' or dataset == 'P19' or dataset == 'physionet':
                auc = roc_auc_score(y_test, probs[:, 1])
                aupr = average_precision_score(y_test, probs[:, 1])
            elif dataset == 'PAM':
                auc = roc_auc_score(one_hot(y_test), probs)
                aupr = average_precision_score(one_hot(y_test), probs)
                precision = precision_score(y_test, ypred, average='macro', )
                recall = recall_score(y_test, ypred, average='macro', )
                F1 = f1_score(y_test, ypred, average='macro', )
                print('Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f' %
                      (precision * 100, recall * 100, F1 * 100))

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

    print("missing ratio:{}".format(missing_ratio))
    # display mean and standard deviation
    mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
    mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
    mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
    print('------------------------------------------')
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

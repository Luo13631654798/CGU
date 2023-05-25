import argparse
import numpy as np
import logging, os
import sys
import time
sys.path.append('..')
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import roc_auc_score, classification_report, \
    confusion_matrix, average_precision_score, precision_score, recall_score, f1_score

import keras
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, GRU, Lambda, Permute
from keras.models import Model
from interpolation_layer import single_channel_interp, cross_channel_interp
# from mimic_preprocessing import load_data, trim_los, fix_input_format
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1)
tf.set_random_seed(1)
# tf.random.set_seed(10)

def hold_out(mask, perc=0.2):
    """To implement the autoencoder component of the loss, we introduce a set
    of masking variables mr (and mr1) for each data point. If drop_mask = 0,
    then we removecthe data point as an input to the interpolation network,
    and includecthe predicted value at this time point when assessing
    the autoencoder loss. In practice, we randomly select 20% of the
    observed data points to hold out from
    every input time series."""
    drop_mask = np.ones_like(mask)
    drop_mask *= mask
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            count = np.sum(mask[i, j], dtype='int')
            if int(0.20*count) > 1:
                index = 0
                r = np.ones((count, 1))
                b = np.random.choice(count, int(0.20*count), replace=False)
                r[b] = 0
                for k in range(mask.shape[2]):
                    if mask[i, j, k] > 0:
                        drop_mask[i, j, k] = r[index]
                        index += 1
    return drop_mask

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def mean_imputation(vitals, mask):
    """For the time series missing entirely, our interpolation network
    assigns the starting point (time t=0) value of the time series to
    the global mean before applying the two-layer interpolation network.
    In such cases, the first interpolation layer just outputs the global
    mean for that channel, but the second interpolation layer performs
    a more meaningful interpolation using the learned correlations from
    other channels."""
    counts = np.sum(np.sum(mask, axis=2), axis=0)   # 12维
    mean_values = np.sum(np.sum(vitals*mask, axis=2), axis=0)/counts  # 12维
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if np.sum(mask[i, j]) == 0:
                mask[i, j, 0] = 1
                vitals[i, j, 0] = mean_values[j]
    return


ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,
                help="# of GPUs to use for training")
ap.add_argument("-batch", "--batch_size", type=int, default=128,
                help="# batch size to use for training")
ap.add_argument("-e", "--epochs", type=int, default=20,
                help="# of epochs for training")
ap.add_argument("-ref", "--reference_points", type=int,
                default=192, help="# of reference points")
ap.add_argument("-d", "--dataset", type=str, default='P19', choices=['P12', 'P19', 'physionet', 'PAM'],
                help="dataset to use")
ap.add_argument("-units", "--hidden_units", type=int,
                default=100, help="# of hidden units")

ap.add_argument("-hfadm", "--hours_from_adm", type=int,
                default=48, help="Hours of record to look at")

args = vars(ap.parse_args())
gpu_num = args["gpus"]
epoch = args["epochs"]
hid = args["hidden_units"]
ref_points = args["reference_points"]
hours_look_ahead = args["hours_from_adm"]
if gpu_num > 0:
    batch = args["batch_size"]*gpu_num
else:
    batch = args["batch_size"]


# Loading dataset
# y : (N,) discrete for classification, real values for regression
# x : (N, D, tn) input multivariate time series data with dimension
#     where N is number of data cases, D is the dimension of
#     sparse and irregularly sampled time series and tn is the union
#     of observed time stamps in all the dimension for a data case n.
#     Since each tn is of variable length, we pad them with zeros to
#     have an array representation.
# m : (N, D, tn) where m[i,j,k] = 0 means that x[i,j,k] is not observed.
# T : (N, D, tn) represents the actual time stamps of observation;

# vitals, label = load_data()
# # vitals = vitals[:3600]
# # label = label[:3600]
# vitals, timestamps = trim_los(vitals, hours_look_ahead)  # 53211 * 12 * 2881 ( 只有每个患者总观察时间戳是具体值或缺失的话是-100， 后面到2881都为零 )
# x, m, T = fix_input_format(vitals, timestamps)  # 53211 * 12 * 200 , 53211 * 12 * 200 ,53211 * 200
# mean_imputation(x, m)  # 对于完全缺失的患者的某一变量 用全部患者该变量的均值作为t=0时刻的插补
# x = np.concatenate((x, m, T[:, 0, :], hold_out(m)), axis=1)  # input format 53211 * 48 * 200 观测值、掩码、时间戳、随机缺失时间戳
# y = np.array(label)
# print(x.shape, y.shape)
# np.save('final_input.npy', x)
# np.save('final_output.npy', y)
# x = np.load('final_input.npy', allow_pickle=True)
# y = np.load('final_output.npy', allow_pickle=True)


dataset = args["dataset"]

if dataset == 'P12' or dataset == 'physionet':
    variables_num = 36
    # timestamp_num = 160
    d_static = 9
    timestamp_num = 215
    n_class = 2
elif dataset == 'P19':
    d_static = 6
    variables_num = 34
    timestamp_num = 60
    n_class = 2
elif dataset == 'PAM':
    d_static = 0
    variables_num = 17
    timestamp_num = 600
    n_class = 8
elif dataset == 'mimic3':
    variables_num = 12
    timestamp_num = 200
    n_class = 2

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


timestamp = timestamp_num
num_features = variables_num
total_time = 0

def customloss(ytrue, ypred):
    """ Autoencoder loss
    """
    # standard deviation of each feature mentioned in paper for MIMIC_III data
    # wc = np.array([3.33, 23.27, 5.69, 22.45, 14.75, 2.32,
    #                3.75, 1.0, 98.1, 23.41, 59.32, 1.41])
    # wc.shape = (1, num_features)
    y = ytrue[:, :num_features, :]
    m2 = ytrue[:, 3*num_features:4*num_features, :]
    m2 = 1 - m2
    m1 = ytrue[:, num_features:2*num_features, :]
    m = m1*m2
    ypred = ypred[:, :num_features, :]
    x = (y - ypred)*(y - ypred)
    x = x*m
    count = tf.reduce_sum(m, axis=2)
    count = tf.where(count > 0, count, tf.ones_like(count))
    x = tf.reduce_sum(x, axis=2)/count
    # x = x/(wc**2)  # dividing by standard deviation
    x = tf.reduce_sum(x, axis=1)/num_features
    return tf.reduce_mean(x)


seed = 0

# interpolation-prediction network


def interp_net():
    dev = "/GPU:0"
    with tf.device(dev):
        main_input = Input(shape=(4*variables_num, timestamp), name='input')
        sci = single_channel_interp(ref_points, hours_look_ahead)
        cci = cross_channel_interp()
        interp = cci(sci(main_input))
        reconst = cci(sci(main_input, reconstruction=True),
                      reconstruction=True)
        aux_output = Lambda(lambda x: x, name='aux_output')(reconst)
        z = Permute((2, 1))(interp)
        z = GRU(hid, activation='tanh', recurrent_dropout=0.2, dropout=0.2)(z)
        main_output = Dense(n_class, activation='sigmoid', name='main_output')(z)
        orig_model = Model([main_input], [main_output, aux_output])
    if gpu_num > 1:
        model = multi_gpu_model(orig_model, gpus=gpu_num)
    else:
        model = orig_model
    print(orig_model.summary())
    return model


earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0000, patience=10, verbose=0)
# callbacks_list = [earlystop]

# 5-fold cross-validation

# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
missingtype = 'time'
#missingtype = 'variable'
for missing_ratio in missing_ratios:
    if missingtype == 'time':
        timestamp = int(timestamp_num * (1 - missing_ratio))
    results = {}
    results['loss'] = []
    results['auc'] = []
    results['acc'] = []
    results['auprc'] = []
    results['precision'] = []
    results['recall'] = []
    results['F1'] = []
    for i in range(5):
        print("Running Fold:", i+1)

        split_idx = i + 1
        if dataset == 'P12':
            split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'physionet':
            split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
        elif dataset == 'P19':
            split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
        elif dataset == 'eICU':
            split_path = '/splits/eICU_split' + str(split_idx) + '.npy'
        elif dataset == 'PAM':
            split_path = '/splits/PAMAP2_split_' + str(split_idx) + '.npy'

        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset)

        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))


        if missingtype == 'variable':
            idx = np.sort(np.random.choice(variables_num, round((1 - missing_ratio) * variables_num), replace=False))
        else:
            idx = None

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

            # Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor = \
            #     tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss, None)
            # Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor = \
            #     tensorize_normalize(Pval, yval, mf, stdf, ms, ss, None)
            # Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor = \
            #     tensorize_normalize(Ptest, ytest, mf, stdf, ms, ss, None)
            if missingtype == 'time':
                Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                    = tensorize_normalize_misssing(Ptrain, ytrain, mf, stdf, ms, ss, missingtype=missingtype,
                                                   missingratio=missing_ratio)
                Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                    = tensorize_normalize_misssing(Pval, yval, mf, stdf, ms, ss, missingtype=missingtype,
                                                   missingratio=missing_ratio)
                Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                    = tensorize_normalize_misssing(Ptest, ytest, mf, stdf, ms, ss, missingtype=missingtype,
                                                   missingratio=missing_ratio)
            else:
                idx = np.sort(
                    np.random.choice(variables_num, round((1 - missing_ratio) * variables_num), replace=False))
                Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                    = tensorize_normalize_misssing(Ptrain, ytrain, mf, stdf, ms, ss, missingtype=None,
                                                   missingratio=0, idx=idx)
                Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                    = tensorize_normalize_misssing(Pval, yval, mf, stdf, ms, ss, missingtype=None,
                                                   missingratio=0, idx=idx)
                Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                    = tensorize_normalize_misssing(Ptest, ytest, mf, stdf, ms, ss, missingtype=missingtype,
                                                   missingratio=missing_ratio, idx=idx)
        elif dataset == 'PAM':
            T, F = Ptrain[0].shape
            D = 1

            Ptrain_tensor = Ptrain
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            mf, stdf = getStats(Ptrain)
            if missingtype == 'time':
                Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                    = tensorize_normalize_other_missing(Ptrain, ytrain, mf, stdf, missingtype=missingtype,
                                                        missingratio=missing_ratio)
                Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                    = tensorize_normalize_other_missing(Pval, yval, mf, stdf, missingtype=missingtype,
                                                        missingratio=missing_ratio)
                Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                    = tensorize_normalize_other_missing(Ptest, ytest, mf, stdf, missingtype=missingtype,
                                                        missingratio=missing_ratio)
            else:
                idx = np.sort(
                    np.random.choice(variables_num, round((1 - missing_ratio) * variables_num), replace=False))
                Ptrain_tensor, Ptrain_static_tensor, Ptrain_delta_t_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain_tensor \
                    = tensorize_normalize_other_missing(Ptrain, ytrain, mf, stdf, missingtype=None,
                                                        missingratio=0)
                Pval_tensor, Pval_static_tensor, Pval_delta_t_tensor, Pval_length_tensor, Pval_time_tensor, yval_tensor \
                    = tensorize_normalize_other_missing(Pval, yval, mf, stdf, missingtype=None,
                                                        missingratio=0)
                Ptest_tensor, Ptest_static_tensor, Ptest_delta_t_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest_tensor \
                    = tensorize_normalize_other_missing(Ptest, ytest, mf, stdf, missingtype=missingtype,
                                                        missingratio=missing_ratio, idx=idx)



        y_train = ytrain_tensor.numpy()
        y_test = ytest_tensor.numpy()

        x_train = Ptrain_tensor[:, :, :variables_num].permute(0, 2, 1).numpy()
        m_train = Ptrain_tensor[:, :, variables_num:-1].permute(0, 2, 1).numpy()
        t_train = torch.repeat_interleave(Ptrain_tensor[:, :, -1].unsqueeze(1), variables_num, dim=1).numpy()
        # mask_t_train = m_train * t_train
        mask_t_train = np.ones_like(t_train) * t_train
        # X_train = np.concatenate([x_train, m_train, t_train, mask_t_train], axis=1).astype(float)
        X_train = np.concatenate([x_train, m_train, t_train, np.ones_like(t_train)], axis=1).astype(float)

        x_test = Ptest_tensor[:, :, :variables_num].permute(0, 2, 1).numpy()
        m_test = Ptest_tensor[:, :, variables_num:-1].permute(0, 2, 1).numpy()
        t_test = torch.repeat_interleave(Ptest_tensor[:, :, -1].unsqueeze(1), variables_num, dim=1).numpy()
        # mask_t_test = m_test * t_test
        mask_t_test = np.ones_like(m_test) * t_test
        # X_test = np.concatenate([x_test, m_test, t_test, mask_t_test], axis=1).astype(float)
        X_test = np.concatenate([x_test, m_test, t_test, np.ones_like(m_test)], axis=1).astype(float)

        x = X_train
        y = y_train
        start = time.time()

        model = interp_net()  # re-initializing every time
        model.compile(
            optimizer='adam',
            loss={'main_output': 'binary_crossentropy', 'aux_output': customloss},
            loss_weights={'main_output': 1., 'aux_output': 0.},
            metrics={'main_output': 'accuracy'})
        weights = model.get_weights()
        # intputout_put = Model(inputs=model.input, outputs=model.get_layer('input').output).predict(x[train])
        # single_channel_output = Model(inputs=model.get_layer('single_channel_interp_1').get_input_at(0),outputs=model.get_layer('single_channel_interp_1').get_output_at(0)).predict(intputout_put)
        model.fit(
            {'input': X_train}, {'main_output': one_hot(y_train), 'aux_output': X_train},
            batch_size=batch,
            callbacks=[earlystop],
            epochs=epoch,
            validation_split=0.1,
            verbose=2)
        # weights = model.get_weights()
        end = time.time()
        total_time = total_time + (end - start)
        y_pred = model.predict(X_test, batch_size=batch)
        y_pred = y_pred[0]
        total_loss, score, reconst_loss, acc = model.evaluate(
            {'input': X_test},
            {'main_output': one_hot(y_test), 'aux_output': X_test},
            batch_size=batch,
            verbose=0)
        results['loss'].append(score)
        results['acc'].append(np.sum(np.argmax(y_pred, axis=-1) == y_test) / y_pred.shape[0])

        if n_class == 2:
            results['auc'].append(auc_score(y_test, y_pred[:, 1]))
            results['auprc'].append(auprc(y_test, y_pred[:, 1]))
        else:
            results['auc'].append(roc_auc_score(one_hot(y_test), y_pred))
            results['auprc'].append(average_precision_score(one_hot(y_test), y_pred))
            results['precision'].append(precision_score(y_test, np.argmax(y_pred, axis=-1), average='macro', ))
            results['recall'].append(recall_score(y_test, np.argmax(y_pred, axis=-1), average='macro', ))
            results['F1'].append(f1_score(y_test, np.argmax(y_pred, axis=-1), average='macro', ))
        print(results)


    if n_class == 2:
        print('total_time', total_time)
        print("avg_auc: %.2f ± %.2f,  avg_auprc: %.2f ± %.2f" % (100 * np.mean(results['auc']), 100 * np.std(results['auc']),
                                                                 100 * np.mean(results['auprc']), 100 * np.std(results['auprc'])))
    else:
        print("avg_accuracy: %.2f ± %.2f, avg_precision: %.2f ± %.2f, avg_recall: %.2f ± %.2f, avg_f1: %.2f ± %.2f,"
              % (100 * np.mean(results['acc']), 100 * np.std(results['acc']),
                 100 * np.mean(results['precision']), 100 * np.std(results['precision']),
                 100 * np.mean(results['recall']), 100 * np.std(results['recall']),
                 100 * np.mean(results['F1']), 100 * np.std(results['F1'])
                 ))

'''
Data Pre-processing on openpack dataset v3.1.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
import pickle
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import *
import pandas as pd
import gc
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from operator import itemgetter
from utils import mask_simi_series, filter_simi_by_threshold
from motif_processing.peak_func import calculate_test


def format_data_x(datafile):
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=np.float)
        if x_data is None:
            x_data = np.zeros((len(item_data), 1))
        x_data = np.hstack((x_data, item_data))
    x_data = x_data[:, 1:]
    print('x_data.shape:', x_data.shape)
    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :])
        row = row.reshape(9, 128).T
        if X is None:
            X = np.zeros((len(x_data), 128, 9))
        X[i] = row
    print('X.shape:', X.shape)
    return X

def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=np.int) - 1
    return data

def load_domain_data(args, domain_idx, domain_dict, classesdict):
    """ to load all the data from the specific domain with index domain_idx
    This function is in a iteration, we need to load each user's data from this func.
    :param domain_idx: userid, the filename
    :return: X and y data of the entire domain
    X.shape(Windowsize, dim), y.shape(windowsize), domain(windowsize)
    """
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))
    data_dir = os.path.join(root_path, 'data', 'omeData')
    saved_filename = os.path.join(data_dir, 'omewd', domain_idx + '_wd.data') # "wd": with domain label

    if os.path.isfile(os.path.join(data_dir, saved_filename)) == True:
        data = np.load(os.path.join(data_dir, saved_filename), allow_pickle=True)
        X = data[0][0]
        yy = data[0][1]
        y = np.vectorize(classesdict.get)(yy)
        d = data[0][2]
        period = data[0][-1]
    else:
        # if os.path.isdir(data_dir) == False:
        #     os.makedirs(data_dir)
        str_folder = data_dir
        datapath = os.path.join(str_folder, domain_idx + '.csv')

        # print('reading data ...')
        data_df = pd.read_csv(datapath)
        data_df.dropna(axis=0, how='any', inplace=True)
        d_df = data_df[['timestamp', 'x', 'y', 'z']].copy()
        l_df = data_df[['event_type','start','end']].copy()
        # print('convert data ...')
        data_dfL = convert_unit_system(args, d_df)

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        X = data_dfL.values[:, 1:]  # remove timestamp column
        y = l_df.values[:, 1:-1]
        d = data_df.job.values.reshape(-1,1)
        period = l_df.job.values.reshape(-1,1)
        # d = np.full(y.shape, int(domain_dict[domain_idx]), dtype=int)
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d, period)]
        f = open(saved_filename, 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    return X, y, d, period


def mask_input_data(mask_ratio, sample):
    L = sample.shape[0]
    keep_mask = np.ones(L, dtype=int)
    masked_num = int(mask_ratio*L)
    ind = random.sample(range(L), masked_num)
    keep_mask[ind] = 0
    return keep_mask

class data_loader_ome(base_loader):
    def __init__(self, samples, labels, domains, mask_ratio):
        super(data_loader_ome, self).__init__(samples, labels, domains)
        self.mask_ratio = mask_ratio

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = sample
        masked = mask_input_data(self.mask_ratio, sample)

        return sample, target, domain, masked


def prep_domains_ome_period(args):
    # 在这个函数内部指定用n个period训练，其他测试。n=1,2,3,4,5?
    # 当前，这个函数同时处理MoIL和supervised模型加载。MoIL要全部data预训练，supervised不是。
    # dataloader格式不要改变

    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    full_motif_path1 = os.path.join(root_path, 'motif_processing',
                                    'save_motif_scores', args.dataset,
                                    'motif_num20_Hamming_%s_Lhand.pkl' % args.user_name)

    # test only one user
    with open(full_motif_path1, 'rb') as f:
        (X_processed, y,
         _, _, _,
         motif_similarity_periods_ham, candidate_motif_ids, _) \
            = pickle.load(f)

    tmp_columns = []
    candidate_motif_ids = calculate_test(args, motif_similarity_periods_ham)
    sortedscore = sorted(candidate_motif_ids.items(), key=itemgetter(1), reverse=True)
    for score in [sortedscore]:
        for motif_idx, _ in score[:args.num_motifs]:  # select topk motifs, k=20
            tmp_periods_simi_list1 = motif_similarity_periods_ham[motif_idx]
            tmp_periods_simi_list = mask_simi_series(tmp_periods_simi_list1, motif_idx)
            tmp_columns.append(tmp_periods_simi_list)

    # first, split data into training and test sets
    if args.split_ratio == 0:  # use all data to pretrain model
        combined_columns = []
        for col in tmp_columns:
            comb_col = np.concatenate(col)
            combined_columns.append(comb_col)
        motif_simi_win = np.concatenate([combined_columns], axis=1).transpose()

        x_win_train = np.concatenate(X_processed)
        y_win_train = np.concatenate(y)
        m_win_train = motif_simi_win
        x_win_test = np.concatenate(X_processed)
        y_win_test = np.concatenate(y)
        m_win_test = motif_simi_win
    else:
        # 根据不同periods，分训练测试集合
        num_train_periods = args.n_periods
        combined_train_columns,combined_test_columns = [], []
        for col in tmp_columns:
            traincol = col[:num_train_periods]
            testcol = col[num_train_periods:]
            comb_train_col = np.concatenate(traincol)
            comb_test_col = np.concatenate(testcol)
            combined_train_columns.append(comb_train_col)
            combined_test_columns.append(comb_test_col)
        x_win_train = np.concatenate(X_processed[:num_train_periods])
        y_win_train = np.concatenate(y[:num_train_periods])
        m_win_train = np.concatenate([combined_train_columns], axis=1).transpose()
        x_win_test = np.concatenate(X_processed[num_train_periods:])
        y_win_test = np.concatenate(y[num_train_periods:])
        m_win_test = np.concatenate([combined_test_columns], axis=1).transpose()


    # initialize args.out_fea
    if args.framework in ['ProposeSSL', 'CNNRNN']:
        args.out_fea = m_win_train.shape[-1]

    # then, generate batch of data by overlapping the training set
    x_win_trainL, y_win_trainL, m_win_trainL = [], [], []
    for idx in range(0, x_win_train.shape[0] - args.len_sw - args.step, args.step):  # step10
        x_win_trainL.append(x_win_train[idx: idx + args.len_sw, :])
        y_win_trainL.append(y_win_train[idx: idx + args.len_sw, :])
        m_win_trainL.append(m_win_train[idx: idx + args.len_sw, :])
    xlist = np.stack(x_win_trainL, axis=0)  # [B, Len, dim]
    x_win_train = xlist.reshape((xlist.shape[0], xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    y_win_train = np.stack(y_win_trainL, axis=0)
    m_win_train = np.stack(m_win_trainL, axis=0)
    print(" ..after sliding window: train inputs {0}, train targets {1}".
          format(x_win_train.shape, y_win_train.shape))
    # ---------for test set, do not overlap----------
    moredata = x_win_test.shape[0] % args.len_sw
    x_win = x_win_test[:-moredata, :]  # len,dim
    y_win = y_win_test[:-moredata, :]  # len, 1
    m_win = m_win_test[:-moredata, :]  # len, n
    x_win_test = x_win.reshape((-1, args.len_sw, args.n_feature))  # [B, Len, dim]
    y_win_test = y_win.reshape((-1, args.len_sw, y_win.shape[-1]))
    m_win_test = m_win.reshape((-1, args.len_sw, m_win.shape[-1]))
    print(" ..after sliding window: test inputs {0}, test targets {1}".
          format(x_win_test.shape, y_win_test.shape))

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    if y_win_train.ndim == 3:
        sample_weights = get_sample_weights(y_win_train[:, 0, 0], weights)
    else:
        sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_ome(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler)

    test_set_r = data_loader_ome(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False)

    # return [train_loader_r], train_loader_r, []
    return [train_loader_r], test_loader_r, []

def prep_domains_ome_random(args, SlidWindowlen, slidwindowstep,
                             if_split_user=False):
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    #algorithm 0605
    # user = 'omeu1'
    # user = 'omeu1_newRef_NOmask'
    user = 'omeu1_13Refmotif'
    # user = 'omeu1_randomMotif'
    motif_path = os.path.join(root_path, 'motif_processing',
                                    args.dataset, user+'.pkl')

    with open(motif_path, 'rb') as f:
        tmpresult = cp.load(f)
    X_processed, y, motif_simi_all = tmpresult

    # motif_simi = motif_simi_all[:-3, :].copy()  # without reference motifs
    motif_simi = motif_simi_all.copy()  # proposed
    print('without reference motifs')

    if args.split_ratio < 0:  # train with a limited periods
        simi_list = []
        count = 0
        for i in y:
            length = i.shape[0]
            simi_list.append(motif_simi[:,count:count+length])
            count += length
        train_n_periods = np.abs(args.split_ratio)  # -1,-2,-3,-4,-5
        # train set
        x_win_train = np.concatenate(X_processed[:train_n_periods])
        x_win_train = np.array(x_win_train, dtype=float)
        y_win_train = np.concatenate(y[:train_n_periods])
        motif_simi = np.concatenate(simi_list[:train_n_periods], axis=1)
        m_win_train = motif_simi.transpose()
        # test set
        x_win_test = np.concatenate(X_processed[train_n_periods:])
        x_win_test = np.array(x_win_test, dtype=float)
        y_win_test = np.concatenate(y[train_n_periods:])
        motif_simi = np.concatenate(simi_list[train_n_periods:], axis=1)
        m_win_test = motif_simi.transpose()
        if args.framework in ['ProposeSSL', 'CNNRNN']:
            args.out_fea = m_win_test.shape[-1]
    else:
        motif_simi_win = motif_simi.transpose()
        x_win = np.concatenate(X_processed)
        x_win = np.array(x_win, dtype=float)
        y_win1 = np.concatenate(y)
        y_win = y_win1.reshape((len(y_win1), 1))
        y_win = y_win.astype(int)

        if args.framework in ['ProposeSSL', 'CNNRNN']:
            args.out_fea = motif_simi_win.shape[-1]

        # first, split data into training and test sets
        if args.split_ratio == 0:  # use all data to pretrain model
            x_win_train = x_win
            y_win_train = y_win
            m_win_train = motif_simi_win
            x_win_test = x_win
            y_win_test = y_win
            m_win_test = motif_simi_win
        else:
            x_win_train, x_win_test, \
            y_win_train, y_win_test, \
            m_win_train, m_win_test = \
                train_test_split(x_win, y_win, motif_simi_win, test_size=args.split_ratio, shuffle=False) #0.2

    # then, generate batch of data by overlapping the training set
    x_win_trainL, y_win_trainL, m_win_trainL = [], [], []
    for idx in range(0, x_win_train.shape[0] - args.len_sw - args.step, args.step):  # step10
        x_win_trainL.append(x_win_train[idx: idx + args.len_sw, :])
        y_win_trainL.append(y_win_train[idx: idx + args.len_sw, :])
        m_win_trainL.append(m_win_train[idx: idx + args.len_sw, :])
    # last run
    x_win_trainL.append(x_win_train[-(args.len_sw):, :])
    y_win_trainL.append(y_win_train[-(args.len_sw):, :])
    m_win_trainL.append(m_win_train[-(args.len_sw):, :])
    # combine
    xlist = np.stack(x_win_trainL, axis=0)  # [B, Len, dim]
    x_win_train = xlist.reshape((xlist.shape[0], xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    y_win_train = np.stack(y_win_trainL, axis=0)
    m_win_train = np.stack(m_win_trainL, axis=0)
    print(" ..after sliding window: train inputs {0}, train targets {1}".
          format(x_win_train.shape, y_win_train.shape))
    # ---------for test set, do not overlap----------
    moredata = x_win_test.shape[0] % args.len_sw
    x_win = x_win_test[:-moredata, :]  # len,dim
    y_win = y_win_test[:-moredata, :]  # len, 1
    m_win = m_win_test[:-moredata, :]  # len, n
    x_win_test = x_win.reshape((-1, args.len_sw, args.n_feature))  # [B, Len, dim]
    y_win_test = y_win.reshape((-1, args.len_sw, y_win.shape[-1]))
    m_win_test = m_win.reshape((-1, args.len_sw, m_win.shape[-1]))
    # last run
    # TODO
    print(" ..after sliding window: test inputs {0}, test targets {1}".
          format(x_win_test.shape, y_win_test.shape))

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    if y_win_train.ndim == 3:
        sample_weights = get_sample_weights(y_win_train[:, 0, 0], weights)
    else:
        sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_ome(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler)

    test_set_r = data_loader_ome(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False)

    # return [train_loader_r], train_loader_r, []
    return [train_loader_r], test_loader_r, []



def prep_ome(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, if_split_user=False):
    '''
    if_split_user: when True, generate a list of dataloaders for each user.
                   When False, only a tuple of dataloader for all users.
    '''
    if args.cases == 'random':
        return prep_domains_ome_random(args, SLIDING_WINDOW_LEN,
                                        SLIDING_WINDOW_STEP, if_split_user)
    elif args.cases == 'period':
        return prep_domains_ome_period(args)
    # elif args.cases == 'subject_large':
    #     return prep_domains_ucihar_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'


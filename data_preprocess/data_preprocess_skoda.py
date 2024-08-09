'''
Data Pre-processing on openpack dataset v3.1.

'''

import os
# import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from sklearn.model_selection import train_test_split
from data_preprocess.base_loader import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import gc
import random
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


def load_domain_data(domain_idx, domain_dict, classesdict):
    """ to load all the data from the specific domain with index domain_idx
    This function is in a iteration, we need to load each user's data from this func.
    :param domain_idx: userid, the filename
    :return: X and y data of the entire domain
    X.shape(Windowsize, dim), y.shape(windowsize), domain(windowsize)
    """
    # abspath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    # root_path = os.getcwd()
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))
    data_dir = os.path.join(root_path, 'data', 'skoda', 'ROOT','COMBINED','PERIOD','Sessions')
    saved_filename = os.path.join(root_path, 'data', 'skoda',
                                  'skoda_wd',
                                  'skoda_' + domain_idx + '_wd.data')  # "wd": with domain label

    if os.path.isfile(os.path.join(saved_filename)) == True:
        data = np.load(os.path.join(saved_filename), allow_pickle=True)
        X = data[0][0]
        yy = data[0][1]
        y = np.vectorize(classesdict.get)(yy)
        d = data[0][2]
        period = data[0][-1]
    else:
        # str_folder = data_dir
        # labelpath = os.path.join(data_dir,  'annotation', 'activity-1s')
        periodIDs = ['INSTANCE00%s'%str(i) for i in range(10,70,1)]
        datalist, labellist, fulllabellist = [], [], []
        max_period = 1
        for file in periodIDs:
            input_path = os.path.join(data_dir, file, 'data', 'float64.npy')
            # print('reading data ...')
            data_dfr = np.load(input_path)
            dataL = data_dfr[3:6,:]
            dataR = data_dfr[27:30,:]
            data = np.concatenate([dataL, dataR], axis=0)

            # --------label------------
            labelpath =  os.path.join(data_dir, file, 'target', 'ID.npy')
            tl_df = np.load(labelpath)

            # plt.figure(file)
            # plt.subplot(211)
            # plt.plot(tl_df)
            # plt.subplot(212)
            # plt.plot(data.transpose())
            # plt.show()

            # downsampling is required
            FS_raw = 98
            FS = 30
            step = int(FS_raw/FS)
            # data_df = signal.decimate(data, step)
            data_df = [data[:,i] for i in range(0, len(tl_df), step)]
            data_df = pd.DataFrame(data=data_df,
                                   columns=["acc_xl", "acc_yl", "acc_zl", "acc_xr", "acc_yr", "acc_zr"])
            # tl_df = signal.decimate(tl_df, step)
            tl_df = [tl_df[i] for i in range(0, len(tl_df), step)]
            l_df = pd.DataFrame(data=np.concatenate([
                [tl_df], [[max_period]*len(tl_df)]],axis=0).transpose(),
                                columns=['operation', 'box'])
            max_period += 1

            datalist.append(data_df)
            # in each period, generate label for each data point
            # fullabel = generate_labels_for_each_point(data_df, l_df)
            fulllabellist.append(l_df)

        datalists = pd.concat(datalist)
        fulllabellists = pd.concat(fulllabellist)

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        X = datalists.values  # remove timestamp column
        y = fulllabellists.values[:, 0:1]
        d = np.full(y.shape, int(domain_dict[domain_idx]), dtype=int)
        period = fulllabellists.values[:, -1:]
        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(domain_idx, X.shape, y.shape, d.shape))

        obj = [(X, y, d, period)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
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

class data_loader_skoda(base_loader):
    def __init__(self, samples, labels, domains, mask_ratio):
        super(data_loader_skoda, self).__init__(samples, labels, domains)
        self.mask_ratio = mask_ratio

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        masked = mask_input_data(self.mask_ratio, sample)

        return sample, target, domain, masked


def pad_similarity_value(x_len, tmp_periods_simi_list):
    tmp_periods_simi_list = np.array(tmp_periods_simi_list)
    for i in range(tmp_periods_simi_list.shape[0]):
        tmp_periods_simi_list[i] = np.pad(tmp_periods_simi_list[i],
                                          (0, x_len[i] - len(tmp_periods_simi_list[i])),
                                          constant_values=0)
    return tmp_periods_simi_list


def prep_domains_skoda_random_motif(args):

    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    user = args.user_name  # u0107
    motif_path = os.path.join(root_path, 'motif_processing',
                              args.dataset, user + '.pkl')

    with open(motif_path, 'rb') as f:
        tmpresult = cp.load(f)
    X_processed, y, motif_simi = tmpresult
    # only use robust motif:
    # motif_simi = motif_simi[-3:, :]
    # motif_simi = motif_simi[:-3, :]
    # X_processed1, y1, motif_simi = tmpresult

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
        y_win_train = np.concatenate(y[:train_n_periods])
        motif_simi = np.concatenate(simi_list[:train_n_periods], axis=1)
        m_win_train = motif_simi.transpose()
        # test set
        x_win_test = np.concatenate(X_processed[train_n_periods:])
        y_win_test = np.concatenate(y[train_n_periods:])
        motif_simi = np.concatenate(simi_list[train_n_periods:], axis=1)
        m_win_test = motif_simi.transpose()
        if args.framework in ['ProposeSSL', 'CNNRNN']:
            args.out_fea = m_win_test.shape[-1]
    else:
        motif_simi_win = motif_simi.transpose()
        x_win = np.concatenate(X_processed)
        y_win = np.concatenate(y)

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
    x_win_trainL, y_win_trainL, m_win_trainL = [],[],[]
    for idx in range(0, x_win_train.shape[0] - args.len_sw - args.step, args.step): #step10
        x_win_trainL.append(x_win_train[idx: idx+args.len_sw, :])
        y_win_trainL.append(y_win_train[idx: idx+args.len_sw, :])
        m_win_trainL.append(m_win_train[idx: idx+args.len_sw, :])
    xlist = np.stack(x_win_trainL, axis=0)  # [B, Len, dim]
    x_win_train = xlist.reshape((xlist.shape[0], xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    y_win_train = np.stack(y_win_trainL, axis=0)
    m_win_train = np.stack(m_win_trainL, axis=0)
    print(" ..after sliding window: train inputs {0}, train targets {1}".
          format(x_win_train.shape, y_win_train.shape))
    #---------for test set, do not overlap----------
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

    train_set_r = data_loader_skoda(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler)

    test_set_r = data_loader_skoda(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False)

    return [train_loader_r], test_loader_r, []


def prep_domains_skoda_period_motif(args):

    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    user = args.user_name  # u0107
    full_motif_path1 = os.path.join(root_path, 'motif_processing', 'save_motif_scores',
                                    args.dataset, 'motif_num40_Hamming_%s_2hands.pkl' % user)

    # only one user
    if os.path.exists(full_motif_path1):
        with open(full_motif_path1, 'rb') as f:
            tmpresult = cp.load(f)
        X_processed, y, _, _, _, motif_similarity_periods_ham, candidate_motif_ids, _ = tmpresult
    else:
        print('please run motif_processing code first.')
        NotImplementedError
        return

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
        combined_train_columns, combined_test_columns = [], []
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

    train_set_r = data_loader_skoda(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler)

    test_set_r = data_loader_skoda(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False)
    print(x_win_train.shape)
    return [train_loader_r], test_loader_r, []


# def prep_domains_skoda_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, if_split_user=False):
#     '''
#     if_split_user: when True, generate a list of dataloaders for each user.
#                    When False, only a tuple of dataloader for all users.
#     '''
#     # 'U0101',
#     source_domain_list = ['U0101', 'U0102',
#                           'U0103', 'U0105', 'U0106', 'U0107', 'U0109',
#                           'U0202', 'U0205',
#                           'U0210']  # remove any user as target domain U0104,U0108,U0110,U0203,U0204,U0207
#
#     if not if_split_user:
#         train_loader_r, val_loader_r, test_loader_r = \
#             prep_domains_openpack_random_single(args,
#                                                 SLIDING_WINDOW_LEN=0,
#                                                 SLIDING_WINDOW_STEP=0,
#                                                 source_domain_list=source_domain_list)
#         return train_loader_r, val_loader_r, test_loader_r
#     else:
#         # create a dict for each user
#         train_dict, val_dict, test_dict = {}, {}, {}
#         for u in source_domain_list:
#             train_loader_r, val_loader_r, test_loader_r = \
#                 prep_domains_openpack_random_single(args,
#                                                     SLIDING_WINDOW_LEN=0,
#                                                     SLIDING_WINDOW_STEP=0,
#                                                     source_domain_list=[u])
#             train_dict[u] = train_loader_r
#             val_dict[u] = val_loader_r
#             test_dict[u] = test_loader_r
#         return train_dict, val_dict, test_dict


def prep_skoda(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, if_split_user=False):
    '''
    if_split_user: when True, generate a list of dataloaders for each user.
                   When False, only a tuple of dataloader for all users.
    '''
    if args.cases == 'random':
        if args.use_motif:
            # 只是dataloader的输出多了一个motif的similarity value
            return prep_domains_skoda_random_motif(args)
        # else:
        #     return prep_domains_skoda_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP, if_split_user)
    elif args.cases == 'period':
        return prep_domains_skoda_period_motif(args)
    else:
        return 'Error! Unknown args.cases!\n'

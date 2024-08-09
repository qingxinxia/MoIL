'''
Data Pre-processing on openpack dataset v3.1.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random
import pickle
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
    # abspath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    # root_path = os.getcwd()
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path,'..'))
    data_dir = os.path.join(root_path, 'data', 'nejiData', 'raw')
    saved_filename = os.path.join(root_path, 'data', 'nejiData',
                                  'neji_wd', 'neji_domain_' + domain_idx + '_wd.data') # "wd": with domain label
    labelpath = os.path.join(data_dir, 'labels.csv')

    if os.path.isfile(os.path.join(data_dir, saved_filename)) == True:
        data = np.load(os.path.join(data_dir, saved_filename), allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        # y = np.vectorize(classesdict.get)(yy)
        d = data[0][2]
        period = data[0][-1]
    else:

        csvlist = ['nejidataCtest.csv']

        datalist, labellist, fulllabellist = [], [], []
        max_period = 1
        for file in csvlist:
            input_path = os.path.join(data_dir, file)
            # print('reading data ...')
            data_df = pd.read_csv(input_path)
            # print('convert data ...')
            data_df = convert_unit_system(args, data_df)

            data_df.columns=["timeP", "acc_xr", "acc_yr", "acc_zr"]
            data_df.dropna(axis=0, how='any', inplace=True)

            #--------label------------
            l_df = pd.read_csv(labelpath)
            # for neji, we use start time to label the boundary of operation
            l_df.columns=['event_type', 'starttime', 'endtime']

            # in each period, generate label for each data point
            # 通过time对齐data和label表
            fullabel, data_df = generate_labels_for_each_point_neji(data_df, l_df)
            fulllabellist.append(fullabel)
            datalist.append(data_df)

        datalists = pd.concat(datalist)
        fulllabellists = pd.concat(fulllabellist)


        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        X = datalists.values[:, 1:]  # remove timestamp column
        y = fulllabellists.values[:, 1:-1]
        d = np.full(y.shape, int(domain_dict[domain_idx]), dtype=int)
        period = fulllabellists.values[:, -1]
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

class data_loader_neji(base_loader):
    def __init__(self, samples, labels, domains, mask_ratio):
        super(data_loader_neji, self).__init__(samples, labels, domains)
        self.mask_ratio = mask_ratio

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = sample
        masked = mask_input_data(self.mask_ratio, sample)

        return sample, target, domain, masked



def prep_domains_neji_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, if_split_user=False):
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))
    full_motif_path1 = os.path.join(root_path, 'motif_processing',
                                    'save_motif_scores', args.dataset,
                                    'motif_num20_Hamming_%s_Rhand.pkl' % args.user_name)

    # test only one user
    with open(full_motif_path1, 'rb') as f:
        (X_processed, y,
         _, _, _,
         motif_similarity_periods_ham, candidate_motif_ids, _) \
            = pickle.load(f)

    # process data
    x_win = np.concatenate(X_processed)
    y_win = np.concatenate(y)
    tmp_columns = []
    # Alg 1
    candidate_motif_ids = calculate_test(args, motif_similarity_periods_ham)
    sortedscore = sorted(candidate_motif_ids.items(), key=itemgetter(1), reverse=True)
    for score in [sortedscore]:
        for motif_idx, _ in score[:args.num_motifs]:  # select topk motifs, k=20
            tmp_periods_simi_list1 = motif_similarity_periods_ham[motif_idx]
            # # algorithm1: mask similarity series
            # tmp_periods_simi_list = mask_simi_series(tmp_periods_simi_list1, motif_idx)
            # algorithm2: set a threshold to the series
            tmp_periods_simi_list = filter_simi_by_threshold(tmp_periods_simi_list1)
            tmp_column = np.concatenate(tmp_periods_simi_list)
            tmp_columns.append(tmp_column)

    motif_simi_win = np.concatenate([tmp_columns], axis=1).transpose()

    # initialize args.out_fea
    if args.framework in ['ProposeSSL', 'CNNRNN']:
        args.out_fea = motif_simi_win.shape[-1]

    # first, split data into training and test sets
    x_win_train, x_win_test, \
    y_win_train, y_win_test, \
    m_win_train, m_win_test = \
        train_test_split(x_win, y_win, motif_simi_win, test_size=args.split_ratio, shuffle=False)  # 0.2

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

    train_set_r = data_loader_neji(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=True, sampler=sampler)

    test_set_r = data_loader_neji(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False)

    # return [train_loader_r], train_loader_r, []
    return [train_loader_r], test_loader_r, []




def prep_domains_neji_random_single(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, source_domain_list=None):

    if source_domain_list is None:
        source_domain_list = ['U0101']
    domain_dict = {'test': 0}

    classes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 8100]
    classesdict = dict(zip(classes, range(len(classes))))

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        # print('source_domain:', source_domain)
        x_win, y_win, d_win, _ = load_domain_data(args, source_domain, domain_dict, classesdict)

        # remove 尾数
        moredata = x_win.shape[0] % args.len_sw
        x_win = x_win[:-moredata, :]
        y_win = y_win[:-moredata, :]
        d_win = d_win[:-moredata, :]
        # n_channel should be 9, H: 1, W:128
        # orig = x_win  # segmentation no problem
        # new = x_win.reshape(-1,args.len_sw, args.n_feature)
        x_win = np.transpose(x_win.reshape((-1, 1, args.len_sw, args.n_feature)), (0, 2, 1, 3))
        y_win = y_win.reshape((-1, args.len_sw, y_win.shape[-1]))
        d_win = d_win.reshape((-1, args.len_sw, d_win.shape[-1]))
        # print(" ..after sliding window: inputs {0}, targets {1}".format(x_win.shape, y_win.shape))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio)

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()
    if y_win_train.ndim == 3:
        sample_weights = get_sample_weights(y_win_train[:,0,0], weights)
    else:
        sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=np.zeros(x_win_train.shape[-1]), std=np.ones(x_win_train.shape[-1]))
        # transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_set_r = data_loader_neji(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_neji(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_neji(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r



def prep_neji(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, if_split_user=False):
    '''
    if_split_user: when True, generate a list of dataloaders for each user.
                   When False, only a tuple of dataloader for all users.
    '''
    if args.cases == 'random':
        return prep_domains_neji_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP, if_split_user)
    # elif args.cases == 'subject':
    #     return prep_domains_ucihar_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    # elif args.cases == 'subject_large':
    #     return prep_domains_ucihar_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'


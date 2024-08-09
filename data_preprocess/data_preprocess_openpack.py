'''
Data Pre-processing on openpack dataset v3.1.

'''

import os
# import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from data_preprocess.base_loader import DataLoader
from torchvision import transforms
import torch
import pickle
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from sklearn.model_selection import train_test_split
from data_preprocess.base_loader import *
import pandas as pd
import matplotlib.pyplot as plt

import gc
import random
from operator import itemgetter
from utils import mask_simi_series, filter_simi_by_threshold
from motif_processing.peak_func import calculate_test
from scipy import signal
from datetime import datetime
import math

# ---------------------------自定义dataloader--------------------------------------
#
# import torch
# from torch.utils.data.sampler import WeightedRandomSampler
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

# Sample dataset
class MyDataset:
    def __init__(self, data, targets, m_win_train, mask_ratio):
        self.data = data
        self.targets = targets
        self.m_win_train = m_win_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print(index)
        return torch.tensor(self.data[index]), torch.tensor(self.targets[index]), torch.tensor(self.m_win_train[index])


# Function to load a batch of data
def load_batch(dataset, batch_indices):
    batch_data = [dataset[idx] for idx in batch_indices]
    data, targets, m_win_train = zip(*batch_data)
    data = torch.stack(data)
    targets = torch.stack(targets)
    m_win_train = torch.stack(m_win_train)
    return data, targets, m_win_train

# Function to load a batch of data
def load_batch_worker(dataset, indices):
    batch_data = [dataset[i][0] for i in indices]
    batch_targets = [dataset[i][1] for i in indices]
    batch_m_win_train = [dataset[i][2] for i in indices]
    data = torch.stack(batch_data)
    targets = torch.stack(batch_targets)
    m_win_train = torch.stack(batch_m_win_train)
    return data, targets, m_win_train
    # return torch.stack(data), torch.stack(targets), torch.stack(m_win_train)


# Custom DataLoader
class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, sampler=None, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sampler = sampler
        self.num_workers = num_workers
        self.indices = list(range(len(dataset)))
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.sampler:
            self.indices = list(self.sampler)
        elif self.shuffle:
            self.indices = torch.randperm(len(self.dataset)).tolist()
        else:
            self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            self.on_epoch_end()
            raise StopIteration

        end_index = self.current_index + self.batch_size
        if self.drop_last and end_index > len(self.indices):
            self.on_epoch_end()
            raise StopIteration

        batch_indices = self.indices[self.current_index:end_index]
        if len(batch_indices) < self.batch_size and self.drop_last:
            self.on_epoch_end()
            raise StopIteration

        self.current_index = end_index

        data, targets, m_win_train = load_batch_worker(self.dataset, batch_indices)


        return data, targets, m_win_train, 0


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
    input_format = "%Y-%m-%d %H:%M:%S"  #2021-11-16 12:17:55.627000+09:00

    # abspath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    # root_path = os.getcwd()
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))
    data_dir = os.path.join(root_path, 'data', 'openpackDataset', 'v_3.1')
    # data_dir = os.path.join(root_path, 'data', 'skodaDataset', 'v_3.1')
    saved_filename = os.path.join(root_path, 'data', 'openpackDataset',
                                  'openpack_wd',
                                  'openpack_domain_' + domain_idx + '_s1_wd.data')  # "wd": with domain label

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
        datapathL = os.path.join(str_folder, domain_idx, 'atr', 'atr01')
        datapathR = os.path.join(str_folder, domain_idx, 'atr', 'atr02')
        labelpath = os.path.join(str_folder, domain_idx, 'annotation', 'openpack-operations')
        # labelpath = os.path.join(str_folder, domain_idx, 'annotation', 'activity-1s')
        csvlist = ['S0100.csv']  # if s1, add name to wd.pickle file
        # csvlist = ['S0100.csv', 'S0200.csv', 'S0300.csv', 'S0400.csv', 'S0500.csv']

        datalist, labellist, fulllabellist = [], [], []
        max_period = 1
        for file in csvlist:
            input_path = os.path.join(datapathL, file)
            # print('reading data ...')
            data_dfL = pd.read_csv(input_path)
            # print('convert data ...')
            # data_dfL = convert_unit_system(args, data_df)
            # right is the same as left
            input_path = os.path.join(datapathR, file)
            data_dfR = pd.read_csv(input_path)
            # data_dfR = convert_unit_system(args, data_df)
            data_df = pd.concat([data_dfL[["unixtime", "acc_x", "acc_y", "acc_z"]],
                                 data_dfR[["acc_x", "acc_y", "acc_z"]]], axis=1)
            data_df.columns = ["time", "acc_xl", "acc_yl", "acc_zl", "acc_xr", "acc_yr", "acc_zr"]
            data_df.dropna(axis=0, how='any', inplace=True)
            # datalist.append(data_df)  # 在生成label时需要去头去尾，所以要更新

            # --------label------------
            input_path = os.path.join(labelpath, file)
            l_df = pd.read_csv(input_path)
            # # for version 3.1
            # l_df = l_df[['unixtime', 'operation', 'box']]
            # l_df.columns = ['time', 'operation', 'box']
            # for version 1, time:2021-11-16 12:17:55.627000+09:00
            l_df = l_df[['start', 'operation', 'box']]
            l_df['time'] = l_df['start'].apply(lambda x: datetime.strptime(x[:19], input_format).timestamp())
            # l_df.columns = ['time', 'operation', 'box']
            l_df = l_df[['time', 'operation', 'box']]

            l_df['box'] = l_df['box'] + max_period
            # todo
            max_period += 100
            labellist.append(l_df)

            # in each period, generate label for each data point
            fullabel, data_clean = generate_labels_for_each_point(data_df, l_df)
            datalist.append(data_clean)
            fulllabellist.append(fullabel)

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

class data_loader_openpack(base_loader):
    def __init__(self, samples, labels, domains, mask_ratio):
        # super(data_loader_openpack, self).__init__()
        super(data_loader_openpack, self).__init__(samples, labels, domains)
        # self.mask_ratio = mask_ratio
        self.samples = samples
        self.labels = labels
        self.domains = domains
        # self.start = 0
        # self.end = len(labels)

    def __getitem__(self, index):
        # print(index)
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        # masked = mask_input_data(self.mask_ratio, sample)

        return sample, target, domain, 0

    def __iter__(self):
        return iter(range(self.start, self.end))

def pad_similarity_value(x_len, tmp_periods_simi_list):
    tmp_periods_simi_list = np.array(tmp_periods_simi_list)
    for i in range(tmp_periods_simi_list.shape[0]):
        tmp_periods_simi_list[i] = np.pad(tmp_periods_simi_list[i],
                                          (0, x_len[i] - len(tmp_periods_simi_list[i])),
                                          constant_values=0)
    return tmp_periods_simi_list


def prep_domains_openpack_period_motif(args, if_split_user):
    # 在这个函数内部指定用n个period训练，其他测试。n=1,2,3,4,5?
    # 当前，这个函数同时处理MoIL和supervised模型加载。MoIL要全部data预训练，supervised不是。
    # dataloader格式不要改变

    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    user = args.user_name  # u0107
    full_motif_path1 = os.path.join(root_path, 'motif_processing', 'save_motif_scores',
                                    args.dataset, 'motif_num20_Hamming_%s_s1_2hands.pkl' % user)

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
    if moredata == 0:
        x_win = x_win_test
        y_win = y_win_test
        m_win = m_win_test
    else:
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

    if args.framework !="CNNRNN":
        # 不读取similarity series
        m_win_train = y_win_train
        m_win_test = y_win_test
    train_set_r = data_loader_openpack(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler, num_workers=8, pin_memory=True)

    test_set_r = data_loader_openpack(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    return [train_loader_r], test_loader_r, []

def replace_values_in_ndarray(ndarray, replace_dict):
    for key, value in replace_dict.items():
        ndarray[ndarray == key] = value
    return ndarray

def seg_simi_period(motif_simi, y):
    motif_simi_period = []
    start, end = 0, 0
    for yp in y:
        end = start + len(yp)
        motif_simi_period.append(motif_simi[:, start: end].transpose())
        start = end
    return motif_simi_period

def prep_domains_openpack_random_motif(args):

    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    #algorithm
    # user = args.user_name + '_newRef_NOmask' # u0107
    # motif_path = os.path.join(root_path, 'motif_processing',
    #                                 args.dataset, user+'.pkl')
    motif_path = os.path.join(root_path, 'data', 'openpackDataset', 'best_motifs_1',
                              args.user_name + '_xys.pkl')

    with open(motif_path, 'rb') as f:
        tmpresult = cp.load(f)
    X_processed, orig_y, motif_simi = tmpresult

    classes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 8100]
    classesdict = dict(zip(classes, range(len(classes))))
    # Apply the replacement to each ndarray in the list
    y = [replace_values_in_ndarray(arr, classesdict) for arr in orig_y]
    # 切分motif_simi
    motif_simi_period = seg_simi_period(motif_simi, y)
    num_p = len(y)
    if args.split_ratio != 1:
        # 随机选择该用户的一部分数据作为训练集
        train_periods = np.sort(random.sample(np.arange(0, num_p, 1).tolist(), int(num_p * args.split_ratio)))

        xtrain, ytrain, strain, xtest, ytest, stest = [], [], [], [], [], []
        for idx in range(num_p):
            if idx in train_periods:
                xtrain.append(X_processed[idx])
                ytrain.append(y[idx])
                strain.append(motif_simi_period[idx])
            else:
                xtest.append(X_processed[idx])
                ytest.append(y[idx])
                stest.append(motif_simi_period[idx])
    else:  # 训练集和测试集一样，所有数据用于训练
        train_periods = np.arange(0, num_p, 1).tolist()
        xtrain, ytrain, strain, xtest, ytest, stest = [], [], [], [], [], []
        for idx in train_periods:
            xtrain.append(X_processed[idx])
            ytrain.append(y[idx])
            strain.append(motif_simi_period[idx])
            xtest.append(X_processed[idx])
            ytest.append(y[idx])
            stest.append(motif_simi_period[idx])
    x_win_train = np.concatenate(xtrain).astype(float)
    y_win_train = np.concatenate(ytrain).astype(int)
    m_win_train = np.concatenate(strain).astype(float)
    # test set
    x_win_test = np.concatenate(xtest).astype(float)
    y_win_test = np.concatenate(ytest).astype(int)
    m_win_test = np.concatenate(stest).astype(float)

    if args.framework in ['ProposeSSL', 'CNNRNN']:
        args.out_fea = m_win_test.shape[-1]

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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)

    if args.framework !="CNNRNN":
        # 不读取similarity series
        m_win_train = y_win_train
        m_win_test = y_win_test

    train_set_r = data_loader_openpack(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False,
                                sampler=sampler, num_workers=0, pin_memory=True)

    test_set_r = data_loader_openpack(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    return [train_loader_r], test_loader_r, []

def prep_domains_openpack_user(args):
    '''
    对同一dataset中的users，leave one user out
    '''
    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))
    # motif_root_path = os.path.join(root_path, 'data', 'openpackDataset', 'best_motifs_1')
    # user = args.user_name + '_newRef_NOmask'
    # motif_root_path = os.path.join(root_path, 'motif_processing', args.dataset) #, user + '.pkl')
    motif_root_path = os.path.join(root_path, 'data', 'openpackDataset', 'best_motifs_1') #, user + '.pkl')

    classes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 8100]
    classesdict = dict(zip(classes, range(len(classes))))

    xtrain, ytrain, strain, xtest, ytest, stest = [], [], [], [], [], []
    # all users paths
    user_paths = os.listdir(motif_root_path)
    for upath in user_paths:
        if True:
        # if '_newRef_NOmask' in upath:
            with open(os.path.join(motif_root_path, upath), 'rb') as f:
                tmpresult = cp.load(f)
            X_processed, orig_y, motif_simi_orig = tmpresult
            motif_simi = motif_simi_orig.transpose()
            y = [replace_values_in_ndarray(arr, classesdict) for arr in orig_y]
            if args.user_name.upper() not in upath:
                xtrain.append(np.concatenate(X_processed).astype(float))
                ytrain.append(np.concatenate(y).astype(int))
                strain.append(motif_simi)
            else:
                xtest.append(np.concatenate(X_processed).astype(float))
                ytest.append(np.concatenate(y).astype(int))
                stest.append(motif_simi)

    x_win_train = np.concatenate(xtrain).astype(float)
    y_win_train = np.concatenate(ytrain).astype(int)
    m_win_train = np.concatenate(strain).astype(float)
    # test set
    x_win_test = np.concatenate(xtest).astype(float)
    y_win_test = np.concatenate(ytest).astype(int)
    m_win_test = np.concatenate(stest).astype(float)


    if args.framework in ['ProposeSSL', 'CNNRNN']:
        args.out_fea = m_win_test.shape[-1]

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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights,
                                                             num_samples=len(sample_weights),
                                                             replacement=True)

    if args.framework !="CNNRNN":
        # 不读取similarity series
        m_win_train = y_win_train
        m_win_test = y_win_test
    # mask参数暂存，原始数据不处理
    train_set_r = data_loader_openpack(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False,
                                sampler=sampler, prefetch_factor=4, num_workers=4,
                                pin_memory=True, worker_init_fn=worker_init_fn)

    test_set_r = data_loader_openpack(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False, prefetch_factor=4,
                               num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    # train_set_r = MyDataset(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    # train_loader_r = CustomDataLoader(train_set_r, batch_size=args.batch_size,
    #                                   shuffle=False, drop_last=False, sampler=sampler, num_workers=0)
    # # train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
    # #                                 shuffle=False, drop_last=False,
    # #                                 sampler=sampler, pin_memory=True)
    #
    # test_set_r = MyDataset(x_win_test, y_win_test, m_win_train, args.mask_ratio)
    # test_loader_r = CustomDataLoader(test_set_r, batch_size=args.batch_size,
    #                                  shuffle=False, drop_last=False, num_workers=0)
    # # test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
    # #                                  shuffle=False, drop_last=False, num_workers=0)

    return [train_loader_r], test_loader_r, []



def prep_domains_openpack_random_motif_old(args):

    rroot_path, _ = os.path.split(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(rroot_path, '..'))

    # # user = args.user_name  # u0107
    # full_motif_path1 = os.path.join(root_path, 'motif_processing', 'save_motif_scores',
    #                                 args.dataset, 'motif_num20_Hamming_%s_s1_2hands.pkl' % user)
    #
    # # only one user
    # if os.path.exists(full_motif_path1):
    #     with open(full_motif_path1, 'rb') as f:
    #         tmpresult = cp.load(f)
    #     X_processed, y, _, _, _, motif_similarity_periods_ham, candidate_motif_ids, _ = tmpresult
    # else:
    #     print('please run motif_processing code first.')
    #     NotImplementedError
    #     return
    # x_p: list of sensor data (processed by standarized) of very periods
    # y: activity label
    # motif_simi: dictionary, key-motif index at period1, list of value-similarity value of each period
    # motif_variance_score: dictionary of motif and variance score
    # motif_score: top-k of motif_variance_s

    #algorithm 0605
    # user = args.user_name + '_newRef_mask' # u0107
    user = args.user_name + '_newRef_NOmask' # u0107
    # user = args.user_name + '_newRef_random' # u0107
    # user = args.user_name
    motif_path = os.path.join(root_path, 'motif_processing',
                                    args.dataset, user+'.pkl')

    with open(motif_path, 'rb') as f:
        tmpresult = cp.load(f)
    X_processed, y, motif_simi = tmpresult

    # only use robust motif:
    # motif_simi = motif_simi[:-3,:]
    # X_processed1, y1, motif_simi = tmpresult
    # motif_simi = motif_simi[:-3, :]  # without reference motifs
    #
    # 选个好的period
    # 重新给simi series装入periods中
    if args.user_name in ['U0101','U0105', 'U0104', 'U0109']:
        y1 = y.copy()
        X_processed1 = X_processed
        simi_list = []
        count = 0
        for i in y1:
            length = i.shape[0]
            simi_list.append(motif_simi[:,count:count+length])
            count += length
        # # sequence = [11,12,15,16,17,18,19,14,0,1,2,3,4,5,6,7,8,9,10,13] #52
        # # sequence = [11,12,13,14,16,17,18,19,8,9,6,4,2,3,5,7,10,0,15,1]  #68
        if args.user_name == 'U0101':
            sequence = [11,12,13,14,16,17,18,19,8,9,6,4,2,3,15,1,5,7,10,0]  # U0101
        elif args.user_name == 'U0105':
            sequence = [8,9,6,4,2,3,5,7,10,0,15,1,11,12,13,14,16,17,18,19]  # u0105
        elif args.user_name == 'U0104':
            sequence = [8,9,6,10,0,15,1,11,12,13,14,16,17,18,19,4,2,3,5,7]  # u0105
        elif args.user_name == 'U0109':
            sequence = [8,15,9,6,10,2,3,0,4,7,12,1,5,13,14,11]  # u0105
        else:
            print('data_process_openpack, check sequence')
        X_processed = [X_processed1[i] for i in sequence]
        y = [y1[i] for i in sequence]
        simi_list1 = [simi_list[i] for i in sequence]
        # motif_simi = np.concatenate(simi_list1, axis=1)
        motif_simi_all = np.concatenate(simi_list1, axis=1)
        # motif_simi = motif_simi_all[:-3, :]  # without reference motifs
        # motif_simi = motif_simi_all[-3:, :]  # without complementary motifs
        motif_simi = motif_simi_all.copy()   # proposed
        # print('without reference motifs')


    # # check similarity series
    # for i in range(12):
    #     print('max is %f,min is %f'%(max(motif_simi_win[:,i]),min(motif_simi_win[:,i])))
    #     fig, ax = plt.subplots()
    #     ax.plot(motif_simi_win[:,i])
    #     ax_twin = ax.twinx()
    #     ax_twin.plot(y_win[:, 0], linewidth=0.3, color='black')
    #     plt.grid()
    #     plt.show()
    #     plt.close()

    # motif_similarity_periods, motif_score
    # 读取pickle数据后，将simi series和score挑选出来
    # tmp_columns = []
    # # Alg 1, proposed, ???????????????????check scores
    # # 选择motif window size为45，motif id为偶数
    # motif_similarity_periods_ham_all = motif_similarity_periods_ham
    # motif_similarity_periods_ham = {}
    # motif_ids = list(motif_similarity_periods_ham_all.keys())
    # for s in motif_ids:
    #     if int(s) % 2 == 0:
    #         motif_similarity_periods_ham[s] = motif_similarity_periods_ham_all[s]
    # # ----
    # # 选择motif window size为180，motif id为ji数
    # # motif_similarity_periods_ham_all = motif_similarity_periods_ham
    # # motif_similarity_periods_ham = {}
    # motif_ids = list(motif_similarity_periods_ham_all.keys())
    # for s in motif_ids:
    #     if int(s) % 2 == 1:
    #         motif_similarity_periods_ham[s] = motif_similarity_periods_ham_all[s]
    #
    # candidate_motif_ids = calculate_test(args, motif_similarity_periods_ham)
    # sortedscore = sorted(candidate_motif_ids.items(), key=itemgetter(1), reverse=True)
    # for score in [sortedscore]:
    #     for motif_idx, _ in score[:args.num_motifs]:  # select topk motifs, k=20
    #         tmp_periods_simi_list1 = motif_similarity_periods_ham[motif_idx]
    #         # algorithm1: mask similarity series
    #         tmp_periods_simi_list = mask_simi_series(tmp_periods_simi_list1, motif_idx)
    #         # # algorithm2: set a threshold to the series
    #         # tmp_periods_simi_list = filter_simi_by_threshold(tmp_periods_simi_list1)
    #         tmp_column = np.concatenate(tmp_periods_simi_list)
    #         tmp_columns.append(tmp_column)
    # motif_simi_win = np.concatenate([tmp_columns], axis=1).transpose()

    # # Alg 2: random select motifs
    # candidate_motif_ids = list(motif_similarity_periods_ham.keys())
    # candidate_motif_ids.sort()
    # motif_scores = random.sample(candidate_motif_ids, args.num_motifs)
    # for motif_idx in motif_scores:  # select topk motifs, k=20
    #     tmp_periods_simi_list1 = motif_similarity_periods_ham[motif_idx]
    #     # algorithm1: mask similarity series
    #     tmp_periods_simi_list = mask_simi_series(tmp_periods_simi_list1, motif_idx)
    #
    #     tmp_column = np.concatenate(tmp_periods_simi_list)
    #     tmp_columns.append(tmp_column)


    # initialize args.out_fea
    if args.split_ratio < 0:  # train with a limited periods
        simi_list = []
        count = 0
        for i in y:
            length = i.shape[0]
            simi_list.append(motif_simi[:, count:count + length])
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
        y_win = np.concatenate(y)

        if args.framework in ['ProposeSSL', 'CNNRNN']:
            args.out_fea = motif_simi_win.shape[-1]

        # first, split data into training and test sets
        if args.split_ratio == 0:  # use all data to pretrain model
            # # use all users of the dataset
            # x_all, y_all, simi_all = [], [], []
            # for user in ['U0101', 'U0102', 'U0103','U0106', 'U0107', 'U0202']:
            #     motif_path = os.path.join(root_path, 'motif_processing', args.dataset, user + '.pkl')
            #     with open(motif_path, 'rb') as f:
            #         tmpresult = cp.load(f)
            #     X_processed, y, motif_simi = tmpresult
            #     x_all.extend(X_processed)
            #     y_all.extend(y)
            #     simi_all.append(motif_simi)
            # x_win = np.concatenate(x_all)
            # x_win = np.array(x_win, dtype=float)
            # y_win = np.concatenate(y_all)
            # motif_simi_win = np.concatenate(simi_all, axis=1, dtype=float)
            # motif_simi_win = motif_simi_win.transpose()

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
                train_test_split(x_win, y_win, motif_simi_win, test_size=args.split_ratio, shuffle=False)  # 0.2


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

    train_set_r = data_loader_openpack(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler, num_workers=8, pin_memory=True)

    test_set_r = data_loader_openpack(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False, num_workers=8, pin_memory=True)

    return [train_loader_r], test_loader_r, []


def prep_domains_openpack_random(args, if_split_user=False):
    '''
    if_split_user: when True, generate a list of dataloaders for each user.
                   When False, only a tuple of dataloader for all users.
    '''
    # 'U0101',
    source_domain_list = ['U0101', 'U0102',
                          'U0103', 'U0105', 'U0106', 'U0107', 'U0109',
                          'U0202', 'U0205',
                          'U0210']  # remove any user as target domain U0104,U0108,U0110,U0203,U0204,U0207

    if not if_split_user:
        train_loader_r, val_loader_r, test_loader_r = \
            prep_domains_openpack_random_single(args,
                                                SLIDING_WINDOW_LEN=0,
                                                SLIDING_WINDOW_STEP=0,
                                                source_domain_list=source_domain_list)
        return train_loader_r, val_loader_r, test_loader_r
    else:
        # create a dict for each user
        train_dict, val_dict, test_dict = {}, {}, {}
        for u in source_domain_list:
            train_loader_r, val_loader_r, test_loader_r = \
                prep_domains_openpack_random_single(args,
                                                    SLIDING_WINDOW_LEN=0,
                                                    SLIDING_WINDOW_STEP=0,
                                                    source_domain_list=[u])
            train_dict[u] = train_loader_r
            val_dict[u] = val_loader_r
            test_dict[u] = test_loader_r
        return train_dict, val_dict, test_dict


def prep_domains_openpack_random_single(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0, source_domain_list=None):
    if source_domain_list is None:
        source_domain_list = ['U0101']

    domain_dict = {'U0101': 1, 'U0102': 2, 'U0103': 3, 'U0104': 4, 'U0105': 5,
                   'U0106': 6, 'U0107': 7, 'U0108': 8, 'U0109': 9,
                   'U0202': 12, 'U0205': 15, 'U0210': 20}

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
    d_win_train, d_win_val, d_win_test = train_test_val_split(x_win_all, y_win_all, d_win_all,
                                                              split_ratio=args.split_ratio)

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=np.zeros(x_win_train.shape[-1]), std=np.ones(x_win_train.shape[-1]))
        # transforms.Normalize(mean=(0, 0, 0, 0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1, 1, 1, 1))
    ])
    train_set_r = data_loader_openpack(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    val_set_r = data_loader_openpack(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)
    test_set_r = data_loader_openpack(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_openpack(args, if_split_user=False):
    '''
    if_split_user: when True, generate a list of dataloaders for each user.
                   When False, only a tuple of dataloader for all users.
    '''
    if args.cases == 'random':
        if args.use_motif:
            # 只是dataloader的输出多了一个motif的similarity value
            return prep_domains_openpack_random_motif(args)
        else:
            return prep_domains_openpack_random(args, if_split_user)
    elif args.cases == 'period':
        # train and test sets divided by periods
        return prep_domains_openpack_period_motif(args, if_split_user)
    elif args.cases == 'user':
        return prep_domains_openpack_user(args)

    else:
        return 'Error! Unknown args.cases!\n'

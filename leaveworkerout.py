import copy
import argparse

from trainer_CNNRNN import *
from tensorboardX import SummaryWriter
from data_preprocess.data_preprocess_openpack import data_loader_openpack
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from torch.utils.data import Dataset, DataLoader

#
# class Classifier_cls(nn.Module):
#     def __init__(self, args):
#         super(Classifier_cls, self).__init__()
#
#         if args.framework == 'multi':
#             indim = 96
#             bn = 183 * 2
#         elif args.framework == 'unet':
#             indim = 64
#             bn = 180 * 2
#         else:
#             indim = 128
#             # bn = 180*2
#             bn = 1080  # 跟input 长度一样
#
#         self.linear1 = nn.Linear(indim, 256)
#         self.bn1 = nn.BatchNorm1d(bn)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(256, 128)
#         self.sigmoid = nn.Sigmoid()
#         self.linear3 = nn.Linear(128, args.n_class)
#         # self.fc = nn.Linear(128, 6)  # 6 means dimension of raw sensor data
#         self.fc = nn.Linear(128, 3)  # 3 means dimension of raw sensor data
#
#         self.conv = nn.Conv2d(indim, indim, (1, 5),
#                               bias=False, padding=(0, 5 // 2))
#         self.BN = nn.BatchNorm2d(indim)
#         self.conv1 = nn.Conv2d(indim, indim, (1, 5),
#                                bias=False, padding=(0, 5 // 2))
#         self.BN1 = nn.BatchNorm2d(64)
#         self.activation = nn.PReLU()
#
#     def forward(self, x):  # input shape(batch,len,dim)
#         x1 = x.unsqueeze(2)
#         x1 = x1.permute(0, 3, 2, 1)
#         x2 = self.activation(self.BN(self.conv(x1)))
#         x2 = x2.squeeze(2)
#         x2 = x2.permute(0, 2, 1)
#         x3 = self.relu(self.bn1(self.linear1(x2)))  # input shape(batch,len,dim)
#         x4 = self.sigmoid(self.bn1(self.linear2(x3)))  # input shape(batch,len,dim)
#         clsresult = self.sigmoid(self.bn1(self.linear3(x4)))  # input shape(batch,len,dim)
#         return clsresult, _

class Classifier_cls(nn.Module):
    def __init__(self, args):
        super(Classifier_cls, self).__init__()

        if args.framework == 'multi':
            indim = 96
            # bn = 183*2
            bn = 1083
        elif args.framework == 'unet':
            indim = 64
            bn = 180*2
        else:
            indim = 128
            # bn = 180*2
            bn = 1080  # 跟input 长度一样

        self.linear1 = nn.Linear(indim, 256)
        self.bn1 = nn.BatchNorm1d(bn)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.sigmoid = nn.Sigmoid()
        self.linear3 = nn.Linear(128, args.n_class)
        # self.fc = nn.Linear(128, 6)  # 6 means dimension of raw sensor data
        self.fc = nn.Linear(128, 3)  # 6 means dimension of raw sensor data


        self.conv = nn.Conv2d(indim, indim, (1, 5),
                              bias=False, padding=(0,5//2))
        self.BN = nn.BatchNorm2d(indim)
        self.conv1 = nn.Conv2d(indim, indim, (1, 5),
                              bias=False, padding=(0, 5 // 2))
        self.BN1 = nn.BatchNorm2d(64)
        self.activation = nn.PReLU()


    def forward(self, x):  # input shape(batch,len,dim)
        x1 = x.unsqueeze(2)
        x1 = x1.permute(0, 3, 2, 1)
        x2 = self.activation(self.BN(self.conv(x1)))
        # x2 = self.activation(self.BN(self.conv1(x2)))
        x2 = x2.squeeze(2)
        x2 = x2.permute(0,2,1)
        # out = self.classifier(x2)  # input shape(batch,len,dim)
        x3 = self.relu(self.bn1(self.linear1(x2)))  # input shape(batch,len,dim)
        x4 = self.sigmoid(self.bn1(self.linear2(x3)))  # input shape(batch,len,dim)
        clsresult = self.sigmoid(self.bn1(self.linear3(x4)))  # input shape(batch,len,dim)
        # senresult = self.fc(x4)
        # return clsresult, senresult
        return clsresult, _  #clsresult:128,1080,12

# -------------arguments-----------------------------
parser = argparse.ArgumentParser(description='argument setting of network')

# parser.add_argument('--cuda', default='0', type=str, help='cuda device ID，0/1')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
# hyperparameter

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# dataset
parser.add_argument('--n_feature', type=int, default=6, help='name of feature dimension')
parser.add_argument('--out_fea', type=int, default=96,
                    help='name of output backbone feature dimension, init dim=input feature dim(n_feature)')
parser.add_argument('--len_sw', type=int, default=1080, help='length of sliding window')
parser.add_argument('--step', type=int, default=90, help='step of sliding window')
parser.add_argument('--n_class', type=int, default=12, help='number of class')
parser.add_argument('--cases', type=str, default='random',
                    choices=['random', 'period'],
                    help='name of scenarios, cross_device and joint_device only applicable when hhar is used')
parser.add_argument('--n_periods', type=int, default=0,
                    help='number of periods for training')

parser.add_argument('--target_domain', type=str, default='1', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')

# augmentation
parser.add_argument('--aug1', type=str, default='t_warp',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample',
                             'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'],
                    help='the type of augmentation transformation')
parser.add_argument('--aug2', type=str, default='perm',
                    choices=['na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample',
                             'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f'],
                    help='the type of augmentation transformation')

parser.add_argument('--p', type=int, default=128,
                    help='byol: projector size, simsiam: projector output size, simclr: projector output size')
parser.add_argument('--phid', type=int, default=128,
                    help='byol: projector hidden size, simsiam: predictor hidden size, simclr: na')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# masked reconstruct transformer
parser.add_argument('--mask_ratio', type=float, default=0.00, help='if 0, no mask for input data')

# byol
parser.add_argument('--lr_mul', type=float, default=10.0,
                    help='lr multiplier for the second optimizer when training byol')
parser.add_argument('--EMA', type=float, default=0.996, help='exponential moving average parameter')

# nnclr
parser.add_argument('--mmb_size', type=int, default=1024, help='maximum size of NNCLR support set')

# TS-TCC
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for temporal contrastive loss')
parser.add_argument('--lambda2', type=float, default=1.0,
                    help='weight for contextual contrastive loss, also used as the weight for reconstruction loss when AE or CAE being backbone network')
parser.add_argument('--temp_unit', type=str, default='tsfm', choices=['tsfm', 'lstm', 'blstm', 'gru', 'bgru'],
                    help='temporal unit in the TS-TCC')

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'],
                    help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')

# plot
parser.add_argument('--plt', type=bool, default=True, help='if or not to plot results')

parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of training epochs')

parser.add_argument('--n_epoch_supervised', type=int, default=50, help='number of training epochs')
parser.add_argument('--split_ratio', type=float, default=1,
                    help='split ratio of test: train(1), test(0.0)')
parser.add_argument('--split_ratio_downtask', type=float, default=0.2,
                    help='split ratio of test: test(0.8)')

# framework.
parser.add_argument('--framework', type=str, default='CNNRNN',
                    choices=['CNN_AEframe', 'CNNRNN', 'SSL', 'byol', 'multi',
                             'simsiam', 'simclr', 'ProposeSSL'],
                    help='name of framework')

parser.add_argument('--backbone', type=str, default='CNNRNN',
                    choices=['CNN_AE', 'CNNRNN', 'Transformer', 'CNN'],
                    help='name of backbone network')
# binary is the multi-task model's pretrain loss
parser.add_argument('--criterion', type=str, default='mse',
                    choices=['mse', 'cos_sim', 'NTXent', 'binary'],
                    help='type of loss function for contrastive learning')

# motif related
parser.add_argument('--dataset', type=str, default='openpack',
                    choices=['openpack', 'logi', 'skoda', 'ome'],
                    help='name of dataset')
parser.add_argument('--use_motif', type=bool, default=True,
                    help='if true, use different dataloader')  # 以后去掉这个，dataloader统一

parser.add_argument('--user_name', type=str, default='U0210', help='openpack users')
parser.add_argument('--num_motifs', type=int, default=20,
                    help='segment 1st period into n parts and select one motif per part')
parser.add_argument('--classifierLayer', type=str, default='MLP', choices=['linear', 'MLP'],
                    help='type of classifier layer in downstream task')
parser.add_argument('--seed', type=int, default=26)

# args = parser.parse_args(args=[])


def generate_gaussian_noise(size, mean=0, std_dev=1):
    # noise = np.random.normal(mean, std_dev, size)
    noise = [random.uniform(0,0.1) for _ in range(size)]
    return noise

if __name__ == '__main__':

    # ---------------------parameter settings----------------------------------
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)
    writer = SummaryWriter(log_dir='scalar_xia/%s' % args.dataset)

    torch.cuda.empty_cache()

    # data loader for multiple users - openpack-----------------------------
    all_users = ['U0101', 'U0102', 'U0103', 'U0104', 'U0105', 'U0106', 'U0107',
                 'U0108', 'U0109', 'U0210']
    train_users = all_users.copy()
    train_users.remove(args.user_name)
    test_user = args.user_name + '_newRef_NOmask.pkl'  # u0107
    test_path = os.path.join('motif_processing', args.dataset, test_user)

    # 2 test data
    with open(test_path, 'rb') as f:
        tmpresult = cp.load(f)
    test_X, test_y, test_motif = tmpresult

    # 1 training data
    train_X, train_y, train_motif = [], [], []
    for train_user in train_users:
        tmpuser = train_user + '_newRef_NOmask.pkl'  # u0107
        tmppath = os.path.join('motif_processing', args.dataset, tmpuser)
        with open(tmppath, 'rb') as f:
            tmpresult = cp.load(f)
        X_tmp, ytmp, motif_simitmp = tmpresult
        train_X.extend(X_tmp)
        train_y.extend(ytmp)
        for motifid in range(motif_simitmp.shape[0]):
            noise = generate_gaussian_noise(motif_simitmp.shape[1])  # motif_simitmp.shape=(n,len)
            tmp_motif = motif_simitmp[motifid, :] + noise
            motif_simitmp[motifid, :] = tmp_motif
        train_motif.append(motif_simitmp)
    train_motif_simi = np.concatenate(train_motif, axis=1)

    # 3 all data
    all_X, all_y, all_motif = [], [], []
    for a_user in all_users:
        tmpuser = a_user + '_newRef_NOmask.pkl'  # u0107
        tmppath = os.path.join('motif_processing', args.dataset, tmpuser)
        with open(tmppath, 'rb') as f:
            tmpresult = cp.load(f)
        X_tmp, ytmp, motif_simitmp = tmpresult
        all_X.extend(X_tmp)
        all_y.extend(ytmp)
        all_motif.append(motif_simitmp)
    all_motif_simi = np.concatenate(all_motif, axis=1)

    m_win_train = train_motif_simi.transpose()
    x_win = np.concatenate(train_X)
    x_win_train = np.array(x_win, dtype=float)
    y_win_train = np.concatenate(train_y)

    m_win_test = test_motif.transpose()
    x_win = np.concatenate(test_X)
    x_win_test = np.array(x_win, dtype=float)
    y_win_test = np.concatenate(test_y)

    # then, generate batch of data by overlapping the training set

    # initialize args.out_fea
    if args.framework in ['ProposeSSL', 'CNNRNN']:
        args.out_fea = m_win_test.shape[-1]

    # ---------for training data, overlap----------
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
    if y_win_test.ndim == 3:
        sample_weights = get_sample_weights(y_win_train[:, 0, 0], weights)
        # sample_weights = get_sample_weights(y_win_all[:, 0, 0], weights)
    else:
        sample_weights = get_sample_weights(y_win_train, weights)
        # sample_weights = get_sample_weights(y_win_all, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights),
                                                             replacement=True)

    train_set_r = data_loader_openpack(x_win_train, y_win_train, m_win_train, args.mask_ratio)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, sampler=sampler)

    test_set_r = data_loader_openpack(x_win_test, y_win_test, m_win_test, args.mask_ratio)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size,
                               shuffle=False, drop_last=False)

    train_loaders = [train_loader_r]
    val_loader = test_loader_r

    # -------------upstream task-------------------------------
    model, optimizers, schedulers, criterion, logger, fitlog, _, _, _ = setup(args, DEVICE)

    #     best_pretrain_model = train_SSL(train_loaders,
    #                                     train_loaders[0],  # val_loader,
    #                                     model, logger,
    #                                     fitlog, DEVICE,
    #                                     optimizers, schedulers,
    #                                     criterion, args, writer)

    #     best_pretrain_model = test_SSL(val_loader,  # test_loader,
    #                                    best_pretrain_model,
    #                                    logger, fitlog,
    #                                    DEVICE, criterion, args)

    if args.framework in ['CNN_AEframe', 'CNNRNN', 'SSL', 'ProposeSSL']:
        best_pretrain_model = train_SSL(train_loaders,
                                        train_loaders[0],  # val_loader,
                                        model, logger,
                                        fitlog, DEVICE,
                                        optimizers, schedulers,
                                        criterion, args, writer)

        best_pretrain_model = test_SSL(val_loader,  # test_loader,
                                       best_pretrain_model,
                                       logger, fitlog,
                                       DEVICE, criterion, args)
    elif args.framework == 'multi':  # multi task learning
        best_pretrain_model = train_mul(train_loaders,
                                        train_loaders[0],  # val_loader,
                                        model, logger,
                                        fitlog, DEVICE,
                                        optimizers, schedulers,
                                        criterion, args, writer)
        best_pretrain_model = test_mul(val_loader,  # test_loader,
                                       best_pretrain_model,
                                       logger, fitlog,
                                       DEVICE, criterion, args)
    else:
        best_pretrain_model = train_contrast(train_loaders,
                                             train_loaders[0],
                                             model, logger, fitlog, DEVICE, optimizers,
                                             schedulers, criterion, args, writer)

        best_pretrain_model = test_contrast(val_loader,
                                            best_pretrain_model, logger, fitlog, DEVICE, criterion, args,
                                            writer)

    # ----------------downstream task----------------------
    trained_backbone = lock_backbone(best_pretrain_model, args)
    copy_backbone = copy.copy(trained_backbone)

    # 0529 test classifier
    classifier = Classifier_cls(args)
    classifier = classifier.to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    best_lincls = train_lincls_CNNRNN(best_pretrain_model,
                                      train_loaders, val_loader,
                                      copy_backbone, classifier,
                                      logger, fitlog, DEVICE,
                                      optimizer_cls, criterion_cls,
                                      args, writer, user_name=args.user_name)  # todo, name加到args

    writer.close()
    print('downstream task finished.')

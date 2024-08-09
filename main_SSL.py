#  D:\Code\openpack_box\baselines-SSL\CL-HAR-main\motif_processing\save_motif_scores\openpack
# SSL model: input sensor data, output similarity values


import copy
import argparse

import torch

from trainer_CNNRNN import *
from tensorboardX import SummaryWriter


class Classifier_cls(nn.Module):
    def __init__(self, args):
        super(Classifier_cls, self).__init__()

        if args.framework == 'multi':
            indim = 96
            bn = 183
            # bn = 1083
        elif args.framework == 'unet':
            indim = 64
            bn = 180*2
        # elif args.framework == 'unet':
        #     indim = 90
        #     bn = 1080
        else:
            indim = 128
            bn = 180
            # bn = 1080  # 跟input 长度一样
        # self.classifier = nn.Sequential(
        #     nn.Linear(indim, 256),
        #     nn.BatchNorm1d(bn),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(bn),
        #     nn.Sigmoid(),
        #     nn.Linear(128, args.n_class),
        #     nn.BatchNorm1d(bn),
        #     nn.Sigmoid()
        # )

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


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
# hyperparameter

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--lr_cls', type=float, default=0.1, help='learning rate for linear classifier')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')
parser.add_argument('--scheduler', type=bool, default=True, help='if or not to use a scheduler')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

# dataset
parser.add_argument('--n_feature', type=int, default=3, help='name of feature dimension')
# parser.add_argument('--n_feature', type=int, default=6, help='name of feature dimension')
parser.add_argument('--out_fea', type=int, default=96, help='name of output backbone feature dimension, init dim=input feature dim(n_feature)')
parser.add_argument('--len_sw', type=int, default=180, help='length of sliding window')
# parser.add_argument('--len_sw', type=int, default=180, help='length of sliding window')
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
# parser.add_argument('--plt', type=bool, default=False, help='if or not to plot results')



parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size of training')
# parser.add_argument('--batch_size', type=int, default=200, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of training epochs')
# parser.add_argument('--n_epoch', type=int, default=1, help='number of training epochs')
# parser.add_argument('--n_epoch_supervised', type=int, default=5, help='number of training epochs')
parser.add_argument('--n_epoch_supervised', type=int, default=50, help='number of training epochs')
# parser.add_argument('--split_ratio', type=float, default=0.2,
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='split ratio of test: train(1), test(0.0)')
parser.add_argument('--split_ratio_downtask', type=float,
                    default=0.8,
                    help='split ratio of test: test(0.8)')

# framework.
# CNNRNN:proposed method, SSL:mask reconstruction loss(transformer backbone)
# multi is the multi-task model's framework
parser.add_argument('--framework', type=str, default='byol',
                    choices=['CNN_AEframe', 'CNNRNN', 'SSL', 'byol', 'multi',
                             'simsiam', 'simclr', 'ProposeSSL'],
                    help='name of framework')

# DCL: maybe is deepConvLSTM
# cnn is the multi-task model's backbone
# CNN_AE: 还没有尝试。用openpack试试。
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

# parser.add_argument('--user_name', type=str, default='neji0310', help='neji0309 to neji0310')
# parser.add_argument('--user_name', type=str, default='acc_06_r', help='acc_05_r to acc_06_r')
parser.add_argument('--user_name', type=str, default='U0110', help='openpack users')
# parser.add_argument('--user_name', type=str, default='u1', help='skoda users')
# parser.add_argument('--user_name', type=str, default='omeu1', help='toshiba users')
parser.add_argument('--num_motifs', type=int, default=20, help='segment 1st period into n parts and select one motif per part')
parser.add_argument('--classifierLayer', type=str, default='MLP', choices=['linear', 'MLP']
                    , help='type of classifier layer in downstream task')
parser.add_argument('--seed', type=int, default=31)

if __name__ == '__main__':

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # DEVICE = 'cpu'
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)
    torch.cuda.empty_cache()

    writer = SummaryWriter(log_dir='scalar_xia/%s' % args.dataset)

    train_loaders, val_loader, _ = setup_dataloaders(args, if_split_user=False)

    # model, optimizers, schedulers, criterion, \
    # logger, fitlog, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)
    # model is backbone+framework
    model, optimizers, schedulers, criterion, logger, fitlog, \
    classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)

    # trained = True
    trained = False
    if trained:
        # m = 'pretrain_try_scheduler_CNNRNN_backb_CNNR NN_pretrain_openpack_eps1000_lr0.0001_bs512_dim-pdim128-128_EMA0.996_criterion_mse_lambda1_1.0_lambda2_1.0_tempunit_tsfm_user_U0102999.pt'
        m = 'pretrain_try_scheduler_CNNRNN_backb_CNNRNN_pretrain_skoda_eps1000_lr0.0001_bs512_dim-pdim128-128_EMA0.996_criterion_mse_lambda1_1.0_lambda2_1.0_tempunit_tsfm_user_u1999.pt'
        # m = 'pretrain_try_scheduler_CNNRNN_backb_CNNRNN_pretrain_logi_eps1000_lr0.0001_bs512_dim-pdim128-128_EMA0.996_criterion_mse_lambda1_1.0_lambda2_1.0_tempunit_tsfm_user_acc_03_r999.pt'
        # m = 'pretrain_try_scheduler_CNNRNN_backb_CNNRNN_pretrain_skoda_eps1000_lr0.0001_bs512_dim-pdim128-128_EMA0.996_criterion_mse_lambda1_1.0_lambda2_1.0_tempunit_tsfm_user_u1999.pt'
        # r = r'D:\Code\openpack_box\baselines-SSL\CL-HAR-main\results'
        r = r'D:\Code\openpack_box\baselines-SSL\CL-HAR-main\results'
        path = os.path.join(r, m)
        checkpoint = torch.load(path)
        # checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print('loaded model parameters.')

    if args.framework in ['CNN_AEframe', 'CNNRNN', 'SSL', 'ProposeSSL']:
        best_pretrain_model = train_SSL(train_loaders,
                                        train_loaders[0],  # val_loader,
                                        model, logger,
                                        fitlog, DEVICE,
                                        optimizers, schedulers,
                                        criterion, args, writer)
        # args.cases = 'period'
        # args.split_ratio = 1
        # args.n_periods = 19
        _, val_loader, _ = setup_dataloaders(args, if_split_user=False)
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
        best_pretrain_model = train_contrast(train_loaders, val_loader, model, logger, fitlog, DEVICE, optimizers,
                                             schedulers, criterion, args, writer)

        best_pretrain_model = test_contrast(val_loader, best_pretrain_model, logger, fitlog, DEVICE, criterion, args,
                                            writer)

    ############################################################################################################

    trained_backbone = lock_backbone(best_pretrain_model, args)  # freeze parameters, use encoder

    # # modify dataloader by different subjects (from same person)
    # # because train:val:test sets are randomly splited, so, we don't use overlapping window to generate data.
    # args.split_ratio = args.split_ratio_downtask
    # # # control number of period for training
    # # args.split_ratio = -2
    # train_loaders_dict, val_loader_dict, _ = setup_dataloaders(args)  # if true, return dict
    # # only a single user at scenario-1

    copy_backbone = copy.copy(trained_backbone)
    # loss改了，没有除以nbatch


    # # commit only for multitask
    if 1:
    # if args.framework != 'multi':
        classifier = Classifier_cls(args)
        classifier = classifier.to(DEVICE)
        criterion_cls = nn.CrossEntropyLoss()
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)


    # best_lincls = train_lincls_CNNRNN([val_loader_dict], val_loader_dict,
    best_lincls = train_lincls_CNNRNN(best_pretrain_model,
                                      # train_loaders_dict, val_loader_dict,
                                      train_loaders, val_loader,
                                      copy_backbone, classifier,
                                      logger, fitlog, DEVICE,
                                      optimizer_cls, criterion_cls,
                                      args, writer, user_name=args.user_name)  # todo, name加到args


    writer.close()
    print('downstream task finished.')

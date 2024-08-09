

# pure supervised baseline methods: deepconvlstm, unet, yoshimura?, jaime?

# proposed method: CNNRNN+CNNRNN, number of SSL epochs=500
# classification layer: linear + MLP
# dataset: 8:2, openpack


'''
This code trains the model from scratch with random initial parameters
'''


# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.backbones import *
from models.loss import *
from trainer_CNNRNN import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import pickle
import numpy as np
import os
import logging
import sys
from data_preprocess.data_preprocess_utils import normalize
from scipy import signal
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger
# fitlog.debug()

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID，0/1')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=100, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=50, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3, help='learning rate for linear classifier')

parser.add_argument('--n_feature', type=int, default=6, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=1080, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=12, help='number of class')
parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'subject_large', 'cross_device', 'joint_device'], help='name of scenarios')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,22,23,24,25,29] for shar, '
                                                                   '[a-i] for hhar')


# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# AE & CNN_AE
parser.add_argument('--lambda1', type=float, default=1.0, help='weight for reconstruction loss when backbone in [AE, CNN_AE]')

# # hhar
# parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'], help='data of which device to use (random case); data of which device to be used as training data (cross-device case, data from the other device as test data)')

parser.add_argument('--classifierLayer', type=str, default='MLP', choices=['linear', 'MLP']
                    , help='type of classifier layer in downstream task')

parser.add_argument('--split_ratio', type=float, default=0.92, help='split ratio of test,if0,all data to pretrain')

# dataset
parser.add_argument('--dataset', type=str, default='openpack',
                    choices=['skoda', 'logi', 'openpack', 'ome'], help='name of dataset')
# parser.add_argument('--user_name', type=str, default='ome')
# parser.add_argument('--user_name', type=str, default='u1', help='neji0309 to neji0310')
# parser.add_argument('--user_name', type=str, default='acc_06_r', help='acc_03_r to acc_06_r')
parser.add_argument('--user_name', type=str, default='U0109', help='openpack users')

# useless parameters
parser.add_argument('--step', type=int, default=90, help='step of sliding window')
parser.add_argument('--use_motif', type=bool, default=True,
                    help='if true, use different dataloader')
parser.add_argument('--num_motifs', type=int, default=40, help='segment 1st period into n parts and select one motif per part')
# backbone model
parser.add_argument('--backbone', type=str, default='LSTM', choices=['CNNRNN', 'UNet', 'LSTM', 'CNN'],
                    help='name of framework')
parser.add_argument('--framework', type=str, default='supervised',
                    choices=['supervised'], help='name of framework')
parser.add_argument('--out_fea', type=int, default=20, help='name of output backbone feature dimension, init dim=input feature dim(n_feature)')
parser.add_argument('--mask_ratio', type=float, default=0.00, help='if 0, no mask for input data')
parser.add_argument('--seed', type=int, default=10, help='if 0, no mask for input data')

# create directory for saving and plots
global plot_dir_name
plot_dir_name = 'plot/'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)

def train(args, u, train_loaders, val_loader, model, DEVICE, optimizer, criterion):
    min_val_loss = 1e8
    for epoch in range(args.n_epoch):
        # logger.debug(f'\nEpoch : {epoch}')

        train_loss = 0
        total = 0
        correct = 0
        n_batches = 1
        model.train()
        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain, _) in enumerate(train_loader):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                if args.backbone == 'CNNRNN':
                    _, out = model(sample)
                else:
                    out, _ = model(sample)

                out = out.permute(0, 2, 1)  # if multi dim, B,Ch,Len
                target = target.squeeze(-1)  # 3 dim-> 2 dim
                out = out[:,:,:target.shape[-1]]
                loss = criterion(out, target)

                # if args.backbone == 'CNNRNN':
                #     # print(loss.item(), nn.MSELoss()(sample, x_decoded).item())
                #     loss += nn.MSELoss()(sample, x_decoded) * args.lambda1
                train_loss += loss.item()
                n_batches += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(out.data, 1)
                # total += target.size(0)
                total += target.size(0) * target.size(1)
                correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / total
        if epoch == args.n_epoch -1:
            fitlog.add_loss(train_loss / n_batches, name="Train Loss of %s"%u, step=epoch)
            fitlog.add_metric({"dev": {"Train Acc of %s"%u: acc_train}}, step=epoch)
            logger.debug(f'Train Loss     : {train_loss / n_batches:.4f}\t | \tTrain Accuracy     : {acc_train:2.4f}\n')

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            print('Saving supervised models at {} epoch to {}'.format(epoch, model_dir))
            torch.save({'sup_model_state_dict of %s'%u: model.state_dict(), 'sup_optimizer_state_dict of %s'%u: optimizer.state_dict()}, model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 1
                total = 0
                correct = 0
                trgs = np.array([])
                preds = np.array([])
                # feats, y_true = None, None
                for idx, (sample, target, domain, _) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    if args.backbone == 'CNNRNN':
                        feat, out = model(sample)
                    else:
                        out, _ = model(sample)
                    # if feats == None:
                    #     feats = feat
                    #     y_true = target
                    # else:
                    #     feats = torch.cat((feats, feat), 0)
                    #     y_true = torch.cat((y_true, target), 0)

                    out = out.permute(0, 2, 1)  # if multi dim, B,Ch,Len
                    target = target.squeeze(-1)  # 3 dim-> 2 dim
                    out = out[:,:,:target.shape[-1]]
                    loss = criterion(out, target)

                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    # total += target.size(0)
                    total += target.size(0) * target.size(1)
                    correct += (predicted == target).sum()
                    trgs = np.append(trgs, target.data.cpu().numpy().reshape(-1))
                    preds = np.append(preds, predicted.data.cpu().numpy().reshape(-1))
                acc_val = float(correct) * 100.0 / total
                miF = f1_score(trgs, preds, average='micro') * 100
                maF = f1_score(trgs, preds, average='weighted') * 100
                fitlog.add_loss(val_loss / n_batches, name="Val Loss of %s"%u, step=epoch)
                if epoch == args.n_epoch - 1:
                    fitlog.add_metric({"dev": {"Val Acc of %s"%u: acc_val}}, step=epoch)
                    fitlog.add_best_metric({"dev": {"miF of %s" % miF}})
                    fitlog.add_best_metric({"dev": {"maF of %s" % maF}})
                # if epoch == 1:
                if epoch == args.n_epoch-1:
                    # # plot
                    # tsne(feats.reshape(-1, feats.shape[-1]),
                    #      y_true.reshape(-1),
                    #      str(epoch) + '_' + args.user_name + '_Supertsne.png')

                    print("miF of %s" % miF)
                    print("maF of %s" % maF)
                    logger.debug(f'Val Loss     : {val_loss / n_batches:.4f}\t | \tVal Accuracy     : {acc_val:2.4f}\n')

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    # print('update')
                    # model_dir = save_dir + args.model_name + '.pt'
                    # print('Saving models {} at {} epoch to {}'.format(u, epoch, model_dir))
                    # torch.save({'model_state_dict of %s'%u: model.state_dict(), 'optimizer_state_dict of %s'%u: optimizer.state_dict()}, model_dir)

    return best_model


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # DEVICE = 'cpu'
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('device:', DEVICE, 'dataset:', args.dataset)

    train_loaders, val_loader, _ = setup_dataloaders(args)

    if args.backbone == 'CNNRNN':
        model = CNNRNN(args, n_channels=args.n_feature, n_classes=args.n_class, simi_dim=args.out_fea,
                          conv_kernels=64, kernel_size=5, datalen=args.len_sw, backbone=False)
        # framework = CNNRNN_SSLframe(backbone=backbone)
        # model_test = CNNRNN(n_channels=args.n_feature, n_classes=args.n_class, simi_dim=args.out_fea,
        #                conv_kernels=64, kernel_size=5, datalen=args.len_sw, backbone=False)
    elif args.backbone == 'convlstm':
        model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    #     model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)
    elif args.backbone == 'LSTM':
        model = LSTM(args, n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128)
        # model_test = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=False)
    elif args.backbone == 'CNN':
        model = CNN(n_channels=args.n_feature, n_classes=args.n_class, backbone=False)
    elif args.backbone == 'AE':
        model = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=False)
        # model_test = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128,
        #                 backbone=False)
    elif args.backbone == 'CNN_AE':
        model = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
        # model_test = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=False)
    elif args.backbone == 'Transformer':
        model = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4,
                            heads=4, mlp_dim=64, dropout=0.1, backbone=False)
        # model_test = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128,
        #                          depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)
    else:
        print('backbone not implemented')
        NotImplementedError


    u = args.user_name
    model = model.to(DEVICE)

    args.model_name = 'Supervised_' + args.backbone + '_' + \
                      args.dataset + '_' +args.user_name +'_lr' + str(args.lr) + \
                      '_bs' + str(args.batch_size) + '_eps'+ str(args.n_epoch)\
                      + '_sw' + str(args.len_sw)

    save_dir = 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # log
    if os.path.isdir(args.logdir) == False:
        os.makedirs(args.logdir)

    log_file_name = os.path.join(args.logdir, args.model_name + f".log")
    logger = _logger(log_file_name)
    logger.debug(args)

    # fitlog
    fitlog.set_log_dir(args.logdir)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, args.lr)

    training_start = datetime.now()

    best_model = train(args, u, train_loaders, val_loader, model, DEVICE, optimizer, criterion)

    # model_test.load_state_dict(best_model)
    # model_test = model_test.to(DEVICE)
    # test_loss = test(u, val_loader, model, DEVICE, criterion, plt=False)

    training_end = datetime.now()
    training_time = training_end - training_start
    logger.debug(f"Training time is : {training_time}")

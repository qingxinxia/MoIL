import torch
import torch.nn as nn
import numpy as np
import os
import pickle as cp
from augmentations import gen_aug
from utils import tsne, mds, _logger
import time
from models.frameworks import *
from models.backbones import *
from models.loss import *
from data_preprocess import data_preprocess_openpack
from data_preprocess import data_preprocess_neji
from data_preprocess import data_preprocess_ome
from data_preprocess import data_preprocess_skoda
from data_preprocess import data_preprocess_logi

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
from tqdm import tqdm
import fitlog
from copy import deepcopy
# import matplotlib.pyplot as plt
from collections import Counter
# from pytorchtools import EarlyStopping

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def setup_dataloaders(args, if_split_user=False):
    if args.dataset == 'openpack':
        args.n_feature = 6  # two wrists
        # args.len_sw = 180  # 45  #180  # 30Hz * 10s
        args.n_class = 12
        # if args.cases not in ['subject']:   # use: [random]
        #     args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_openpack.prep_openpack(
            args)
    if args.dataset == 'skoda':
        args.n_feature = 6  # two wrists
        # args.len_sw = 45  # 90  # 30Hz * 10s
        args.n_class = 11
        train_loaders, val_loader, test_loader = data_preprocess_skoda.prep_skoda(
            args,
            SLIDING_WINDOW_LEN=args.len_sw,
            SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
            if_split_user=if_split_user)
    if args.dataset == 'logi':
        args.n_feature = 3  # two wrists
        # args.len_sw = 45  # 90  # 30Hz * 10s
        args.n_class = 10
        train_loaders, val_loader, test_loader = data_preprocess_logi.prep_logi(
            args,
            SLIDING_WINDOW_LEN=args.len_sw,
            SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
            if_split_user=if_split_user)

    if args.dataset == 'ome':
        args.n_feature = 3  # two wrists
        # args.len_sw = 45  # 90  # 30Hz * 10s
        args.n_class = 8
        train_loaders, val_loader, test_loader = data_preprocess_ome.prep_ome(
            args,
            SLIDING_WINDOW_LEN=args.len_sw,
            SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
            if_split_user=if_split_user)

    return train_loaders, val_loader, test_loader


def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_linclf_CNNRNN(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier


def setup_linclf_AE(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier

def setup_model_optm(args, DEVICE, classifier=True):
    # set up backbone network
    if args.backbone == 'FCN':
        backbone = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
    elif args.backbone == 'CNN':
        backbone = CNN(n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
        # backbone = CNN(args, n_channels=args.n_feature, n_classes=args.n_class)
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                                LSTM_units=128, backbone=True)
    elif args.backbone == 'CNNRNN':  # finial layer is different from dcl, use yoshimura code
        backbone = CNNRNN(args, n_channels=args.n_feature, n_classes=args.n_class, simi_dim=args.out_fea,
                          conv_kernels=64, kernel_size=5, datalen=args.len_sw, backbone=True)
    elif args.backbone == 'LSTM':
        backbone = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True)
    elif args.backbone == 'AE':
        backbone = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        backbone = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True)
    elif args.backbone == 'Transformer':
        backbone = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class,
                               dim=128, depth=3, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif args.backbone == 'unet':
        backbone = unet_backbone(args, n_channels=args.n_feature, n_classes=args.n_class,
                     out_channels=128, simi_dim=args.num_motifs, backbone=True)
    else:
        print('not implement backbone')
        NotImplementedError

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:
        model = BYOL(DEVICE, backbone, window_size=args.len_sw, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * args.lr_mul,
                                      weight_decay=args.weight_decay)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr':
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'CNNRNN':
        model = CNNRNN_SSLframe(backbone=backbone)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'unet':
        model = unet_SSLframe(backbone=backbone)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'multi':
        model = CNN_SSLframe(backbone=backbone)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework in ['ProposeSSL', 'SSL']:
        model = Base_SSLframe(args, backbone=backbone)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'CNN_AEframe':  # autoencoder
        model = CNN_AEframe(args, backbone=backbone)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    else:
        print('not implement framework')
        NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
        if args.framework in ['CNN_AEframe', 'CNN', 'CNNRNN', 'unet']:
            classifier = backbone.classifier
            # classifier.weight.data.normal_(mean=0.0, std=0.01)
            # classifier.bias.data.zero_()
            classifier = classifier.to(DEVICE)
            return model, classifier, optimizers
        bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier, optimizers
    else:
        return model, optimizers


def delete_files(args):
    for epoch in range(args.n_epoch):
        model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(model_dir):
            os.remove(model_dir)

        cls_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        if os.path.isfile(cls_dir):
            os.remove(cls_dir)


def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
        args.criterion = 'cos_sim'
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 1.0
        args.lr_mul = 1.0
        args.criterion = 'cos_sim'
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'  # contrastive loss NT-Xent
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        assert args.criterion == 'NTXent'
        assert args.backbone == 'FCN'
        args.weight_decay = 3e-4
    if args.framework in ['CNN_AEframe', 'SSL', 'CNNRNN', 'ProposeSSL', 'unet']:
        args.criterion = 'mse'
    if args.framework == 'multi':
        args.criterion = 'binary'
        args.backbone = 'CNN'

    # model, _, optimizers = setup_model_optm(args, DEVICE, classifier=True)
    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    classifier = classifier.to(DEVICE)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'mse':  # 下面定义autoencoder的criterion为mse loss。
        criterion = nn.MSELoss(reduction='mean')
    elif args.criterion == 'binary':
        criterion = nn.BCELoss()  # pre(B,y), true(B,y)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)

    args.model_name = 'try_scheduler_' + args.framework + '_backb_' + args.backbone +\
                      '_pretrain_' + args.dataset + '_eps' + str(
        args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + \
                      args.criterion + '_lambda1_' + str(args.lambda1) + \
                      '_lambda2_' + str(args.lambda2) + \
                      '_tempunit_' + args.temp_unit + '_user_' + args.user_name
    # + '_aug1' + args.aug1 + '_aug2' + args.aug2

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

    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    return model, optimizers, schedulers, criterion, \
           logger, fitlog, classifier, criterion_cls, optimizer_cls


def calculate_model_loss_CNNRNN(args, sample, target, masked, model,
                                criterion, DEVICE, p=0):
    '''
    model training sample(B,Len,Dim), target(B,L,N12)
    return loss: mse = (np.square(A - B)).mean()
    '''
    masked = masked.to(DEVICE)   # 当前没有mask任何值
    sample = sample.to(DEVICE).float()
    target = target.to(DEVICE).float()

    # masked 为0的位置为需要mask的点
    sample1 = sample * masked.unsqueeze(-1).repeat(1, 1, sample.shape[-1])
    z2 = model(sample1)
    # kl_loss = F.kl_div(q.log(), p)

    # 求loss只取mask部位
    z3 = z2 * ((masked.unsqueeze(-1)).repeat(1, 1, z2.shape[-1]))
    target3 = target * ((masked.unsqueeze(-1)).repeat(1, 1, target.shape[-1]))
    loss = criterion(z3, target3)
    # loss = criterion((torch.exp(z3)), (torch.exp(target3)))
    return loss, z2


def calculate_model_loss(args, sample, target, model,
                         criterion, DEVICE, recon=None, nn_replacer=None):
    '''
    model training
    return loss
    '''
    aug_sample1 = gen_aug(sample, args.aug1)  # t_warp
    aug_sample2 = gen_aug(sample, args.aug2)  # negate

    aug_sample1, aug_sample2, target = \
        aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(DEVICE).long()
    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    # if args.framework in ['CNN_AEframe']:
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)  # BYOL
        if args.framework == 'nnclr':
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)
        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        elif args.criterion == 'NTXent':
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'simclr':
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2)  # deepconvlstm
        loss = criterion(z1, z2)
        if args.backbone in ['AE', 'CNN_AE']:  # autoencoder相关模型都加两个loss
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    return loss


def train_contrast(train_loaders, val_loader, model, logger, fitlog, DEVICE, optimizers, schedulers, criterion, args,
                   writer):
    best_model = copy.deepcopy(model.state_dict())
    # best_model = None
    min_val_loss = 1e8

    for epoch in range(args.n_epoch):
        # logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 1
        model.train()
        start = time.time()
        for i, train_loader in enumerate(train_loaders):
            # for idx, (sample, target, _) in enumerate(train_loader):
            for idx, (sample, target, _, _) in enumerate(train_loader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                # model training here...
                # #sample(batch,len,dim),target(B,L,1),domain---
                # because augmentation requires same data shape, we use:
                dur1 = time.time() - start
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon,
                                            nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if args.framework == 'byol':
                    model.update_moving_average()

                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
        dur = time.time() - start
        print('duration loader %s, one epoch %s'%(dur1, dur))
        fitlog.add_loss(optimizers[0].param_groups[0]['lr'], name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        if (epoch == args.n_epoch - 1):
            # save model
            model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            # torch.save(model.state_dict(), model_dir)
            torch.save({'model_state_dict': model.state_dict()}, model_dir)

            logger.debug(f'Train Loss     : {total_loss / n_batches:.4f}')
            writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)

            fitlog.add_loss(total_loss / n_batches, name="pretrain training loss", step=epoch)

            if args.cases in ['subject', 'subject_large']:
                with torch.no_grad():
                    best_model = copy.deepcopy(model.state_dict())
                    break
            else:
                with torch.no_grad():
                    model.eval()
                    total_loss = 0
                    n_batches = 1
                    # for idx, (sample, target, _) in enumerate(val_loader):
                    for idx, (sample, target, _, _) in enumerate(val_loader):
                        loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon,
                                                    nn_replacer=nn_replacer)
                        total_loss += loss.item()
                        if sample.size(0) != args.batch_size:
                            continue
                        n_batches += 1
                    if total_loss <= min_val_loss:
                        min_val_loss = total_loss
                        best_model = copy.deepcopy(model.state_dict())
                        print('update')
                    logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                    writer.add_scalar('scalar/upValLoss', total_loss / n_batches, epoch)
                    fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)
    return best_model


def calculate_multitask_loss(args, sample, model, criterion, DEVICE):
    # prepare data augmentations (7 in total)
    # each x, generate transf(x), their corresponding labels are (x,False)(t(x),True)
    aug_sample1 = gen_aug(sample, 'noise')
    aug_sample2 = gen_aug(sample, 'scale')
    aug_sample3 = gen_aug(sample, 'rotation')
    aug_sample4 = gen_aug(sample, 'negate')
    aug_sample5 = gen_aug(sample, 'perm')
    aug_sample6 = gen_aug(sample, 't_warp')
    aug_sample7 = gen_aug(sample, 'shuffle')
    sample, aug_sample1, aug_sample2 = \
        sample.to(DEVICE).float(), aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float()
    aug_sample3, aug_sample4 = \
        aug_sample3.to(DEVICE).float(), aug_sample4.to(DEVICE).float()
    aug_sample5, aug_sample6, aug_sample7 = \
        aug_sample5.to(DEVICE).float(), aug_sample6.to(DEVICE).float(), aug_sample7.to(DEVICE).float()

    # training
    xlist = model(sample, [aug_sample1, aug_sample2, aug_sample3, aug_sample4,
                           aug_sample5, aug_sample6, aug_sample7])

    loss = 0
    for x in xlist:
        # calculate binary cross entropy loss for each augment data
        # in x, half label is 0, half is 1. (x,False)(t(x),True)
        len = int(x.shape[0] / 2)
        concate_y = torch.concatenate([torch.zeros(len),
                                       torch.ones(len)]).to(DEVICE)
        tmploss = criterion(x, concate_y.unsqueeze(1))  # x(B,y), target(B,y)
        loss += tmploss

    return loss


def train_mul(train_loaders, val_loader, model,
              logger, fitlog, DEVICE, optimizers,
              schedulers, criterion, args, writer):
    best_model = copy.deepcopy(model.state_dict())
    min_val_loss = 1e8

    for epoch in range(args.n_epoch):
        total_loss = 0
        n_batches = 1
        model.train()
        for i, train_loader in enumerate(tqdm(train_loaders)):
            for idx, (sample, y, simi_value, masked) in enumerate(train_loader):
                for optimizer in optimizers:
                    optimizer.zero_grad()

                loss = calculate_multitask_loss(args, sample,
                                                model, criterion, DEVICE)

                # # L2 norm
                # for param in model.projector.parameters(): loss += 0.0001 * torch.norm(param)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
        fitlog.add_loss(optimizers[0].param_groups[0]['lr'],
                        name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        if (epoch == args.n_epoch - 1):

            # save model
            model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            # torch.save(model.state_dict(), model_dir)
            torch.save({'model_state_dict': model.state_dict()}, model_dir)

            logger.debug(f'Train Loss     : {total_loss / n_batches:.4f}')
            writer.add_scalar('scalar_xia/upTrainLoss', total_loss / n_batches, epoch)

            fitlog.add_loss(total_loss / n_batches, name="pretrain training loss", step=epoch)

            if args.cases in ['subject', 'subject_large']:
                with torch.no_grad():
                    best_model = copy.deepcopy(model.state_dict())
                    break
            else:
                with torch.no_grad():
                    model.eval()
                    total_loss = 0
                    n_batches = 1
                    for idx, (sample, y, simi_value, masked) in enumerate(val_loader):

                        loss = calculate_multitask_loss(args, sample,
                                                        model, criterion, DEVICE)

                        total_loss += loss.item()
                        if sample.size(0) != args.batch_size:
                            continue
                        n_batches += 1
                    if total_loss <= min_val_loss:
                        min_val_loss = total_loss
                        best_model = copy.deepcopy(model.state_dict())
                        print('update')
                    logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                    writer.add_scalar('scalar_xia/upValLoss', total_loss / n_batches, epoch)
                    fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)

    return best_model


def test_mul(test_loader, best_model, logger, fitlog, DEVICE, criterion, args):
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        for idx, (sample, y, simi_value, masked) in enumerate(test_loader):
            loss = calculate_multitask_loss(args, sample,
                                            model, criterion, DEVICE)
            n_batches += 1
            total_loss += loss.item()
        if n_batches == 0:
            logger.debug(f'Test Loss     : {total_loss:.4f}')
            # writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)
            fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss}})
        else:
            logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
            # writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)
            fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss / n_batches}})
    return model

def plot_tsne(model, feats, y_true, savefig):
    model.eval()
    tsne(feats.reshape(-1, feats.shape[-1])[:10000,:],
         y_true.reshape(-1)[:10000],
         savefig)
    model.train()
    return

def train_SSL(train_loaders, val_loader, model,
              logger, fitlog, DEVICE, optimizers,
              schedulers, criterion, args, writer):
    best_model = copy.deepcopy(model.state_dict())
    # best_model = None
    min_val_loss = 1e8
    # pltTiming = int(args.n_epoch / 10)

    for epoch in range(args.n_epoch):
        total_loss = 0
        n_batches = 1
        model.train()
        latents, y_true = None, None
        for i, train_loader in enumerate(train_loaders):
            # true_values, pred_values = [], []
            for idx, (sample, y, simi_value, masked) in enumerate(train_loader):
                # true_values.append(simi_value.detach().cpu().numpy())
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # model training here...
                # sample(B,Len,Dim), simi(B,L,N12)
                # # train_loader.data
                # sample = sample.to(DEVICE).float()
                # _, tmp_q = model(sample)
                # p = model.target_distribution(tmp_q)
                if args.framework in ['CNNRNN', 'ProposeSSL', 'unet']:  # 这几个的output是simi value
                    loss, pred_value = calculate_model_loss_CNNRNN(
                        args, sample, simi_value, masked,
                        model, criterion, DEVICE)

                    # # IDEC
                    # if epoch>5 and epoch%10==0:
                    #     loss += 0.1*kl_loss

                elif args.framework in ['CNN_AEframe', 'SSL']:  # SSL的output需要和input接近
                    # ae=autoencoder: model output=input
                    loss, pred_value = calculate_model_loss_CNNRNN(
                        args, sample, sample, masked,
                        model, criterion, DEVICE)

                else:
                    print('not implement loss function')
                    NotImplementedError
                # pred_values.append(pred_value)
                # L2 norm
                # if args.framework not in ['CNN_AEframe', 'unet']:  # autoencoder CNN_AE没有framework
                #     for param in model.projector.parameters():
                #         loss += 0.0001 * torch.norm(param)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1

        fitlog.add_loss(optimizers[0].param_groups[0]['lr'],
                        name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

        # save model
        # if True:
        if (epoch == args.n_epoch-1):
        # if (epoch % 100 == 0) or (epoch == args.n_epoch-1):
            model_dir = model_dir_name + '/pretrain_' + args.model_name + str(epoch) + '.pt'
            print('Saving model at {} epoch to {}'.format(epoch, model_dir))
            # torch.save(model.state_dict(), model_dir)
            torch.save({'model_state_dict': model.state_dict()}, model_dir)

        # logger.debug(f'Train Loss     : {total_loss / n_batches:.4f}')
        writer.add_scalar('scalar_xia/upTrainLoss', total_loss / n_batches, epoch)

        fitlog.add_loss(total_loss / n_batches, name="pretrain training loss", step=epoch)

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_model = copy.deepcopy(model.state_dict())
                break
        else:
            with torch.no_grad():
                model.eval()
                total_loss = 0
                n_batches = 1
                v_pred,v_true = [],[]  # 存储所有pre/true output，检验reconstruction效果
                latents, y_true = None, None
                for idx, (samplev, y, simivalue, masked) in enumerate(val_loader):

                    if args.framework in ['CNNRNN', 'ProposeSSL', 'unet']:
                        loss, predvalue = calculate_model_loss_CNNRNN(
                            args, samplev, simivalue, masked,
                            model, criterion, DEVICE)
                        # 1.1 plot data information
                        v_true.append(simivalue.detach().cpu().numpy())

                    elif args.framework in ['CNN_AEframe', 'SSL']:
                        loss, predvalue = calculate_model_loss_CNNRNN(
                            args, samplev, samplev, masked,
                            model, criterion, DEVICE)
                        # 1.2 plot data information
                        v_true.append(samplev.detach().cpu().numpy())
                    else:
                        NotImplementedError
                    total_loss += loss.item()
                    if samplev.size(0) != args.batch_size:
                        continue
                    n_batches += 1

                    # 2 plot data information
                    v_pred.append(predvalue.detach().cpu().numpy())

                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                    # print('update')
                # logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                writer.add_scalar('scalar_xia/upValLoss', total_loss / n_batches, epoch)
                fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)

    return best_model


def test_SSL(test_loader, best_model, logger, fitlog, DEVICE, criterion, args):
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        latents, y_true = [],[]
        # latents, y_true = None, None
        v_pred, v_true = [], []
        v_raw, v_label = [], []
        for idx, (sample, y, simi_value, masked) in enumerate(test_loader):
            if args.framework in ['CNNRNN', 'ProposeSSL', 'unet']:
                loss, predvalue = calculate_model_loss_CNNRNN(args, sample, simi_value, masked,
                                                              model, criterion, DEVICE)
                v_true.append(simi_value.detach().cpu().numpy())

                sample = sample.to(DEVICE).float()
                _, latent = model.encoder(sample)
                latents.append(latent.cpu().detach().numpy())
                y_true.append(y)

            elif args.framework in ['CNN_AEframe', 'SSL']:
                loss, predvalue = calculate_model_loss_CNNRNN(args, sample, sample, masked,
                                                              model, criterion, DEVICE)
                v_true.append(simi_value.detach().cpu().numpy())

            else:
                print('not implement loss')
                NotImplementedError
            # if sample.size(0) != args.batch_size:
            #     continue
            n_batches += 1
            total_loss += loss.item()

            v_pred.append(predvalue.detach().cpu().numpy())
            v_raw.append(sample.detach().cpu().numpy())
            v_label.append(y.detach().cpu().numpy())

        # # if (True):
        # if (args.plt == True):
        #     # ---------------similarity series--------------------
        #     plt.figure(figsize=(12, 8))
        #     dim = predvalue.shape[-1]
        #     v_true = np.concatenate(v_true)  # 3dim,concat first dim
        #     v_pred = np.concatenate(v_pred)
        #
        #     for subidx in range(dim):  # todo 只画几条
        #         plt.subplot(dim, 1, subidx + 1)  # todo 只画几条
        #         a = v_pred.reshape(-1, dim)[:4000, subidx]
        #         b = v_true.reshape(-1, dim)[:4000, subidx]
        #         plt.plot(b, 'b', linewidth=0.5, label='true')
        #         plt.plot(a, 'r-.', linewidth=0.5, label='pred')
        #         plt.axis('off')
        #     plt.savefig('%ssimi.png'%args.dataset, dpi=600)
        #     plt.close()
        #
        #     # ---------------part data--------------------
        #     fig,ax1 = plt.subplots(figsize=(12,8))
        #     dim = sample.shape[-1]
        #     v_true = np.concatenate(v_raw)  # 3dim,concat first dim
        #     a = v_true.reshape(-1, dim)[:4000, :3]
        #     ax1.plot(a, label='raw')
        #     ax1.set_ylabel('accelerometer')
        #     ax1.tick_params('y')
        #
        #     ax2 = ax1.twinx()
        #     dim = y.shape[-1]
        #     v_true1 = np.concatenate(v_label)
        #     b = v_true1.reshape(-1, dim)[:4000, :]
        #     ax2.plot(b, label='opelabel')
        #     ax2.set_ylabel('opelabel')
        #     ax2.tick_params('y')
        #     # Add legends
        #     lines, labels = ax1.get_legend_handles_labels()
        #     lines2, labels2 = ax2.get_legend_handles_labels()
        #     ax1.legend(lines + lines2, labels + labels2)
        #     plt.savefig('%srawpart.png'%args.dataset, dpi=600)
        #     plt.close()

        if n_batches == 0:
            # logger.debug(f'Test Loss     : {total_loss:.4f}')
            # writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)
            fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss}})
        else:
            # logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
            # writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)
            fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss / n_batches}})

        # latents1 = np.concatenate(latents)
        # y_true1 = np.concatenate(y_true)
        # np.save('u0102ssl_latentT.npy', latents1)
        # np.save('u0102ssl_yT.npy', y_true1)
        # np.save('acc6ssl_latentT.npy', latents1)
        # np.save('acc6ssl_yT.npy', y_true1)
        # np.save('u1ssl_latentT.npy', latents1)
        # np.save('u1ssl_yT.npy', y_true1)
        # np.save('acc6ssl_latentT.npy', latents.cpu().detach().numpy())
        # np.save('acc6ssl_yT.npy', y_true.cpu().detach().numpy())
        # tsne(latents.reshape(-1,128)[:10000,:], y_true.reshape(-1)[:10000],
        #      args.user_name + '_ssltsneT.png')
        # print('')

    return model


def test_contrast(test_loader, best_model, logger, fitlog, DEVICE, criterion, args, writer):
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 1
        for idx, (sample, target, domain, masked) in enumerate(test_loader):
            loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon,
                                        nn_replacer=nn_replacer)
            total_loss += loss.item()
            if sample.size(0) != args.batch_size:
                continue
            n_batches += 1
        # logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
        # writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)
        fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss / n_batches}})

    return model


def lock_backbone(model, args):
    for name, param in model.named_parameters():  # freeze the whole framework
        param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net  # =backbone=cnnrnn
    elif args.framework in ['simclr', 'CNNRNN', 'SSL', 'ProposeSSL', 'multi', 'unet']:
        trained_backbone = model.encoder
    elif args.framework in ['CNN_AEframe']:  # todo, test it
        trained_backbone = model.encoder
    else:
        print('not copy backbone parameters')
        NotImplementedError

    return trained_backbone


def train_lincls_CNNRNN(best_pretrain_model,
                        train_loaders, val_loader, trained_backbone,
                        classifier, logger, fitlog, DEVICE, optimizer,
                        criterion, args, writer, user_name='u0101'):
    best_lincls = None
    min_val_loss = 1e8

    num_epochs = args.n_epoch_supervised
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        classifier.train()
        # best_pretrain_model.train()
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        total = 0
        correct = 0
        n_batches = 1
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain, masked) in enumerate(train_loader):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                # # 0529 test, reconstruct similarity series
                # pred_simi = best_pretrain_model(sample)
                # true_simi = domain

                if args.framework in ['CNN_AEframe']:
                    loss, predicted, _ = calculate_lincls_output_AE(sample, target,
                                            trained_backbone, classifier, criterion)
                else:
                    loss, predicted, _,probability = calculate_lincls_output_CNNRNN(sample, target,
                                                                    trained_backbone, classifier, criterion)

                if target.shape != predicted.shape:  # openpack
                    target = target[:, :, 0]

                total_loss += loss.item()
                n_batches += 1
                total += target.size(0) * target.size(
                    1)  # the accuracy of every data point, if not use (1), then accuracy is segment result
                correct += (predicted == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save model
        # model_dir = model_dir_name + '/lincls_' + user_name + '_' + args.model_name + str(epoch) + '.pt'
        # print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        # torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()},
        #            model_dir)

        acc_train = float(correct) * 100.0 / total
        logger.debug(f'epoch train loss     : {total_loss / n_batches:.4f}, '
                     f'train acc     : {acc_train:.4f}')
        writer.add_scalar('scalar_xia/%s/downTrainLoss' % user_name, total_loss, epoch)
        writer.add_scalar('scalar_xia/%s/downTrainAcc' % user_name, acc_train, epoch)
        fitlog.add_loss(total_loss, name="Train Loss of %s" % user_name, step=epoch)
        fitlog.add_metric({"dev": {"Train Acc of %s" % user_name: acc_train}}, step=epoch)

        if args.scheduler:
            scheduler.step()

        probalist = []
        trgs = np.array([])
        preds = np.array([])
        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:  # args.cases = random

            with torch.no_grad():
                classifier.eval()
                # best_pretrain_model.eval()
                total_loss = 0
                total2 = 0
                correct2 = 0
                n_batches = 1

                for idx, (sample, target, domain, masked) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    if args.framework in ['CNN_AEframe']:
                        loss, predicted, _ = calculate_lincls_output_AE(sample, target,
                                                                        trained_backbone, classifier, criterion)
                    else:
                        loss, predicted, _, probability = calculate_lincls_output_CNNRNN(sample, target,
                                                                            trained_backbone, classifier, criterion)
                    # probalist = probability.cpu().numpy()
                    if target.shape != predicted.shape:  # openpack
                        target = target[:, :, 0]

                    total_loss += loss.item()
                    n_batches += 1
                    total2 += target.size(0) * target.size(1)
                    correct2 += (predicted == target).sum()

                    trgs = np.append(trgs, target.data.cpu().numpy().reshape(-1))
                    preds = np.append(preds, predicted.data.cpu().numpy().reshape(-1))

                acc_val = float(correct2) * 100.0 / (total2+0.000001)
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
                    print('update')
                logger.debug(f'epoch val loss     : {total_loss / n_batches:.4f}, '
                             f'val acc     : {acc_val:.4f}')
                writer.add_scalar('scalar_xia/%s/downValLoss' % user_name, total_loss, epoch)
                writer.add_scalar('scalar_xia/%s/downValAcc' % user_name, acc_val, epoch)
                fitlog.add_loss(total_loss, name="Val Loss of %s" % user_name, step=epoch)
                fitlog.add_metric({"dev": {"Val Acc of %s" % user_name: acc_val}}, step=epoch)

        # # 将main函数的test函数放到这
        if epoch == num_epochs-1:
            # probalist
            if args.framework == 'CNNRNN':
                test_lincls_CNNRNN_lastepoch(best_pretrain_model, val_loader, trained_backbone, classifier,
                                         logger, fitlog, DEVICE, criterion, args, plts=args.plt, writer=writer,
                                         target_user=user_name)
            else:
                acc = accuracy_score(trgs, preds) * 100
                maF = f1_score(trgs, preds, average='macro') * 100
                weiF = f1_score(trgs, preds, average='weighted') * 100
                fitlog.add_best_metric({"dev": {"Test Acc of %s" % acc}})
                fitlog.add_best_metric({"dev": {"maF of %s" %  maF}})
                fitlog.add_best_metric({"dev": {"weightedF1 of %s" %  weiF}})
                result = 'accuracy--  ' + str(acc) + '\n' + 'macroF1--  ' + str(maF) + '\n' + 'weightedF1-- ' + str(weiF)
                print(result)
        #
        #     # test if trainloader is good or not, because the accuracy is low
        #     test_lincls_CNNRNN_lastepoch2(best_pretrain_model, train_loaders[0], trained_backbone, classifier,
        #                                  logger, fitlog, DEVICE, criterion, args, plts=args.plt, writer=writer,
        #                                  target_user=user_name)
    return best_lincls


def test_lincls_CNNRNN(test_loader, trained_backbone, best_lincls,
                       logger, fitlog, DEVICE, criterion, args, plt=False, writer=None, target_user='u'):
    if args.framework in ['CNN_AEframe']:
        classifier = setup_linclf_AE(args, DEVICE, 128)
    else:
        classifier = setup_linclf_CNNRNN(args, DEVICE, trained_backbone.out_dim)
    if args.framework in ['CNNRNN', 'CNN_AEframe']:
        classifier.classifier.load_state_dict(best_lincls)
    else:
        classifier.load_state_dict(best_lincls)  # 因为best_lincls不同
    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    sensor_data = np.array([])
    trgs = np.array([])
    preds = np.array([])
    with torch.no_grad():
        classifier.eval()
        for idx, (sample, target, domain, masked) in enumerate(test_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            if args.framework in ['CNN_AEframe']:
                loss, predicted, feat = calculate_lincls_output_AE(sample, target,
                                                        trained_backbone, classifier, criterion)
            else:
                loss, predicted, feat = calculate_lincls_output_CNNRNN(sample, target,
                                                            trained_backbone, classifier, criterion)

            total_loss += loss.item()

            if target.shape != predicted.shape:  # 处理openpack
                target = target[:, :, 0]

            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            if len(sensor_data) == 0:
                sensor_data = sample.data.cpu().numpy().reshape(-1, sample.shape[-1])
            else:
                sensor_data = np.concatenate([sensor_data, sample.data.cpu().numpy().reshape(-1, sample.shape[-1])])
            trgs = np.append(trgs, target.data.cpu().numpy().reshape(-1))
            preds = np.append(preds, predicted.data.cpu().numpy().reshape(-1))
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum()
        acc_test = float(correct) * 100.0 / total

        maF = f1_score(trgs, preds, average='macro') * 100
        weiF = f1_score(trgs, preds, average='weighted') * 100

        logger.debug(
            f'epoch test loss     : {total_loss:.4f}, accuracy    : {acc_test:.4f}'
            f' , macroF1     : {maF:.4f}, weightedF1     : {weiF:.4f}')

        fitlog.add_best_metric({"dev": {"Test Loss of %s" % target_user: total_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc of %s" % target_user: acc_test}})
        fitlog.add_best_metric({"dev": {"maF of %s" % target_user: maF}})
        fitlog.add_best_metric({"dev": {"weightedF1 of %s" % target_user: weiF}})

        print('confusion matrix of %s:' % target_user)
        logger.debug(confusion_matrix)
        print('accuracy of each class:')
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

    if plt == True:
        # plot sensor data and labels
        plot_data2label(sensor_data, trgs, preds, plot_dir_name)
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)
    return

def test_lincls_CNNRNN_lastepoch(best_pretrain_model, test_loader, trained_backbone, classifier,
                       logger, fitlog, DEVICE, criterion, args, plts=False, writer=None, target_user='u'):

    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    sensor_data = np.array([])
    trgs = np.array([])
    preds = np.array([])
    probs = np.array([])
    simis = np.array([])
    n_batches = 1
    with torch.no_grad():
        classifier.eval()
        # v_pred, v_true = [], []
        for idx, (sample, target, simi, masked) in enumerate(test_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            if args.framework in ['CNN_AEframe']:
                loss, predicted, feat = calculate_lincls_output_AE(sample, target,
                                                        trained_backbone, classifier, criterion)
            else:
                loss, predicted, feat, probability = calculate_lincls_output_CNNRNN(sample, target,
                                                            trained_backbone, classifier, criterion)

            pred_simi = best_pretrain_model(sample)

            total_loss += loss.item()
            n_batches += 1

            if target.shape != predicted.shape:  # 处理openpack
                target = target[:, :, 0]

            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            if len(sensor_data) == 0:
                sensor_data = sample.data.cpu().numpy().reshape(-1, sample.shape[-1])
                simi_series = pred_simi.cpu().numpy().reshape(-1, pred_simi.shape[-1])
            else:
                sensor_data = np.concatenate([sensor_data, sample.data.cpu().numpy().reshape(-1, sample.shape[-1])])
                simi_series = np.concatenate([simi_series, pred_simi.cpu().numpy().reshape(-1, simi.shape[-1])])
            trgs = np.append(trgs, target.data.cpu().numpy().reshape(-1))
            preds = np.append(preds, predicted.data.cpu().numpy().reshape(-1))
            # tmp = np.transpose(probability.cpu().numpy(), (0, 2, 1))
            # if len(probs) == 0:
            #     probs = tmp.reshape(-1,8)
            #     simis = simi.data.cpu().numpy().reshape(-1,13)
            # else:
            #     probs = np.concatenate([probs, tmp.reshape(-1, 8)], axis=1)
            #     simis = np.concatenate([simis, simis.data.cpu().numpy().reshape(-1, 13)], axis=1)

            # probs = np.append([probs, probability.cpu().numpy().reshape(-1,8)])
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum()

        acc_test = float(correct) * 100.0 / total
        total_loss = total_loss / n_batches

        acc = accuracy_score(trgs, preds) * 100
        maF = f1_score(trgs, preds, average='macro') * 100
        weiF = f1_score(trgs, preds, average='weighted') * 100

        logger.debug(
            f'epoch test loss     : {total_loss:.4f}, accuracy    : {acc:.4f}'
            f' , macroF1     : {maF:.4f}, weightedF1     : {weiF:.4f}')

        fitlog.add_best_metric({"dev": {"Test Loss of %s" % target_user: total_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc of %s" % target_user: acc_test}})
        fitlog.add_best_metric({"dev": {"maF of %s" % target_user: maF}})
        fitlog.add_best_metric({"dev": {"weightedF1 of %s" % target_user: weiF}})

        print('confusion matrix of %s:' % target_user)
        logger.debug(confusion_matrix)
        print('accuracy of each class:')
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

        # 批量处理时写入新文件
        result = 'accuracy--  ' + str(acc) + '\n' + 'macroF1--  ' + str(maF) + '\n' + 'weightedF1-- ' + str(weiF)
        fname = 'Period_' + str(args.n_periods) + args.dataset + '_' + args.user_name + '_' + args.framework + '_' + str(args.num_motifs) +'_seed'+ str(args.seed)+'.txt'
        with open(fname, 'w') as f:
            f.write(result)


    # if plts == True:
        # # plot sensor data and labels
        # plot_data2label(sensor_data, simi_series, trgs, preds, plot_dir_name, probs)
        # # tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        # # mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        # sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        # sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        # print('plots saved to ', plot_dir_name)

    return


def test_lincls_CNNRNN_lastepoch2(best_pretrain_model, train_loader, trained_backbone, classifier,
                       logger, fitlog, DEVICE, criterion, args, plt=False, writer=None, target_user='u'):

    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    sensor_data = np.array([])
    trgs = np.array([])
    preds = np.array([])
    with torch.no_grad():
        classifier.eval()
        # v_pred, v_true = [], []
        for idx, (sample, target, simi, masked) in enumerate(train_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            if args.framework in ['CNN_AEframe']:
                loss, predicted, feat = calculate_lincls_output_AE(sample, target,
                                                                trained_backbone, classifier, criterion)
            else:
                loss, predicted, feat = calculate_lincls_output_CNNRNN(sample, target,
                                                        trained_backbone, classifier, criterion)

            total_loss += loss.item()
            pred_simi = best_pretrain_model(sample)

            if target.shape != predicted.shape:  # 处理openpack
                target = target[:, :, 0]

            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)

            # because data overlapped, use args.steps to get data
            sample_unoverlap = sample[:,:90,:]
            target_unoverlap = target[:,:90]
            predicted_unoverlap = predicted[:,:90]
            pred_simi_unoverlap = pred_simi[:, :90, :]

            if len(sensor_data) == 0:
                sensor_data = sample_unoverlap.data.cpu().numpy().reshape(-1, sample_unoverlap.shape[-1])
                simi_series = pred_simi_unoverlap.data.cpu().numpy().reshape(-1, pred_simi_unoverlap.shape[-1])
            else:
                sensor_data = np.concatenate([sensor_data,
                                              sample_unoverlap.data.cpu().numpy().reshape(-1,
                                                                                          sample_unoverlap.shape[-1])])
                simi_series = np.concatenate([simi_series, pred_simi_unoverlap.data.cpu().numpy().reshape(-1, pred_simi_unoverlap.shape[-1])])
            trgs = np.append(trgs, target_unoverlap.data.cpu().numpy().reshape(-1))
            preds = np.append(preds, predicted_unoverlap.data.cpu().numpy().reshape(-1))


            for t, p in zip(target_unoverlap.reshape(-1), predicted_unoverlap.reshape(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target_unoverlap.size(0) * target_unoverlap.size(1)
            correct += (predicted_unoverlap == target_unoverlap).sum()

        print('confusion matrix of %s:' % target_user)
        logger.debug(confusion_matrix)
        print('accuracy of each class:')
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

    if plt == True:
        # plot sensor data and labels
        plot_data2label(sensor_data, simi_series, trgs, preds, plot_dir_name,  comment='train')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_train_confmatrix.png')
        print('plots saved train results to ', plot_dir_name)

    return

def plot_data2label(sensor_data, simi_series, trgs, preds, plot_dir_name, probs, comment='test'):
    if comment=='train':
        data_length = 5000
    else:
        data_length = 4000 # ome
        # data_length = 10000
    assert sensor_data.shape[0] == trgs.shape[0]
    moredata = sensor_data.shape[0] % data_length
    x_win = sensor_data[:-moredata, :]
    y_win = trgs[:-moredata]
    d_win = preds[:-moredata]
    s_win = simi_series[:-moredata, :]
    x_win = x_win.reshape(-1, data_length, sensor_data.shape[-1])
    y_win = y_win.reshape(-1, data_length)
    d_win = d_win.reshape(-1, data_length)
    s_win = s_win.reshape(-1, data_length, simi_series.shape[-1])
    p_win = probs[:-moredata, :]
    p_win = p_win.reshape(-1, data_length, probs.shape[-1])

    # colors = ['red', 'blue', 'green', 'orange','grey','yellow','black']  # List of colors
    # for i in range(x_win.shape[0]):
    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    #     # plt.subplot(3, 1, 1)
    #     ax1.plot(x_win[i])
    #     ax2.plot(y_win[i], 'k', linewidth=2, label='true')
    #     ax2.plot(d_win[i], 'b-.', linewidth=0.5, label='pred')
    #     ax2.legend()
    #     ax2.set_ylabel('label ID')
    #     count = 1
    #     for simiid in range(3):
    #     # for simiid in range(8):
    #         color = colors[simiid % len(colors)]
    #         ax3.plot(s_win[i, :, simiid]+count, 'g-', linewidth=1, color=color, label='simi%d' % simiid)
    #         # print('max is %f,min is %f' % (max(s_win[i, :, simiid]), min(s_win[i, :, simiid])))
    #         count += 1
    #     ax3.legend()
    #     ax3twin = ax3.twinx()
    #     ax3twin.plot(y_win[i], 'k', linewidth=0.3)
    #     ax3twin.grid()
    #     if comment=='train':
    #         plt.savefig(plot_dir_name + '/' + str(i) + '_trian_th_data2label.png')
    #     else:
    #         plt.savefig(plot_dir_name + '/' + str(i) + '_th_data2label.png')
    #     plt.close()
    # # 最后
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # # plt.subplot(3, 1, 1)
    # ax1.plot(sensor_data[-moredata:, :])
    # ax2.plot(trgs[-moredata:], 'k', linewidth=2, label='true')
    # ax2.plot(preds[-moredata:], 'b-.', linewidth=0.5, label='pred')
    # ax2.legend()
    # ax2.set_ylabel('label ID')
    # s_win = simi_series[-moredata:, :]
    # dim = s_win.shape[-1]
    # # for simiid in range(8):
    # for simiid in range(dim):
    #     color = colors[simiid % len(colors)]
    #     ax3.plot(s_win[:, simiid], 'g-', linewidth=1, color=color, label='simi%d' % simiid)
    #     # print('max is %f,min is %f' % (max(s_win[:, simiid]), min(s_win[:, simiid])))
    # ax3.legend()
    # ax3twin = ax3.twinx()
    # ax3twin.plot(trgs[-moredata:], 'k', linewidth=0.3)
    # ax3twin.grid()
    # if comment == 'train':
    #     plt.savefig(plot_dir_name + '/' + str(i+1) + '_trian_th_data2label.png')
    # else:
    #     plt.savefig(plot_dir_name + '/' + str(i+1) + '_th_data2label.png')
    # plt.close()
    return


def smoothloss(target, predict):
    # 对batch里每个数分别求loss
    # test2: smooth, (y_t-yy_t - (y_t-1 - yy_t-1))^2
    # tmp = target - predict
    # differ = tmp[:,1:,:] - tmp[:,:-1,:]
    # loss = torch.sum(differ**2) / (tmp.shape[0]*tmp.shape[1]*tmp.shape[2])
    # print(loss.item())
    dist_target = torch.norm(target[:, 1:, :] - target[:, :-1, :], dim=2)
    dist_predict = torch.norm(predict[:, 1:, :] - predict[:, :-1, :], dim=2)
    loss = torch.sum((dist_target-dist_predict) ** 2) / (dist_predict.shape[0] * dist_predict.shape[1])
    return loss  # average loss

def calculate_lincls_output_CNNRNN(sample, target, trained_backbone,
                                   classifier, criterion):
    feat = trained_backbone(sample)  #sample:128,1080,6
    if type(feat)==tuple:
        feat = feat[1]  # feat[1]128,1080,128, transformer backbone+proposeSSL 情况

    if len(feat.shape) == 4:
        # print('processing multi-task model..')
        feat = feat.squeeze(2).permute(0, 2, 1)

    try:
        output, senresult = classifier(feat)  # fea input(B,Len180,dim128)  # 现在输出两个结果
    except:
        print('Multitask: trainer_CNNRNN, calculate_lincls_output_CNNRNN.')
        output = classifier(feat)  # only for multi task, output one result
    # output, senresult = classifier(feat)  # fea input(B,Len180,dim128)  # 现在输出两个结果
    if len(output.shape) == 3:
        minlen = min(output.shape[1], target.shape[1])
        output = output[:, :minlen, :]
        target = target[:, :minlen, :]
        output = output.permute(0, 2, 1)  # if multi dim, B,Ch,Len
        target = target.squeeze(-1)  # 3 dim-> 2 dim
    try:
        # target = target.squeeze(2)
        loss = criterion(output, target)   # crossentropy loss: input(B,class,len),target(B, len)
        # loss = criterion(output, target) + 0.1*smoothloss(sample, senresult)  # crossentropy loss: input(B,class,len),target(B, len)
        # loss = 20*smoothloss(sample, senresult)  # crossentropy loss: input(B,class,len),target(B, len)
    except:
        loss = criterion(output, target[:, 0])  # simsiam
        # loss = criterion(output, target[:, 0, 0])
    _, predicted = torch.max(output.data, 1)  # 返回每一行的最大值，且返回索引
    outputsoftmax = torch.softmax(output, dim=1)
    return loss, predicted, feat, outputsoftmax.data


def calculate_lincls_output_AE(sample, target, trained_backbone, classifier, criterion):
    feat, _ = trained_backbone(sample)  # encoder output(batch,128,15)
    feat = trained_backbone.upsample(feat)
    feat = feat.permute(0,2,1)  # framework的classifier只有一个linear，所以fea要flat
    output,_ = classifier(feat)  # fea out(batch, datalen180, #class12)

    if len(output.shape) == 3:
        minlen = min(output.shape[1], target.shape[1])
        output = output[:, :minlen, :]
        target = target[:, :minlen, :]

    # calculate crossentropy loss
    output = output.permute(0, 2, 1)  # bacth, dim, datalen
    loss = criterion(output, target[:, :, 0])  # target=truth(batch, dim)
    _, predicted = torch.max(output.data, 1)  # 返回每一行的最大值，且返回索引

    return loss, predicted, feat

def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion):
    _, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    output = classifier(feat)
    try:
        loss = criterion(output, target)  # crossentropy loss: input(B,class),target(B)
    except:
        loss = criterion(output, target[:, 0, 0])
    _, predicted = torch.max(output.data, 1)
    return loss, predicted, feat


def train_lincls(train_loaders, val_loader, trained_backbone,
                 classifier, logger, fitlog, DEVICE, optimizer,
                 criterion, args, writer, target_user='0'):
    best_lincls = None
    min_val_loss = 1e8

    num_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        # for epoch in range(args.n_epoch):
        classifier.train()
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        total = 0
        correct = 0
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                loss, predicted, _ = calculate_lincls_output(sample, target,
                                                             trained_backbone, classifier, criterion)

                if target.shape != predicted.shape:  # openpack
                    target = target[:, 0, 0]

                total_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save model
        model_dir = model_dir_name + '/lincls_' + target_user + '_' + args.model_name + str(epoch) + '.pt'
        print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()},
                   model_dir)

        acc_train = float(correct) * 100.0 / total
        logger.debug(f'epoch train loss     : {total_loss:.4f}, train acc     : {acc_train:.4f}')
        writer.add_scalar('scalar/%s/downTrainLoss' % target_user, total_loss, epoch)
        writer.add_scalar('scalar/%s/downTrainAcc' % target_user, acc_train, epoch)
        fitlog.add_loss(total_loss, name="Train Loss of %s" % target_user, step=epoch)
        fitlog.add_metric({"dev": {"Train Acc of %s" % target_user: acc_train}}, step=epoch)

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                total_loss = 0
                total2 = 0
                correct2 = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    loss, predicted, _ = calculate_lincls_output(sample, target,
                                                                 trained_backbone, classifier,
                                                                 criterion)

                    if target.shape != predicted.shape:  # openpack
                        target = target[:, 0, 0]

                    total_loss += loss.item()
                    total2 += target.size(0)
                    correct2 += (predicted == target).sum()
                acc_val = float(correct2) * 100.0 / total2
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
                    # print('update')
                logger.debug(f'epoch val loss     : {total_loss:.4f}, val acc     : {acc_val:.4f}')
                writer.add_scalar('scalar/%s/downValLoss' % target_user, total_loss, epoch)
                writer.add_scalar('scalar/%s/downValAcc' % target_user, acc_val, epoch)
                fitlog.add_loss(total_loss, name="Val Loss of %s" % target_user, step=epoch)
                fitlog.add_metric({"dev": {"Val Acc of %s" % target_user: acc_val}}, step=epoch)
    return best_lincls


def test_lincls(test_loader, trained_backbone, best_lincls,
                logger, fitlog, DEVICE, criterion, args, plt=False, writer=None, target_user='u'):
    classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
    classifier.load_state_dict(best_lincls)
    total_loss = 0
    total = 0
    correct = 0
    confusion_matrix = torch.zeros(args.n_class, args.n_class)
    feats = None
    trgs = np.array([])
    preds = np.array([])
    with torch.no_grad():
        classifier.eval()
        for idx, (sample, target, domain) in enumerate(test_loader):
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            loss, predicted, feat = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)
            total_loss += loss.item()

            if target.shape != predicted.shape:  # 处理openpack
                target = target[:, :, 0]

            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            trgs = np.append(trgs, target.data.cpu().numpy())
            preds = np.append(preds, predicted.data.cpu().numpy())
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum()
        acc_test = float(correct) * 100.0 / total

        miF = f1_score(trgs, preds, average='micro') * 100
        maF = f1_score(trgs, preds, average='weighted') * 100

        logger.debug(
            f'epoch test loss     : {total_loss:.4f}, test acc     : {acc_test:.4f}, microF1     : {miF:.4f}, weightedF1     : {maF:.4f}')

        fitlog.add_best_metric({"dev": {"Test Loss of %s" % target_user: total_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc of %s" % target_user: acc_test}})
        fitlog.add_best_metric({"dev": {"miF of %s" % target_user: miF}})
        fitlog.add_best_metric({"dev": {"maF of %s" % target_user: maF}})

        print('confusion matrix of %s:' % target_user)
        logger.debug(confusion_matrix)
        print('accuracy of each class:')
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)

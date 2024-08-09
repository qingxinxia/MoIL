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
from data_preprocess import data_preprocess_logi
from data_preprocess import data_preprocess_ucihar
from data_preprocess import data_preprocess_shar
from data_preprocess import data_preprocess_hhar

from sklearn.metrics import f1_score
import seaborn as sns
import fitlog
from copy import deepcopy
import matplotlib.pyplot as plt

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
        args.len_sw = 45  #90  # 30Hz * 10s
        args.n_class = 12
        # if args.cases not in ['subject']:   # use: [random]
        #     args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_openpack.prep_openpack(
                                                            args,
                                                            SLIDING_WINDOW_LEN=args.len_sw,
                                                            SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                            if_split_user=if_split_user)
    if args.dataset == 'neji':
        args.n_feature = 3  # two wrists
        args.len_sw = 45  #90  # 30Hz * 10s
        args.n_class = 12
        # if args.cases not in ['subject']:   # use: [random]
        #     args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_neji.prep_neji(
                                                            args,
                                                            SLIDING_WINDOW_LEN=args.len_sw,
                                                            SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                            if_split_user=if_split_user)
    if args.dataset == 'logi':
        args.n_feature = 6  # two wrists
        args.len_sw = 45  #90  # 30Hz * 10s
        args.n_class = 10
        # if args.cases not in ['subject']:   # use: [random]
        #     args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_logi.prep_logi(
                                                            args,
                                                            SLIDING_WINDOW_LEN=args.len_sw,
                                                            SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                            if_split_user=if_split_user)

    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '0'
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))
    if args.dataset == 'shar':
        args.n_feature = 3
        args.len_sw = 151
        args.n_class = 17
        if args.cases not in ['subject', 'subject_large']:
            args.target_domain == '1'
        train_loaders, val_loader, test_loader = data_preprocess_shar.prep_shar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5))
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)

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


def setup_model_optm(args, DEVICE, classifier=True):
    # set up backbone network
    if args.backbone == 'FCN':
        backbone = FCN(n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True)
    elif args.backbone == 'CNNRNN':  # finial layer is different from dcl, use yoshimura code
        backbone = CNNRNN(n_channels=args.n_feature, n_classes=args.n_class, simi_dim=30,
                          conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True)
    elif args.backbone == 'LSTM':
        backbone = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True)
    elif args.backbone == 'AE':
        backbone = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        backbone = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True)
    elif args.backbone == 'Transformer':
        backbone = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    else:
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

    else:
        NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
        if args.framework == 'CNNRNN':
            classifier = backbone.classifier
            classifier.weight.data.normal_(mean=0.0, std=0.01)
            classifier.bias.data.zero_()
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
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 1.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'   # contrastive loss NT-Xent
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay = 3e-4
    if args.framework == 'CNNRNN':
        args.criterion = 'mse'
        args.backbone = 'CNNRNN'

    classifier = True
    if args.framework == 'CNNRNN':
        classifier = False  # classifier 不一样
    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=classifier)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    if args.criterion == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.2)
        else:
            criterion = NTXentLoss(DEVICE, args.batch_size, temperature=0.1)

    args.model_name = 'try_scheduler_' + args.framework + '_pretrain_' + args.dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit

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

    return model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls


def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None):
    aug_sample1 = gen_aug(sample, args.aug1)  # t_warp
    aug_sample2 = gen_aug(sample, args.aug2)  # negate

    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(
        DEVICE).long()
    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
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
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    return loss


def train_contrast(train_loaders, val_loader, model, logger, fitlog, DEVICE, optimizers, schedulers, criterion, args, writer):
    best_model = copy.deepcopy(model.state_dict())
    # best_model = None
    min_val_loss = 1e8

    for epoch in range(args.n_epoch):
        logger.debug(f'\nEpoch : {epoch}')
        total_loss = 0
        n_batches = 0
        model.train()
        for i, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                for optimizer in optimizers:
                    optimizer.zero_grad()
                if sample.size(0) != args.batch_size:
                    continue
                n_batches += 1
                loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                total_loss += loss.item()
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                if args.framework == 'byol':
                    model.update_moving_average()
        fitlog.add_loss(optimizers[0].param_groups[0]['lr'], name="learning rate", step=epoch)
        for scheduler in schedulers:
            scheduler.step()

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
                n_batches = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    if sample.size(0) != args.batch_size:
                        continue
                    n_batches += 1
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
                    total_loss += loss.item()
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_model = copy.deepcopy(model.state_dict())
                    print('update')
                logger.debug(f'Val Loss     : {total_loss / n_batches:.4f}')
                writer.add_scalar('scalar/upValLoss', total_loss / n_batches, epoch)
                fitlog.add_loss(total_loss / n_batches, name="pretrain validation loss", step=epoch)
    return best_model


def test_contrast(test_loader, best_model, logger, fitlog, DEVICE, criterion, args, writer):
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        for idx, (sample, target, domain) in enumerate(test_loader):
            if sample.size(0) != args.batch_size:
                continue
            n_batches += 1
            loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)
            total_loss += loss.item()
        logger.debug(f'Test Loss     : {total_loss / n_batches:.4f}')
        # writer.add_scalar('scalar/upTrainLoss', total_loss / n_batches, epoch)
        fitlog.add_best_metric({"dev": {"pretrain test loss": total_loss / n_batches}})

    return model


def lock_backbone(model, args):
    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'nnclr', 'tstcc']:
        trained_backbone = model.encoder
    else:
        NotImplementedError

    return trained_backbone


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

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
                loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)

                if target.shape != predicted.shape:  # openpack
                    target = target[:,0,0]

                total_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save model
        model_dir = model_dir_name + '/lincls_' + target_user + '_' + args.model_name + str(epoch) + '.pt'
        print('Saving model at {} epoch to {}'.format(epoch, model_dir))
        torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir)

        acc_train = float(correct) * 100.0 / total
        logger.debug(f'epoch train loss     : {total_loss:.4f}, train acc     : {acc_train:.4f}')
        writer.add_scalar('scalar/%s/downTrainLoss'%target_user, total_loss, epoch)
        writer.add_scalar('scalar/%s/downTrainAcc'%target_user, acc_train, epoch)
        fitlog.add_loss(total_loss, name="Train Loss of %s"%target_user, step=epoch)
        fitlog.add_metric({"dev": {"Train Acc of %s"%target_user: acc_train}}, step=epoch)

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                total_loss = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                    loss, predicted, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion)

                    if target.shape != predicted.shape:  # openpack
                        target = target[:, 0, 0]

                    total_loss += loss.item()
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
                    print('update')
                logger.debug(f'epoch val loss     : {total_loss:.4f}, val acc     : {acc_val:.4f}')
                writer.add_scalar('scalar/%s/downValLoss'%target_user, total_loss, epoch)
                writer.add_scalar('scalar/%s/downValAcc'%target_user, acc_val, epoch)
                fitlog.add_loss(total_loss, name="Val Loss of %s"%target_user, step=epoch)
                fitlog.add_metric({"dev": {"Val Acc of %s"%target_user: acc_val}}, step=epoch)
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
                target = target[:,0,0]

            if feats is None:
                feats = feat
            else:
                feats = torch.cat((feats, feat), 0)
            trgs = np.append(trgs, target.data.cpu().numpy())
            preds = np.append(preds, predicted.data.cpu().numpy())
            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total += target.size(0)
            correct += (predicted == target).sum()
        acc_test = float(correct) * 100.0 / total

        miF = f1_score(trgs, preds, average='micro') * 100
        maF = f1_score(trgs, preds, average='weighted') * 100

        logger.debug(f'epoch test loss     : {total_loss:.4f}, test acc     : {acc_test:.4f}, microF1     : {miF:.4f}, weightedF1     : {maF:.4f}')

        fitlog.add_best_metric({"dev": {"Test Loss of %s"%target_user: total_loss}})
        fitlog.add_best_metric({"dev": {"Test Acc of %s"%target_user: acc_test}})
        fitlog.add_best_metric({"dev": {"miF of %s"%target_user: miF}})
        fitlog.add_best_metric({"dev": {"maF of %s"%target_user: maF}})

        print('confusion matrix of %s:'%target_user)
        logger.debug(confusion_matrix)
        print('accuracy of each class:')
        logger.debug(confusion_matrix.diag() / confusion_matrix.sum(1))

    if plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + '/' + args.model_name + '_confmatrix.png')
        print('plots saved to ', plot_dir_name)

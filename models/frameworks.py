import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from .backbones import *
from .TC import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter


class SimCLR(nn.Module):
    def __init__(self, backbone, dim=128):
        super(SimCLR, self).__init__()

        self.encoder = backbone
        self.len = backbone.datalen
        self.bb_dim = self.encoder.out_dim   # backbone的输出维度，只和backbone结构有关
        self.projector = Projector(model='SimCLR', bb_dim=self.bb_dim,
                                   prev_dim=self.bb_dim, dim=dim)
        # because CNNRNN backbone outputs same length of input,
        # while contrastive methods outputs one vector for each input.
        self.projector_CNNRNN = Projector(model='SimCLR', bb_dim=self.bb_dim * self.len,
                                          prev_dim=self.bb_dim, dim=dim)

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)  # DeepConvLSTM x(B,T,D)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:  # 将三维(B,T,D)变成两维(B,T*D)，由于从同一个data augumented，Projector标签相同
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)
        if self.encoder.__class__.__name__ in ['CNNRNN']:
            z1 = self.projector_CNNRNN(z1)
            z2 = self.projector_CNNRNN(z2)
        else:
            z1 = self.projector(z1)
            z2 = self.projector(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, z1, z2
        else:
            return z1, z2

class cnnrnnprojector(nn.Module):
    def __init__(self, bb, out):
        super(cnnrnnprojector, self).__init__()
        self.bb_dim = bb
        self.out_dim = out
        self.conv = nn.Conv1d(self.bb_dim, 256, kernel_size=5, padding=2)
        self.linear3 = nn.Linear(256, self.out_dim)
        self.bn = nn.BatchNorm1d(180)
        self.active = nn.PReLU()
        # self.active = nn.ReLU()
        return

    def forward(self, x): #128,1080,128
        x1 = x.permute(0,2,1) #x1=128,128,1080
        x = self.conv(x1) # x=128,256,1080
        x = x.permute(0, 2, 1) # x=128,1080,256
        x = self.active(x) # x=128,1080,256
        x = self.linear3(x) # x=128,1080,13
        x = self.active(x)
        return x


class unetprojector(nn.Module):
    def __init__(self, bb, out):
        super(unetprojector, self).__init__()
        self.bb_dim = bb
        self.out_dim = out
        self.conv = nn.Conv1d(self.bb_dim, 256, kernel_size=5, padding=2)
        self.linear3 = nn.Linear(256, self.out_dim)
        self.bn = nn.BatchNorm1d(180)
        self.active = nn.ReLU()
        return

    def forward(self, x):  # batch, 64, len180
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.active(x)
        x = self.linear3(x)
        x = self.active(x)
        return x  # batch, len180, 96


class CNNRNN_SSLframe(nn.Module):  # ssl model
    def __init__(self, backbone):
        super(CNNRNN_SSLframe, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.out_dim = self.encoder.simi_dim

        # self.projector = nn.Sequential(nn.Linear(self.bb_dim, self.out_dim),
        #                                nn.ReLU(),
        #                                )
        # self.projector = nn.Sequential(nn.Linear(self.bb_dim, 256),
        #                                nn.ReLU(),
        #                                nn.Linear(256, 128),
        #                                nn.ReLU(),
        #                                nn.Linear(128, self.out_dim),
        #                                nn.ReLU(),
        #                                )
        self.projector = cnnrnnprojector(self.bb_dim, self.out_dim)

    #     # IDEC
    #     self.alpha = 1.0
    #     self.n_clusters = 20
    #     self.n_z = self.bb_dim
    #     self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, self.n_z))
    #     torch.nn.init.xavier_normal_(self.cluster_layer.data)
    #
    # def target_distribution(self, q):
    #     weight = q ** 2 / q.sum(0)
    #     return (weight.t() / weight.sum(1)).t()

    def forward(self, x):  # x: batch, len_sw, dim
        _, z1 = self.encoder(x)
        out = self.projector(z1)  # batch, dim', len

        # IDEC
        # # cluster
        # q = 1.0 / (1.0 + torch.sum(
        #     torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        # q = q.pow((self.alpha + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()
        return out


class unet_SSLframe(nn.Module):  # ssl model
    def __init__(self, backbone):
        super(unet_SSLframe, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim  # 128
        self.out_dim = self.encoder.simi_dim  #96
        self.projector = unetprojector(self.bb_dim, self.out_dim)

        # self.n_classes = backbone.n_classes
        # self.unetclassifier = backbone.projector

    def forward(self, x):  # x: batch, len_sw, dim
        _, z1 = self.encoder(x)
        # out = self.unetclassifier(z1)  # batch, dim', len
        out = self.projector(z1)  # batch, dim, len
        return out  # batch, len, dim


class CNN_AEframe(nn.Module):  # ssl model
    def __init__(self, args, backbone):
        super(CNN_AEframe, self).__init__()

        self.backbone = backbone  # encoder and decoder
        # backbone 为autoencoder，encoder为batch,128,15
        # 对15，up pooling到datalength 180
        self.datalen = args.len_sw  #180
        # transpose后，对128维度加linear为n_classes
        self.bb_dim = self.backbone.encoder.out_dim  #128
        self.out_dim = args.n_class

        self.encoder = backbone.encoder

    def forward(self, x):  # x: batch, len_sw, dim
        x_decoded, x_encoded = self.backbone(x)  # x_decoded:batch,len180,dim6, x_encoded(batch,dim128,len15)
        return x_decoded



class CNN_SSLframe(nn.Module):  # ssl model
    def __init__(self, backbone):
        super(CNN_SSLframe, self).__init__()

        self.encoder = backbone
        # self.maxpool = nn.MaxPool2d(kernel_size=(1080,1), padding=0,
        self.maxpool = nn.MaxPool2d(kernel_size=(180, 1), padding=0,
                                    return_indices=False)
        # -----------pretrain multitask classifiers-
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(96, 256), #96 for others,6 for multitask
                                                    nn.ReLU(),
                                                    nn.Linear(256, 1),
                                                    nn.Sigmoid()) for i in range(7)])
        # self.adjustlen = nn.Linear(1083, 1080)
        return

    def forward(self, x, aug_xlist):  # x: batch, len_sw, dim
        xlist = []
        for i, layer in enumerate(self.linears):
            z = self.encoder(x)  # input(B,Len,Ch), out (B,Dim,1,Len)
            # z = self.adjustlen(z)
            z_in = z.squeeze(2).permute(0, 2, 1)
            # z_in = self.adjustlen(z_in)
            z_out = self.maxpool(z_in)  # out(B,1,dim)
            z_out = z_out.squeeze(1)
            tmp_raw = layer(z_out)

            z_aug = self.encoder(aug_xlist[i])
            z_aug = z_aug.squeeze(2).permute(0, 2, 1)
            z_aug = self.maxpool(z_aug)
            z_aug = z_aug.squeeze(1)
            tmp_aug = layer(z_aug)
            concate_x = torch.concatenate([tmp_raw, tmp_aug], dim=0)
            xlist.append(concate_x)
        return xlist


class Base_SSLframe(nn.Module):
    def __init__(self, args, backbone):
        super(Base_SSLframe, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        if (args.framework=='SSL') and (args.backbone =='Transformer'):
            self.out_dim = args.n_feature
        else:
            self.out_dim = args.out_fea  # SSL framework, output dim = sensor data dim, proposeSSL=simidim

        self.projector = nn.Sequential(nn.Linear(self.bb_dim, self.out_dim),
                                       nn.ReLU(), )

    def forward(self, x):  # x: batch, len_sw, dim
        _, z1 = self.encoder(x)
        out = self.projector(z1)  # batch, dim', len
        return out


class NNCLR(nn.Module):
    def __init__(self, backbone, dim=128, pred_dim=64):
        super(NNCLR, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(model='NNCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.predictor = Predictor(model='NNCLR', dim=dim, pred_dim=pred_dim)

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, p1, p2, z1.detach(), z2.detach()
        else:
            return p1, p2, z1.detach(), z2.detach()


class BYOL(nn.Module):
    def __init__(
            self,
            DEVICE,
            backbone,
            window_size=30,
            n_channels=77,
            hidden_layer=-1,
            projection_size=64,
            projection_hidden_size=256,
            moving_average=0.99,
            use_momentum=True,
    ):
        super().__init__()

        net = backbone
        self.bb_dim = net.out_dim  # SSL的backbone模型的输出维度，不加projector
        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, DEVICE=DEVICE,
                                         layer=hidden_layer)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average)  # momentum更新

        self.online_predictor = Predictor(model='byol', dim=projection_size, pred_dim=projection_hidden_size)

        self.to(DEVICE)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, window_size, n_channels, device=DEVICE),
                     torch.randn(2, window_size, n_channels, device=DEVICE))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
            self,
            x1,
            x2,
            return_embedding=False,
            return_projection=True,
            require_lat=False
    ):
        assert not (self.training and x1.shape[
            0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x1, return_projection=return_projection)

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            online_proj_one, x1_decoded, lat1 = self.online_encoder(x1)
            online_proj_two, x2_decoded, lat2 = self.online_encoder(x2)
        else:
            online_proj_one, lat1 = self.online_encoder(x1)  # raw data to (B,Dim)
            online_proj_two, lat2 = self.online_encoder(x2)

        online_pred_one = self.online_predictor(online_proj_one)  # input,output(B,Dim)同
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
                target_proj_one, _, _ = target_encoder(x1)
                target_proj_two, _, _ = target_encoder(x2)
            else:
                target_proj_one, _ = target_encoder(x1)  # input raw data, out(B,Dim)
                target_proj_two, _ = target_encoder(x2)
            target_proj_one.detach_()
            target_proj_two.detach_()

        if self.online_encoder.net.__class__.__name__ in ['AE', 'CNN_AE']:
            if require_lat:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2
            else:
                return x1_decoded, x2_decoded, online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()
        else:
            if require_lat:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach(), lat1, lat2  # lat:latent representation,后两位相乘
            else:
                return online_pred_one, online_pred_two, target_proj_one.detach(), target_proj_two.detach()  # return(B,Dim)


class TSTCC(nn.Module):
    def __init__(self, backbone, DEVICE, temp_unit='tsfm', tc_hidden=100):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(TSTCC, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_channels
        self.TC = TC(self.bb_dim, DEVICE, tc_hidden=tc_hidden, temp_unit=temp_unit).to(DEVICE)
        self.projector = Projector(model='TS-TCC', bb_dim=self.bb_dim, prev_dim=None, dim=tc_hidden)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        _, z1 = self.encoder(x1)
        _, z2 = self.encoder(x2)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        nce1, c_t1 = self.TC(z1, z2)
        nce2, c_t2 = self.TC(z2, z1)

        p1 = self.projector(c_t1)
        p2 = self.projector(c_t2)

        return nce1, nce2, p1, p2

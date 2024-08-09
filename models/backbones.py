import torch
from torch import nn
from .attention import *
from .MMB import *
import matplotlib.pyplot as plt


class FCN(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(FCN, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        if n_channels == 9: # ucihar
            self.out_len = 18
        elif n_channels == 3: # shar
            self.out_len = 21
        if n_channels == 6: # hhar
            self.out_len = 15

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if backbone == False:
            self.logits = nn.Linear(self.out_len * out_channels, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x

class MLP(nn.Module):
    def __init__(self, args, n_class):
        super(MLP, self).__init__()
        if args.backbone in ['CNN', 'CNNRNN']:
            indim = 96
            # bn = 183
            bn = 1083  # multi task only, the same as the sliding window length
        elif args.backbone == 'LSTM':
            indim = 128
            bn = 1080
        self.classifier = nn.Sequential(
            nn.Linear(indim, 256),
            nn.BatchNorm1d(bn),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(bn),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(128, n_class),
            nn.BatchNorm1d(bn),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

    def forward(self,x):
        y = self.classifier(x)
        return y

# class CNN(nn.Module):
#     def __init__(self, args, n_channels, n_classes):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(n_channels, 32, (1, 24),
#                                bias=False, padding=(0, 12))
#         self.conv2 = nn.Conv2d(32, 64, (1, 16), bias=False, padding=(0, 8))
#         self.conv3 = nn.Conv2d(64, 96, (1, 8), bias=False, padding=(0, 4))
#         self.BN1 = nn.BatchNorm2d(32)
#         self.BN2 = nn.BatchNorm2d(64)
#         self.BN3 = nn.BatchNorm2d(96)
#         self.dropout = nn.Dropout(0.1)
#         self.activation = nn.ReLU()
#         # self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0, return_indices=False)
#
#         self.out_dim = 1083
#         # self.linear1083 = nn.Linear(1083,1080)
#         # self.simi_dim = 1
#
#         #-----------framework layer---------------
#         # input(B,dim,1,Len)
#         self.classifier = nn.Linear(self.out_dim, 1080)
#         # self.classifier = nn.Linear(self.out_dim, n_classes)
#         # self.classifier = MLP(args,n_classes)
#
#     def forward(self, x):
#         x = x.unsqueeze(2).permute(0, 3, 2, 1)
#         x = self.activation(self.BN1(self.conv1(x)))  # input (B60, CH6, 1, Len45)
#         x = self.dropout(x)
#         x = self.activation(self.BN2(self.conv2(x)))
#         x = self.dropout(x)
#         x = self.activation(self.BN3(self.conv3(x)))
#         # x = self.maxpool(x)  # input size (N,C,H,W), output(N,C,Ho,Wo)
#         # x = x.squeeze(2)
#         # for multi task experiment only:
#         # x = self.linear1083(x)
#
#         # x = x.permute(0, 2, 1)
#         x = self.classifier(x)
#         # x = x.permute(0, 2, 1)
#         # return 0, x
#         return x  # for multi task experiment only:

class CNN(nn.Module):
    def __init__(self, n_channels, n_classes, backbone=True):
        super(CNN, self).__init__()
        self.backbone = backbone  # ssl model
        self.conv1 = nn.Conv2d(n_channels, 32, (1, 24), bias=False, padding=(0, 24 // 2))
        self.conv2 = nn.Conv2d(32, 64, (1, 16), bias=False, padding=(0, 16 // 2))
        self.conv3 = nn.Conv2d(64, 96, (1, 8), bias=False, padding=(0, 8 // 2))
        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
        self.BN3 = nn.BatchNorm2d(96)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0, return_indices=False)

        self.out_dim = 96
        # self.simi_dim = 1
        self.reshape = nn.Linear(1083,1080)
        #-----------framework layer---------------
        # input(B,dim,1,Len)
        self.classifier = nn.Linear(self.out_dim, n_classes)


    def forward(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = self.activation(self.BN1(self.conv1(x)))  # input (B60, CH6, 1, Len45)
        x = self.dropout(x)
        x = self.activation(self.BN2(self.conv2(x)))
        x = self.dropout(x)
        out = self.activation(self.BN3(self.conv3(x)))
        # x = self.maxpool(x)  # input size (N,C,H,W), output(N,C,Ho,Wo)
        # x = self.reshape(x)
        if self.backbone==False:
            out = out.squeeze(2)
            out = out.permute(0, 2, 1)
            return out, x
        else:
            return out


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True):
        super(DeepConvLSTM, self).__init__()

        self.backbone = backbone

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.out_dim = LSTM_units

        if backbone == False:
            self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):  # x(Batch, Len, Dim)
        self.lstm.flatten_parameters()  #为了提高内存的利用率和效率
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))  # out(batch, 64, Len, dim)

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)  # x(len, batch, 128)
        x = x[-1, :, :]

        # return None, x
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x


class CNNRNN(nn.Module):
    def __init__(self, args, n_channels, n_classes, simi_dim=30,
                 conv_kernels=64, kernel_size=5, datalen=128, backbone=True):
        super(CNNRNN, self).__init__()

        self.backbone = backbone  # ssl model
        self.datalen = datalen
        self.conv1 = nn.Conv2d(n_channels, conv_kernels, (1, kernel_size),
                               bias=False, padding=(0,kernel_size//2))
        # self.conv1_dilated1 = nn.Conv2d(n_channels, conv_kernels//2, (1, kernel_size),
        #                                 dilation=(1, 4),
        #                        bias=False, padding=(0, 8))
        # self.conv1_dilated2 = nn.Conv2d(n_channels, conv_kernels//2, (1, kernel_size),
        #                                F dilation=8,
        #                                 bias=False, padding=(0, 16))
        # self.conv1_dilated3 = nn.Conv2d(n_channels, conv_kernels//2, (1, kernel_size),
        #                                 dilation=16,
        #                                 bias=False, padding=(0, 32))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (1, kernel_size),
                               bias=False, padding=(0,kernel_size//2))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (1, kernel_size),
                               bias=False, padding=(0,kernel_size//2))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (1, kernel_size),
                               bias=False, padding=(0,kernel_size//2))
        self.BN1 = nn.BatchNorm2d(conv_kernels)
        self.BN2 = nn.BatchNorm2d(conv_kernels)
        self.BN3 = nn.BatchNorm2d(conv_kernels)
        self.BN4 = nn.BatchNorm2d(conv_kernels)

        self.dropout = nn.Dropout(0.5)
        LSTM_units = conv_kernels
        self.lstm6 = nn.LSTM(conv_kernels, LSTM_units, batch_first=True, bidirectional=True)
        # because it is bidirectional, the input dim is two times
        self.lstm7 = nn.LSTM(LSTM_units*2, LSTM_units, batch_first=True, bidirectional=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.5)

        self.out_dim = LSTM_units*2
        self.simi_dim = simi_dim
        # 需要加ssl 和 supervised learning. 在trainer_CNNRNN全更新了
        self.classifier = nn.Linear(self.out_dim, n_classes)
        # if args.classifierLayer == 'linear':
        #     self.classifier = nn.Linear(self.out_dim, n_classes)
        # elif args.classifierLayer == 'MLP':
        #     self.classifier = nn.Sequential(
        #             nn.Linear(128, 256),
        #             nn.BatchNorm1d(180),
        #             nn.Dropout(0.5),
        #             nn.ReLU(),
        #             nn.Linear(256, 128),
        #             nn.BatchNorm1d(180),
        #             nn.Dropout(0.5),
        #             nn.Sigmoid(),
        #             nn.Linear(128, n_classes),  #args.n_class
        #             nn.BatchNorm1d(180),
        #             nn.Dropout(0.3),
        #             nn.Sigmoid()
        #         )


        self.activation = nn.PReLU()
        # self.activation = nn.ReLU()
        return

    def forward(self, x):  # x(Batch, Len, CH)
        self.lstm6.flatten_parameters()  #为了提高内存的利用率和效率
        self.lstm7.flatten_parameters()  #为了提高内存的利用率和效率

        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x1 = self.activation(self.BN1(self.conv1(x)))  # input (B60, CH6, 1, Len45)
        # x1 = self.BN1(self.conv1(x))  # input (B60, CH6, 1, Len360)
        # x2 = self.BN1(self.conv1_dilated1(x))
        # x3 = self.BN1(self.conv1_dilated2(x))
        # x4 = self.BN1(self.conv1_dilated3(x))
        # x = torch.concatenate([x1,x2,x3,x4],dim=1)
        # x = self.activation(x)
        x2 = self.activation(self.BN2(self.conv2(x1)))
        # x3 = torch.concatenate([x1, x2], dim=1)
        x4 = self.activation(self.BN3(self.conv3(x2)))
        x5 = self.activation(self.BN4(self.conv4(x4)))  # output (B, dim, 1, Len)

        x = self.dropout(x5)

        # -- [2] LSTM --
        # Reshape: (B, dim, 1, Len) -> (B, Len, dim64)
        x = x.squeeze(2).transpose(1, 2)

        x, _ = self.lstm6(x)  # input (B, Len, dim)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)  # output (B, Len, dim*2=128)

        if not self.backbone:  # for pure supervised learning
            out = self.classifier(x)
            return x, out  # (B, T, Dim)
        else:
            return _, x



class LSTM(nn.Module):
    def __init__(self,args, n_channels, n_classes, LSTM_units=128):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(n_channels, LSTM_units, num_layers=2)
        self.out_dim = LSTM_units

        # if backbone == False:
        #     self.classifier = nn.Linear(LSTM_units, n_classes)
        self.classifier = MLP(args,n_classes)

    def forward(self, x):  # input(B,L180,dim6)
        self.lstm.flatten_parameters()
        # x = x.permute(1, 0, 2)
        x, (h, c) = self.lstm(x)
        # x = x[-1, :, :]

        # if self.backbone:
        #     return None, x
        # else:
        out = self.classifier(x)
        return x, out

class AE(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, outdim=128, backbone=True):
        super(AE, self).__init__()

        self.backbone = backbone
        self.len_sw = len_sw

        self.e1 = nn.Linear(n_channels, 8)
        self.e2 = nn.Linear(8 * len_sw, 2 * len_sw)
        self.e3 = nn.Linear(2 * len_sw, outdim)

        self.d1 = nn.Linear(outdim, 2 * len_sw)
        self.d2 = nn.Linear(2 * len_sw, 8 * len_sw)
        self.d3 = nn.Linear(8, n_channels)

        self.out_dim = outdim

        if backbone == False:
            self.classifier = nn.Linear(outdim, n_classes)

    def forward(self, x):
        x_e1 = self.e1(x)
        x_e1 = x_e1.reshape(x_e1.shape[0], -1)
        x_e2 = self.e2(x_e1)
        x_encoded = self.e3(x_e2)

        x_d1 = self.d1(x_encoded)
        x_d2 = self.d2(x_d1)
        x_d2 = x_d2.reshape(x_d2.shape[0], self.len_sw, 8)
        x_decoded = self.d3(x_d2)

        if self.backbone:
            return x_decoded, x_encoded
        else:
            out = self.classifier(x_encoded)
            return out, x_decoded

class CNN_AE_encoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_encoder, self).__init__()

        self.n_channels = n_channels
        # self.datalen = 180  #args.len_sw
        # self.n_classes = n_classes   # check if correct a

        kernel_size = 5
        self.e_conv1 = nn.Sequential(nn.Conv2d(n_channels, 16,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size//2)),
                                     nn.BatchNorm2d(16),
                                     nn.Tanh())  # Tanh is MoIL paper
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout = nn.Dropout(0.3)  # probability of samples to be zero
        # v = 0.8
        # self.dropout = nn.Dropout(v)  # probability of samples to be zero
        # print(v)
        self.e_conv2 = nn.Sequential(nn.Conv2d(16, 16,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size//2)),
                                     nn.BatchNorm2d(16),
                                     nn.Tanh())
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.e_conv3 = nn.Sequential(nn.Conv2d(16, out_channels,
                                               (1, kernel_size), bias=False,
                                               padding=(0, kernel_size//2)),
                                     nn.BatchNorm2d(out_channels),
                                     nn.PReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0, return_indices=True)

        self.out_samples = 15
        self.out_dim = out_channels

        # backbone 为autoencoder，encoder为batch,128,15
        # 对15，up pooling到datalength 180
        self.datalen =180  #180
        # transpose后，对128维度加linear为n_classes
        self.bb_dim = 128  #128
        self.out_dim = 12#args.n_class

        kernelsize = int(self.datalen / self.out_samples)
        self.upsample = nn.Upsample(scale_factor=kernelsize, mode='nearest')
        self.classifier = nn.Linear(self.bb_dim, self.out_dim)
        return

    def forward(self, x):  #x(batch,len180,dim6)
        x = x.unsqueeze(2).permute(0, 3, 2, 1)  # outx(batch,dim,1,len)
        x1 = self.e_conv1(x)  #x1(batch,64,1,180)
        x1 = x1.squeeze(2)  # batch,32,180
        x, indice1 = self.pool1(x1)  # (batch,32,90)len减半,最后一维maxpool
        x = x.unsqueeze(2)
        x = self.dropout(x)
        #---------
        x2 = self.e_conv2(x)  # batch,64,90
        x2 = x2.squeeze(2)
        x, indice2 = self.pool2(x2)
        x = x.unsqueeze(2)  # batch,64,45
        x = self.dropout(x)
        # ---------
        x3 = self.e_conv3(x)  # batch,128,45
        x3 = x3.squeeze(2)
        x_encoded, indice3 = self.pool3(x3)  #xencoded(batch,128,15)
        # x_encoded # batch,128,15
        return x_encoded, [indice1, indice2, indice3]


class CNN_AE_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE_decoder, self).__init__()

        self.n_channels = n_channels
        kernel_size = 5
        self.unpool1 = nn.MaxUnpool1d(kernel_size=3, stride=3, padding=0)
        self.d_conv1 = nn.Sequential(nn.ConvTranspose2d(out_channels, 16,
                                                        kernel_size=(1, kernel_size),
                                                        bias=False,
                                                        padding=(0, kernel_size//2)),
                                     nn.BatchNorm2d(16),
                                     nn.Tanh())

        if n_channels == 9:  # ucihar
            self.lin1 = nn.Linear(33, 34)
        elif n_channels == 6:  # hhar
            self.lin1 = nn.Identity()
            # self.lin1 = nn.Linear(47, 48)
        elif n_channels == 3:  # shar
            self.lin1 = nn.Linear(39, 40)

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv2 = nn.Sequential(nn.ConvTranspose2d(16, 16,
                                                        kernel_size=(1, kernel_size),
                                                        stride=1, bias=False,
                                                        padding=(0, kernel_size//2)),
                                     nn.BatchNorm2d(16),
                                     nn.PReLU())

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        self.d_conv3 = nn.Sequential(nn.ConvTranspose2d(16, n_channels,
                                                        kernel_size=(1, kernel_size),
                                                        stride=1, bias=False,
                                                        padding=(0, kernel_size//2)),
                                     nn.BatchNorm2d(n_channels),
                                     nn.PReLU())

        return

    def forward(self, x_encoded, encode_indices):  #x_encoded(batch, 128, 25)
        x = self.unpool1(x_encoded, encode_indices[-1]) #out(batch, 64, 47)
        x = x.unsqueeze(2)
        x = self.d_conv1(x)  # out(batch, 128, 45)
        x = x.squeeze(2)
        # x = self.lin1(x)
        # ---------
        x = self.unpool2(x, encode_indices[-2])  #out(batch, 64, 90)
        x = x.unsqueeze(2)
        x = self.d_conv2(x)  #out(batch, 32, 91)
        x = x.squeeze(2)
        # ---------
        x = self.unpool3(x, encode_indices[0])  #x_decoded(batch,32,180)
        x = x.unsqueeze(2)
        x_decoded = self.d_conv3(x)
        x_decoded = x_decoded.squeeze(2)  # batch, 6, 180 = AE input
        return x_decoded

class CNN_AE(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(CNN_AE, self).__init__()

        self.backbone = backbone
        self.n_channels = n_channels  # input data dimension

        # if n_channels == 9:  # ucihar
        #     self.lin2 = nn.Linear(127, 128)
        #     self.out_dim = 18 * out_channels
        # elif n_channels == 6:  # hhar
        self.lin2 = nn.Identity()
        self.out_dim = 25 * out_channels
        # elif n_channels == 3:  # shar
        #     self.out_dim = 21 * out_channels

        self.encoder = CNN_AE_encoder(n_channels, n_classes,
                                      out_channels=128, backbone=True)
        self.decoder = CNN_AE_decoder(n_channels, n_classes,
                                      out_channels=128, backbone=True)

        # # if backbone == False:
        self.classifier = self.encoder.classifier
        # # self.out_dim

        return

    def forward(self, x):  # x(batch, len180, dim6)
        x_encoded, encode_indices = self.encoder(x)  # x_encoded(batch, 128, 25)
        decod_out = self.decoder(x_encoded, encode_indices)  # x_decoded(batch, 6, 179)

        # if self.n_channels == 3:  # ucihar
        #     x_decoded = self.lin2(decod_out)
        # elif self.n_channels == 6:  # hhar
        x_decoded = self.lin2(decod_out)

        x_decoded = x_decoded.permute(0, 2, 1)
        # x_decoded(batch, 180, 6), x_encoded(batch, 128, 15)
        return x_decoded, x_encoded
        # if self.backbone:
        #     return x_decoded, x_encoded
        # else:
        #     out = self.classifier(x_encoded)
        #     return out, x_decoded


class unet_encoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(unet_encoder, self).__init__()

        kernel_size = 3
        self.startcnn = nn.Sequential(nn.Conv2d(n_channels, 32,
                                               (1,kernel_size), bias=False,
                                               padding=(0,kernel_size // 2)),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(32, 64,
                                                (1,kernel_size), bias=False,
                                                padding=(0,kernel_size // 2)),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU()
                                   )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128,
                                             (1,kernel_size), bias=False,
                                             padding=(0,kernel_size // 2)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU()
                                   )

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256,
                                             (1,kernel_size), bias=False,
                                             padding=(0,kernel_size // 2)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
                                   )

        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0, return_indices=True)

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512,
                                             (1,kernel_size), bias=False,
                                             padding=(0,kernel_size // 2)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU()
                                   )

        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=3, padding=0, return_indices=True)

        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024,
                                             (1,kernel_size), bias=False,
                                             padding=(0,kernel_size // 2)),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU()
                                   )

        self.pool5 = nn.MaxPool1d(kernel_size=5, stride=5, padding=0, return_indices=True)

    def forward(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = self.startcnn(x)
        x = self.conv1(x)
        x1 = x.squeeze(2)  # batch,32,180
        x1, indice1 = self.pool1(x1)

        x1 = x1.unsqueeze(2)
        x1 = self.conv2(x1)
        x2 = x1.squeeze(2)
        x2, indice2 = self.pool2(x2)

        x2 = x2.unsqueeze(2)
        x2 = self.conv3(x2)
        x3 = x2.squeeze(2)
        x3, indice3 = self.pool3(x3)

        x3 = x3.unsqueeze(2)
        x3 = self.conv4(x3)
        x4 = x3.squeeze(2)
        x4, indice4 = self.pool4(x4)

        x4 = x4.unsqueeze(2)
        x4 = self.conv5(x4)
        x5 = x4.squeeze(2)
        # x5, indice5 = self.pool5(x5)
        return x5, [x,x1,x2,x3,x4]


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class unet_classifier(nn.Module):
    def __init__(self, n_classes):
        super(unet_classifier, self).__init__()
        self.classifier = nn.Conv2d(64, n_classes,
                                    kernel_size=(1, 1), bias=False)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x5 = x.unsqueeze(2)
        x5 = self.classifier(x5)
        x5 = x5.squeeze(2)
        x5 = x5.permute(0, 2, 1)
        return x5  # (B,L180,Class11)

class unet_decoder(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True):
        super(unet_decoder, self).__init__()

        kernel_size = 3

        self.cnnt1 = nn.Sequential(
                                nn.Upsample(scale_factor=(1,3),
                                            mode='bilinear', align_corners=True),
                                nn.BatchNorm2d(1024),
                                nn.Conv2d(1024, 512,
                                          (1, kernel_size), bias=False,
                                          padding=(0, kernel_size // 2)),
                                nn.BatchNorm2d(512),
                                )

        self.conv1 = nn.Sequential(nn.Conv2d(1024, 512,
                                             (1, kernel_size), bias=False,
                                             padding=(0, kernel_size//2)),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU()
                                   )
        self.att5 = Attention_block(F_g=512, F_l=512, F_int=256)

        self.cnnt2 = nn.Sequential(
                                nn.Upsample(scale_factor=(1,3),
                                            mode='bilinear', align_corners=True),
                                nn.BatchNorm2d(512),
                                nn.Conv2d(512, 256,
                                          (1, kernel_size), bias=False,
                                          padding=(0, kernel_size // 2)),
                                nn.BatchNorm2d(256),
                                )

        self.conv2 = nn.Sequential(nn.Conv2d(512, 256,
                                             (1, kernel_size), bias=False,
                                             padding=(0, kernel_size//2)),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
                                   )
        self.att4 = Attention_block(F_g=256, F_l=256, F_int=128)

        self.cnnt3 = nn.Sequential(
                                nn.Upsample(scale_factor=(1,2),
                                            mode='bilinear', align_corners=True),
                                nn.BatchNorm2d(256),
                                nn.Conv2d(256, 128,
                                          (1, kernel_size), bias=False,
                                          padding=(0, kernel_size // 2)),
                                nn.BatchNorm2d(128),
                                )

        self.conv3 = nn.Sequential(nn.Conv2d(256, 128,
                                             (1, kernel_size), bias=False,
                                             padding=(0, kernel_size//2)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU()
                                   )
        self.att3 = Attention_block(F_g=128, F_l=128, F_int=64)

        self.cnnt4 = nn.Sequential(
                                nn.Upsample(scale_factor=(1,2),
                                            mode='bilinear', align_corners=True),
                                nn.BatchNorm2d(128),
                                nn.Conv2d(128, 64,
                                          (1, kernel_size), bias=False,
                                          padding=(0, kernel_size // 2)),
                                nn.BatchNorm2d(64),
                                )

        self.conv4 = nn.Sequential(nn.Conv2d(128, 64,
                                             (1, kernel_size), bias=False,
                                             padding=(0, kernel_size//2)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU()
                                   )
        self.att2 = Attention_block(F_g=64, F_l=64, F_int=32)

        self.cnnt5 = nn.Sequential(
                                nn.Upsample(scale_factor=(1,2),
                                            mode='bilinear', align_corners=True),
                                nn.BatchNorm2d(64),
                                nn.Conv2d(64, 32,
                                          (1, kernel_size), bias=False,
                                          padding=(0, kernel_size // 2)),
                                nn.BatchNorm2d(32),
                                )

        # self.classifier = nn.Conv2d(64, n_classes,
        #                              kernel_size=(1, 1), bias=False)
        # self.classifier = unet_classifier(n_classes)

    def forward(self, x, indices, if_backbone=False):
        [ex,ex1,ex2,ex3,ex4] = indices
        x = x.unsqueeze(2)
        x0 = self.cnnt1(x)
        x0 = self.att5(ex3, x0)
        x0 = torch.cat([ex3, x0], dim=1)  #512,15;512,15
        x = self.conv1(x0)
        x1 = x.squeeze(2)
        # x1,_ = self.pool1(x)

        x1 = x1.unsqueeze(2)
        x1 = self.cnnt2(x1)
        x1 = self.att4(ex2, x1)
        x1 = torch.cat([ex2, x1], dim=1)
        x1 = self.conv2(x1)
        x2 = x1.squeeze(2)
        # x2,_ = self.pool2(x1)

        x2 = x2.unsqueeze(2)
        x2 = self.cnnt3(x2)
        x2 = self.att3(ex1, x2)
        x2 = torch.cat([ex1, x2], dim=1)
        x2 = self.conv3(x2)
        x3 = x2.squeeze(2)
        # x3,_ = self.pool3(x2)

        x3 = x3.unsqueeze(2)
        x3 = self.cnnt4(x3)
        x3 = self.att2(ex, x3)
        x3 = torch.cat([ex, x3], dim=1)
        x3 = self.conv4(x3)
        x4 = x3.squeeze(2)

        return x4  # batch, 64, len180
        # if if_backbone:  # do not concatenate classifier layers
        #     return x4
        # else:
        #     x5 = self.classifier(x4)
        #     return x5


class unet(nn.Module):
    def __init__(self, args, n_channels, n_classes, out_channels=128, backbone=True):
        super(unet, self).__init__()
        self.n_classes = n_classes
        self.encoder = unet_encoder(n_channels, n_classes, out_channels=128, backbone=True)
        self.decoder = unet_decoder(n_channels, n_classes, out_channels=128, backbone=True)

    def forward(self, x):  # x(B,L180,D6)
        encode, indices = self.encoder(x)  # encode(B, 1024, 5)
        # [indice1, indice2, indice3, indice4, indice5] = indices
        result = self.decoder(encode, indices)
        return result


class unet_backbone(nn.Module):
    def __init__(self, args, n_channels, n_classes, out_channels=128, simi_dim=30, backbone=True):
        super(unet_backbone, self).__init__()
        self.backbone = backbone  # ssl model
        self.n_classes = n_classes
        self.encoder = unet_encoder(n_channels, n_classes, out_channels=128, backbone=True)
        self.decoder = unet_decoder(n_channels, n_classes, out_channels=128, backbone=True)
        # self.classifier = unet_classifier(n_classes)
        # self.out_dim = 128
        self.out_dim = 64  # output dimension of backbone
        self.simi_dim = simi_dim
        self.classifier = nn.Linear(self.out_dim, n_classes)
        # self.projector = unet_classifier(args.num_motifs)


    def forward(self, x):  # x(B,L180,D6)
        encode, indices = self.encoder(x)  # encode(B, 1024, 5)
        result = self.decoder(encode, indices, if_backbone=True)

        if not self.backbone:  # for pure supervised learning
            out = self.classifier(result)
            return result, out  # (B, T, Dim)
        else:
            return 0, result
        # return result


class Transformer(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
        super(Transformer, self).__init__()

        self.backbone = backbone
        self.out_dim = dim
        self.datalen = len_sw
        self.input_dim = n_channels  #用reconstruction loss时，framework需要一个input维度
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw,
                                           n_classes=n_classes, dim=dim, depth=depth,
                                           heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        # if backbone == False:
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):  # input(B,L,D)
        x = self.transformer(x) # input(B,L,D180)
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x

class Classifier(nn.Module):
    def __init__(self, bb_dim, n_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Linear(bb_dim, n_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(bb_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, n_classes),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        out = self.classifier(x)

        return out


class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        self.bb_dim = bb_dim
        self.dim = dim
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x


class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


from functools import wraps


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projector and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, DEVICE, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.DEVICE = DEVICE

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        children = [*self.net.children()]
        print('children[self.layer]:', children[self.layer])
        return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = Projector(model='byol', bb_dim=dim, prev_dim=self.projection_hidden_size, dim=self.projection_size)
        return projector.to(hidden)

    def get_representation(self, x):

        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            x_decoded, representation = self.get_representation(x)
        else:
            _, representation = self.get_representation(x)  #input(B,Len,Dim), out same

        if len(representation.shape) == 3:
            representation = representation.reshape(representation.shape[0], -1)

        projector = self._get_projector(representation)
        projection = projector(representation)  # projector处理representation
        if self.net.__class__.__name__ in ['AE', 'CNN_AE']:
            return projection, x_decoded, representation
        else:
            return projection, representation  #2*128， 2*（128*180）。representation是backbone


class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.
    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548
    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
    """

    def __init__(self, size: int = 2 ** 16):
        super(NNMemoryBankModule, self).__init__(size)

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank
        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it
        """

        output, bank = \
            super(NNMemoryBankModule, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = \
            torch.index_select(bank, dim=0, index=index_nearest_neighbours)

        return nearest_neighbours

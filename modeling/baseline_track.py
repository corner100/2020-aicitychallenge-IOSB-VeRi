import torch
from torch import nn
from torch.nn import functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
import numpy as np
from .resnet_cbam import ResNetCBAM

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Baseline_track(nn.Module):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,seq_len,in_planes,global_feat_fc,head,train_last_base_params):
        super(Baseline_track, self).__init__()
        self.head = head
        self.in_planes = in_planes
        self.global_feat_fc = global_feat_fc
        self.train_last_base_params = train_last_base_params
        self.seq_len =seq_len
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50' or model_name == 'resnet50_track':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'resnet18-cbam':
            self.in_planes = 512
            self.base = ResNetCBAM(last_stride=last_stride,
                                   block=BasicBlock,
                                   layers=[2, 2, 2, 2])
        elif model_name == 'resnet34-cbam':
            self.in_planes = 512
            self.base = ResNetCBAM(last_stride=last_stride,
                                   block=BasicBlock,
                                   layers=[3, 4, 6, 3])
        elif model_name == 'resnet50-cbam':
            self.base = ResNetCBAM(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 4, 6, 3])
        elif model_name == 'resnet101-cbam':
            self.base = ResNetCBAM(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 4, 23, 3])
        elif model_name == 'resnet152-cbam':
            self.base = ResNetCBAM(last_stride=last_stride,
                                   block=Bottleneck,
                                   layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.fc_keypoints = nn.Linear(4*1024, 4)
        self.softmax_keypoints = nn.Softmax(dim=1)
        self.fc_global_feat = nn.Linear(in_features=self.in_planes, out_features=self.in_planes)  ####
        if self.head == "TA":
            self.track_head = TA(self.base,self.in_planes)
        elif self.head == "TA_LSTM":
            self.track_head = TA_LSTM(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA2":
            self.track_head = TA2(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA3":
            self.track_head = TA3(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA4":
            self.track_head = TA4(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA5":
            self.track_head = TA5(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA6":
            self.track_head = TA6(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA7":
            self.track_head = TA7(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA4_tanh":
            self.track_head = TA4_tanh(self.base,self.in_planes,self.seq_len)
        elif self.head == "TA5_tanh":
            self.track_head = TA5_tanh(self.base,self.in_planes,self.seq_len)

        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        if pretrain_choice == 'self-no-head':
            self.load_param_for_base(model_path)
            print('Loading pretrained self-no-head model......')
        elif pretrain_choice == 'self':
            self.load_param(model_path)
            print('Loading pretrained self model......')
        for param in list(self.base.parameters())[:-self.train_last_base_params]:
            param.requires_grad = False

    def get_attention(self, track):
        global_feat, a = self.track_head(track)
        return a

    def forward(self, track):
        global_feat,a = self.track_head(track)
        if self.global_feat_fc:
            global_feat=self.fc_global_feat(global_feat)
            global_feat = torch.tanh(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def forward_body(self, track):
        global_feat, a = self.track_head(track)
        if self.global_feat_fc:
            global_feat = self.fc_global_feat(global_feat)
            global_feat = torch.tanh(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if self.neck_feat == 'after':
            # print("Test with feature after BN")
            return feat
        else:
            # print("Test with feature before BN")
            return global_feat

    def load_param_for_base(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            if 'classifier' in k:
                # print(i[0])
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])

    def load_param(self, trained_path):
        # zum beispiel, wenn du continue willst. dann willst du ja auch die classifier weights
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            #if 'classifier' in k:
                # print(i[0])
            #    continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])

class TA(nn.Module):
    def __init__(self,base,in_planes):
        super(TA, self).__init__()
        self.base = base
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim,
                                        [16, 16])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=3, padding=1)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats = self.base(track)
        a = F.relu(self.attention_conv(global_feats))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        if self.att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a

class TA_LSTM(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA_LSTM, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.in_planes,
                                        [16, 16])
        self.batchNorm = nn.BatchNorm1d(num_features=self.in_planes, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.gru = torch.nn.GRU(input_size=self.seq_len, hidden_size=5, bidirectional=True)
        self.dense1 = nn.Linear(in_features=self.in_planes*5*2, out_features=self.in_planes, bias=True)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats_spatial = self.base(track) # [320, 512, 16, 16]
        global_feats_one_channel = F.relu(self.attention_conv(global_feats_spatial))  # b*t,middle_dim,1,1 [320, 256, 1, 1]
        global_feats = global_feats_one_channel.view(b, t, self.in_planes)  # b,t,middle_dim [64, 5, 256]
        global_feats = global_feats.permute(0, 2, 1)#[64, 256, 5]

        x = self.batchNorm(global_feats)
        x, _ = self.gru(input=x)
        x = x.view(b, -1)
        att_x = self.dense1(x)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,None

class TA2(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA2, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim,
                                        [16, 16])
        self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=3, padding=1)
        self.batchNorm = nn.BatchNorm1d(num_features=self.in_planes, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.gru = torch.nn.GRU(input_size=self.seq_len, hidden_size=5, bidirectional=True)
        self.dense1 = nn.Linear(in_features=self.middle_dim, out_features=1, bias=True)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats_spatial = self.base(track) # [320, 512, 16, 16]
        x_conv1_one_channel = F.relu(self.attention_conv(global_feats_spatial))  # b*t,middle_dim,1,1 [320, 256, 1, 1]
        x_conv1_one_channel = x_conv1_one_channel.squeeze()
        #global_feats = global_feats_one_channel.view(b, t, self.in_planes)  # b,t,middle_dim [64, 5, 256]
        a = self.dense1(x_conv1_one_channel)
        a = torch.tanh(a)
        a = a.view(b, t)
        if self.att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        global_feat = F.avg_pool2d(global_feats_spatial, global_feats_spatial.size()[2:])
        global_feat = global_feat.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(global_feat, a)
        att_x = torch.sum(att_x, 1)
        global_feat_with_att = att_x.view(b, self.in_planes)

        return global_feat_with_att,a

class TA3(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA3, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.in_planes,
                                        [16, 16])
        self.attention_tconv = nn.Conv1d(self.in_planes, out_channels=1, kernel_size=3, padding=1)
        self.batchNorm = nn.BatchNorm1d(num_features=self.in_planes, eps=1e-05, momentum=0.1, affine=True,
                                        track_running_stats=True)
        self.gru = torch.nn.GRU(input_size=self.seq_len, hidden_size=5, bidirectional=True)
        self.dense1 = nn.Linear(in_features=self.in_planes*self.seq_len, out_features=self.in_planes*self.seq_len, bias=True)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats_spatial = self.base(track) # [320, 512, 16, 16]
        x_conv1_one_channel = F.relu(self.attention_conv(global_feats_spatial))  # b*t,middle_dim,1,1 [320, 256, 1, 1]
        x_conv1_one_channel = x_conv1_one_channel.squeeze()
        #global_feats = global_feats_one_channel.view(b, t, self.in_planes)  # b,t,middle_dim [64, 5, 256]
        a = self.dense1(x_conv1_one_channel.view(b,-1))
        a = torch.tanh(a)
        #a = a.view(b, t,-1)
        a = a.view(-1, t)
        if self.att_gen == 'softmax':
            #a = F.softmax(a, dim=1)
            a = F.softmax(a, dim=1).view(b,t,-1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        global_feat = F.avg_pool2d(global_feats_spatial, global_feats_spatial.size()[2:])
        global_feat = global_feat.view(b, t, -1)
        #a = torch.unsqueeze(a, -1)
        #a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(global_feat, a)
        att_x = torch.sum(att_x, 1)
        global_feat_with_att = att_x.view(b, self.in_planes)

        return global_feat_with_att,a

class TA4(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA4, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim,
                                        [16, 16])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=self.seq_len, padding=(int(self.seq_len/2)))
        #self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=1, padding=0)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats = self.base(track)
        a = F.relu(self.attention_conv(global_feats))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        if self.att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a

class TA5(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA5, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.in_planes,
                                        [16, 16])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.in_planes, out_channels=self.in_planes, kernel_size=self.seq_len, padding=(int(self.seq_len/2)))
        #self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=1, padding=0)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.contiguous().view(b * t, track.size(2), track.size(3), track.size(4)) #.contiguous()
        global_feats = self.base(track)
        a = F.relu(self.attention_conv(global_feats))
        a = a.view(b, t, self.in_planes)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        #a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        a = a.view(-1, t)
        if self.att_gen == 'softmax':
            # a = F.softmax(a, dim=1)
            a = F.softmax(a, dim=1).view(b, t, -1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        #a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a

class TA6(nn.Module):
    #wie TA5 nur mir groups
    def __init__(self,base,in_planes,seq_len):
        super(TA6, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.in_planes,
                                        [16, 16])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.in_planes, out_channels=self.in_planes, kernel_size=self.seq_len,groups=self.in_planes, padding=(int(self.seq_len/2)))
        #self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=1, padding=0)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats = self.base(track)
        a = F.relu(self.attention_conv(global_feats))
        a = a.view(b, t, self.in_planes)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        #a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        a = a.view(-1, t)
        if self.att_gen == 'softmax':
            # a = F.softmax(a, dim=1)
            a = F.softmax(a, dim=1).view(b, t, -1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        #a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a

class TA4_tanh(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        # wie TA4 nur mit tanh statt relu
        super(TA4_tanh, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim,
                                        [16, 16])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=self.seq_len, padding=(int(self.seq_len/2)))
        #self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=1, padding=0)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats = self.base(track)
        a = F.tanh(self.attention_conv(global_feats))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0, 2, 1)
        a = F.tanh(self.attention_tconv(a))
        a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        if self.att_gen == 'softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a

class TA5_tanh(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA5_tanh, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.in_planes,
                                        [16, 16])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.in_planes, out_channels=self.in_planes, kernel_size=self.seq_len, padding=(int(self.seq_len/2)))
        #self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=1, padding=0)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.view(b * t, track.size(2), track.size(3), track.size(4))
        global_feats = self.base(track)
        a = F.tanh(self.attention_conv(global_feats))
        a = a.view(b, t, self.in_planes)
        a = a.permute(0, 2, 1)
        a = F.tanh(self.attention_tconv(a))
        #a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        a = a.view(-1, t)
        if self.att_gen == 'softmax':
            # a = F.softmax(a, dim=1)
            a = F.softmax(a, dim=1).view(b, t, -1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        #a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a

class TA7(nn.Module):
    def __init__(self,base,in_planes,seq_len):
        super(TA7, self).__init__()
        self.base = base
        self.seq_len = seq_len
        self.in_planes = in_planes
        self.att_gen = 'softmax'
        self.middle_dim = 256
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.attention_conv = nn.Conv2d(self.in_planes, self.in_planes,
        #                                [20, 20])  # ^6,16 cooresponds to 256, 256 input image size
        self.attention_tconv = nn.Conv1d(self.in_planes, out_channels=self.in_planes, kernel_size=self.seq_len, padding=(int(self.seq_len/2)))
        #self.attention_tconv = nn.Conv1d(self.middle_dim, out_channels=1, kernel_size=1, padding=0)

        self.fc_global_local_feat = nn.Linear(4 * 1024 + self.in_planes, self.in_planes)

    def forward(self, track):
        b = track.size(0)
        t = track.size(1)
        track = track.contiguous().view(b * t, track.size(2), track.size(3), track.size(4)) #.contiguous()
        global_feats = self.base(track)
        a = F.relu(self.gap(global_feats))
        a = a.view(b, t, self.in_planes)
        a = a.permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a))
        #a = a.view(b, t)

        x = F.avg_pool2d(global_feats, global_feats.size()[2:])

        a = a.view(-1, t)
        if self.att_gen == 'softmax':
            # a = F.softmax(a, dim=1)
            a = F.softmax(a, dim=1).view(b, t, -1)
        elif self.att_gen == 'sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)

        x = x.view(b, t, -1)
        #a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.in_planes)
        att_x = torch.mul(x, a)
        att_x = torch.sum(att_x, 1)
        global_feat = att_x.view(b, self.in_planes)
        return global_feat,a
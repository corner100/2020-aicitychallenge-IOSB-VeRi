# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import os

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

class Baseline_EfficientNet(nn.Module):
    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,in_planes,global_feat_fc):
        super(Baseline_EfficientNet, self).__init__()
        self.in_planes = in_planes
        self.global_feat_fc = global_feat_fc
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
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        else:
            os.environ[
                'TORCH_HOME'] = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/PretrainedModels'
            self.base = EfficientNet.from_pretrained(model_name) # 'efficientnet-b7'
            #self.model = self.model.to('cuda')
            if model_name == 'efficientnet-b0':
                self.in_planes = 1280
            if model_name == 'efficientnet-b3':
                self.in_planes = 1536
            if model_name == 'efficientnet-b7':
                self.in_planes = 2560

        if pretrain_choice == 'self':
            print('Es kann mit efficientnet bisher nur imagnet weights geladen werdden')
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)

        self.fc_global_feat = nn.Linear(in_features=self.in_planes, out_features=self.in_planes) ####
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

    def forward(self, x):

        global_feat = self.gap(self.base.extract_features(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.global_feat_fc:
            global_feat = self.fc_global_feat(global_feat)  ######

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            if self.global_feat_fc:
                feat = F.relu(feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for i in param_dict:
    #         if 'classifier' in i:
    #             continue
    #         self.state_dict()[i].copy_(param_dict[i])
# changed load_params according to https://github.com/michuanhaohao/reid-strong-baseline/issues/85
    def load_param_for_base(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            if 'classifier' in k:
                # print(i[0])
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.state_dict().items():
            # print(i)
            # ich hab die folgenden 3 zeilen auskommentiert
            #if 'classifier' in k:
            #    # print(i[0])
            #    continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])
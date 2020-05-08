# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch
from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth, ViktorTripletLoss1, ViktorTripletLoss2, ViktorTripletLoss3
from .center_loss import CenterLoss
from .some_loss_functions import AngularPenaltySMLoss


def make_loss(cfg, num_classes, dim_feature):    # modified by gu
    if cfg['MODEL.IF_LABELSMOOTH'] == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if cfg['MODEL.METRIC_LOSS_TYPE'] == 'triplet':
        triplet = TripletLoss(cfg['SOLVER.MARGIN'])  # triplet loss
        def loss_func(score, feat, target,engine=None):
            return triplet(feat, target)[0]
    elif cfg['MODEL.METRIC_LOSS_TYPE']  == 'softmax':
        def loss_func(score, feat, target,engine=None):
            return F.cross_entropy(score, target)
    elif cfg['MODEL.METRIC_LOSS_TYPE'] == 'softmax_triplet':
        triplet = TripletLoss(cfg['SOLVER.MARGIN'])  # triplet loss
        def loss_func(score, feat, target,engine=None):
            if cfg['MODEL.IF_LABELSMOOTH'] == 'on':
                return xent(score, target) + triplet(feat, target)[0]
            else:
                return F.cross_entropy(score, target) + triplet(feat, target)[0]
    elif cfg['MODEL.METRIC_LOSS_TYPE'] == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
        def loss_func(pred, feat, target, engine=None):
            return criterion(input=pred, target=target)
    elif cfg['MODEL.METRIC_LOSS_TYPE'] == 'MSE':
        criterion = torch.nn.MSELoss()
        def loss_func(pred, feat, target, engine=None):
            return criterion(input=pred, target=target)
    elif cfg['MODEL.METRIC_LOSS_TYPE'] == 'arcface_softmax_triplet' or cfg['MODEL.METRIC_LOSS_TYPE'] == 'sphereface_softmax_triplet' or cfg['MODEL.METRIC_LOSS_TYPE'] == 'cosface_softmax_triplet':
        if cfg['MODEL.METRIC_LOSS_TYPE'] == 'arcface_softmax_triplet':
            angular_criterion = AngularPenaltySMLoss(in_features=dim_feature, out_features=num_classes, loss_type='arcface')
        elif cfg['MODEL.METRIC_LOSS_TYPE'] == 'sphereface_softmax_triplet':
            angular_criterion = AngularPenaltySMLoss(in_features=dim_feature, out_features=num_classes, loss_type='sphereface')
        elif cfg['MODEL.METRIC_LOSS_TYPE'] == 'cosface_softmax_triplet':
            angular_criterion = AngularPenaltySMLoss(in_features=dim_feature, out_features=num_classes, loss_type='cosface')
        angular_criterion = angular_criterion.to('cuda')
        triplet = TripletLoss(cfg['SOLVER.MARGIN'])  # triplet loss
        def loss_func(score, feat, target, engine=None):
            if cfg['MODEL.IF_LABELSMOOTH'] == 'on':
                return xent(score, target) + triplet(feat, target)[0] + angular_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + triplet(feat, target)[0] + angular_criterion(feat, target)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg['MODEL.METRIC_LOSS_TYPE']))
    return loss_func

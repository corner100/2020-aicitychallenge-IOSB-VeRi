# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from .baseline import Baseline
from .baseline_track import Baseline_track
from .baseline_efficientnet import Baseline_EfficientNet
from .baseline_regression import Baseline_Regression

def build_model(cfg, num_classes):
    if cfg['DATASETS.TRACKS']:
            model = Baseline_track(num_classes=num_classes, last_stride=cfg['MODEL.LAST_STRIDE'],
                                   model_path=cfg['MODEL.PRETRAIN_PATH'], neck=cfg['MODEL.NECK'],
                                   neck_feat=cfg['TEST.NECK_FEAT'], model_name=cfg['MODEL.NAME'],
                                   pretrain_choice=cfg['MODEL.PRETRAIN_CHOICE'], seq_len=cfg['DATASETS.TRACKS_LENGTH'],
                                   head=cfg['MODEL.TRACK_HEAD'], in_planes=cfg['MODEL.FEAT_SIZE'],
                                   global_feat_fc=cfg['MODEL.FEAT_FC'],
                                   train_last_base_params=cfg['MODEL.N_BASE_PARAM_TRAIN'])
    else:
            model = Baseline(num_classes=num_classes, last_stride=cfg['MODEL.LAST_STRIDE'],
                             model_path=cfg['MODEL.PRETRAIN_PATH'], neck=cfg['MODEL.NECK'],
                             neck_feat=cfg['TEST.NECK_FEAT'], model_name=cfg['MODEL.NAME'],
                             pretrain_choice=cfg['MODEL.PRETRAIN_CHOICE'], in_planes=cfg['MODEL.FEAT_SIZE'],
                             global_feat_fc=cfg['MODEL.FEAT_FC'])
    return model


def build_regression_model(cfg):
    model = Baseline_Regression(last_stride=cfg['MODEL.LAST_STRIDE'], model_path=cfg['MODEL.PRETRAIN_PATH'],
                                neck=cfg['MODEL.NECK'], neck_feat=cfg['TEST.NECK_FEAT'], model_name=cfg['MODEL.NAME'],
                                pretrain_choice=cfg['MODEL.PRETRAIN_CHOICE'], in_planes=cfg['MODEL.FEAT_SIZE'],
                                global_feat_fc=cfg['MODEL.FEAT_FC'],
                                train_last_base_params=cfg['MODEL.N_BASE_PARAM_TRAIN'])
    return model
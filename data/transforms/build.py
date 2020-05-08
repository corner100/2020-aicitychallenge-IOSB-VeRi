# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg['INPUT.PIXEL_MEAN'], std=cfg['INPUT.PIXEL_STD'])
    if is_train:
        tranform_keypoints_list = []
        tranform_list = []
        tranform_list.append(T.Resize(cfg['INPUT.SIZE_TRAIN']))

        if cfg['TRANSFORM.RANDOM_HORIZONTAL_FLIP']:
            tranform_list.append(T.RandomHorizontalFlip(p=cfg['INPUT.PROB']))
            tranform_keypoints_list.append(T.RandomHorizontalFlip(p=cfg['INPUT.PROB']))

        if cfg['TRANSFORM.PAD']:
            tranform_list.append(T.Pad(cfg['TRANSFORM.PADDING_SIZE']))
            tranform_keypoints_list.append(T.Pad(cfg['TRANSFORM.PADDING_SIZE']))

        if cfg['TRANSFORM.RANDOM_CROP']:
            tranform_list.append(T.RandomCrop(cfg['INPUT.SIZE_TRAIN']))
            tranform_keypoints_list.append(T.RandomCrop(cfg['INPUT.SIZE_TRAIN']))

        tranform_list.append(T.ToTensor())
        tranform_keypoints_list.append(T.ToTensor())

        tranform_list.append(normalize_transform)

        if cfg['TRANSFORM.RANDOM_ERASING']:
            tranform_list.append(RandomErasing(probability=cfg['INPUT.RE_PROB'], mean=cfg['INPUT.PIXEL_MEAN']))
            tranform_keypoints_list.append(RandomErasing(probability=cfg['INPUT.RE_PROB'], mean=cfg['INPUT.PIXEL_MEAN']))

        transform = T.Compose(tranform_list)
        tranform_keypoints = T.Compose(tranform_keypoints_list)
    else:
        tranform_keypoints_list = []
        tranform_list = []
        tranform_list.append(T.Resize(cfg['INPUT.SIZE_TEST']))
        tranform_list.append(T.ToTensor())
        tranform_keypoints_list.append(T.ToTensor())
        tranform_list.append(normalize_transform)
        transform = T.Compose(tranform_list)
        tranform_keypoints = T.Compose(tranform_keypoints_list)

    return transform, tranform_keypoints

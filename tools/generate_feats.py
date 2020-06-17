import argparse
import os
from datetime import date, datetime
import sys
import torch
import glob
import yaml
print('curr device: ', torch.cuda.current_device())
from torch.backends import cudnn
sys.path.append('.')
sys.path.append('..')
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger
from utils.random_seed import seed_everything
from PIL import Image
import albumentations
import cv2
import six
from utils import wandb_config
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np
seed_everything()

class ToTensorV2(albumentations.core.transforms_interface.BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}

class Augmenation():
    def __init__(self, cfg):
        self.resize_transformation = albumentations.Resize(height=cfg['INPUT.SIZE_TEST'][1], width=cfg['INPUT.SIZE_TEST'][0])
        self.normalize_transform = albumentations.Normalize(mean=cfg['INPUT.PIXEL_MEAN'], std=cfg['INPUT.PIXEL_STD'])
        self.tensor_transformation = ToTensorV2()
        self.aug = albumentations.Compose([self.resize_transformation, self.normalize_transform, self.tensor_transformation])
    def read_and_augment_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.aug(image=img)
        img = augmented['image']
        return img

def main():
    parser = argparse.ArgumentParser(description='cityAI Vehicle ReID')
    parser.add_argument('--input', default="../input", help='folder to the input image')
    parser.add_argument('--output', default="../output", help='folder where the output featurevectors should be saved')
    parser.add_argument('--config', default="../config.yaml", help='path to config file')
    parser.add_argument('--weights', default="../weights.pth", help='path to model weights')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    cfg = wandb_config.Config(args.config)
    device = cfg['MODEL.DEVICE']
    model = build_model(cfg, num_classes=cfg['DATASETS.NUM_CLASSES'])

    model.load_param(args.weights)
    model.to(device)

    aug = Augmenation(cfg=cfg)

    img_paths = glob.glob(os.path.join(args.input,'*'))
    model.eval()
    with torch.no_grad():
        for img_path in img_paths:
            img = aug.read_and_augment_image(img_path)
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            feat = model(img.unsqueeze(dim=0))
            np.save(os.path.join(args.output, os.path.basename(img_path).split('.')[0] + '.npy'), np.array(feat[0].cpu()))




if __name__ == '__main__':
    main()

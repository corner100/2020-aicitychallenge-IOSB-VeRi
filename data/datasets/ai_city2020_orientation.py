import glob
import re

import os
import random
from utils.random_seed import seed_everything
seed_everything()
import copy
import numpy as np
from sklearn.model_selection import KFold
from .bases import BaseImageDataset
# import xml.etree.ElementTree as ET
# xmlp = ET.XMLParser(encoding="utf-8")
# from bs4 import BeautifulSoup
import wandb
import matplotlib.pyplot as plt
import logging

class AI_CITY2020_ORIENTATION(BaseImageDataset):
    """
       AI_CITY2020 by viktor
       """

    def __init__(self, cfg, **kwargs):
        super(AI_CITY2020_ORIENTATION, self).__init__()
        root = cfg['DATASETS.ROOT_DIR']
        self.cfg = cfg
        self.synthetic_dir = cfg['DATASETS.SYNTHETIC_DIR']
        self.synthetic_dataset_path = os.path.join(root, self.synthetic_dir)
        self.label_dir = cfg['DATASETS.LABEL_DIR']
        self.label_dir_path_train = os.path.join(root, self.label_dir,'image_train')
        self.label_dir_path_query = os.path.join(root, self.label_dir,'image_query')
        self.real_dir = cfg['DATASETS.DATASET_DIR']
        self.real_dir_path_train = os.path.join(root, self.real_dir,'image_train')
        self.real_dir_path_query = os.path.join(root, self.real_dir,'image_query')

        trainval_synth = self.read_synthetic_dataset_orient_data()
        train_real = self.read_real_dataset_orient_data(self.label_dir_path_train,self.real_dir_path_train)
        val_real = self.read_real_dataset_orient_data(self.label_dir_path_query,self.real_dir_path_query)

        random.shuffle(trainval_synth)
        ratio = 0.2
        val = trainval_synth[:int(len(trainval_synth)*ratio)]
        train = trainval_synth[int(len(trainval_synth)*ratio):]
        self.train = train + train_real
        #self.train = train_real
        self.val = val_real
        if wandb.run is not None:
            wandb.config.update({'numTrain': len(self.train)})
            wandb.config.update({'numVal': len(self.val)})


    def print_dataset_statistics_orientation(self, train, val):
        num_train, x_train, y_train,z_train = self.get_imagedata_info(train)
        num_val, x_val, y_val,z_val = self.get_imagedata_info(val)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train, x_train, y_train,z_train))
        print("  val    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_val, x_val, y_val,z_val))
        print("  ----------------------------------------")

        logger = logging.getLogger("reid_baseline.dataset")
        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train, x_train, y_train,z_train))
        logger.info("  val    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_val, x_val, y_val,z_val))
        logger.info("  ----------------------------------------")

    def read_synthetic_dataset_orient_data(self):
        img_paths = glob.glob(os.path.join(self.synthetic_dataset_path,'*.jpg'))
        dataset = []
        for img_path in img_paths:
            result = re.search('image_(.*)_bg_', img_path)
            result = re.search('_alpha_(.*)_beta_', img_path)
            alpha = result.group(1)
            x = np.sin(np.pi*float(alpha)/180)
            y = np.cos(np.pi*float(alpha)/180)
            result = re.search('_beta_(.*).jpg', img_path)
            beta = result.group(1)
            z = np.sin(np.pi*float(beta)/180)

            dataset.append([img_path, x, y, z, alpha, beta])
        return dataset

    def read_real_dataset_orient_data(self, label_dir_path,real_dir_path):
        label_paths = glob.glob(os.path.join(label_dir_path,'*.npy'))
        label_names = [os.path.basename(path).split('.')[0] for path in label_paths]
        img_paths = np.array(glob.glob(os.path.join(real_dir_path,'*.jpg')))
        remain = np.array(
            [i for i, path in enumerate(img_paths) if os.path.basename(path).split('.')[0] in label_names])
        img_paths_remain = img_paths[remain]
        img_paths_remain.sort()
        label_paths.sort()
        dataset = []
        alphas = []
        betas = []
        for img_path,label_path in zip(img_paths_remain,label_paths):
            gt = np.load(label_path)
            alpha = gt[0]
            x = np.sin(np.pi*float(alpha)/180)
            y = np.cos(np.pi*float(alpha)/180)
            beta = gt[1]
            z = np.sin(np.pi*float(beta)/180)
            dataset.append([img_path, x, y, z, alpha, beta])
            alphas.append(alpha)
            betas.append(beta)
        return dataset



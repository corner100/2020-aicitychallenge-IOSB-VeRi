# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import datetime
import sys
import torch
print('curr device: ', torch.cuda.current_device())
from torch.backends import cudnn
import wandb
sys.path.append('.')
sys.path.append('..')
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR
from utils.logger import setup_logger
from utils.random_seed import seed_everything
seed_everything()

def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)
    print('Train without center loss, the loss type is', cfg['MODEL.METRIC_LOSS_TYPE'])
    optimizer = make_optimizer(cfg, model)
    loss_func = make_loss(cfg, num_classes, model.in_planes)    # modified by gu

    # Add for using self trained model
    if cfg['MODEL.PRETRAIN_CHOICE'] == 'continue':
        #start_epoch = eval(cfg['MODEL.PRETRAIN_PATH'].split('/')[-1].split('.')[0].split('_')[-1])
        start_epoch = 0
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg['MODEL.PRETRAIN_PATH'].replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        model.load_param(cfg['MODEL.PRETRAIN_PATH'])
        #optimizer = make_optimizer(cfg, model)
        scheduler = WarmupMultiStepLR(optimizer, cfg['SOLVER.STEPS'], cfg['SOLVER.GAMMA'], cfg['SOLVER.WARMUP_FACTOR'],
                                      cfg['SOLVER.WARMUP_ITERS'], cfg['SOLVER.WARMUP_METHOD'])
    elif cfg['MODEL.PRETRAIN_CHOICE'] == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg['SOLVER.STEPS'], cfg['SOLVER.GAMMA'], cfg['SOLVER.WARMUP_FACTOR'],
                                      cfg['SOLVER.WARMUP_ITERS'], cfg['SOLVER.WARMUP_METHOD'])
    elif cfg['MODEL.PRETRAIN_CHOICE'] == 'self' or cfg['MODEL.PRETRAIN_CHOICE'] == 'self-no-head':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg['SOLVER.STEPS'], cfg['SOLVER.GAMMA'],
                                      cfg['SOLVER.WARMUP_FACTOR'],
                                      cfg['SOLVER.WARMUP_ITERS'], cfg['SOLVER.WARMUP_METHOD'])
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg['MODEL.PRETRAIN_CHOICE']))

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,      # modify for using self trained model
        loss_func,
        num_query,
        start_epoch,     # add for using self trained model
        dataset
    )

def main():
    parser = argparse.ArgumentParser(description='cityAI Vehicle ReID')
    parser.add_argument('--project', default="test", help='which wandb project!')
    args = parser.parse_args()
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(entity="user",project=args.project, name=run_name + "_justfold1")

    print('curr device: ', torch.cuda.current_device())
    cfg = wandb.config
    cfg['OUTPUT_DIR'] = wandb.run.dir + '_my_logs_justfold1'
    cfg['WANDB_DIR'] = wandb.run.dir
    if not os.path.exists(cfg['OUTPUT_DIR']):
        os.makedirs(cfg['OUTPUT_DIR'])

    logger = setup_logger("reid_baseline", cfg['OUTPUT_DIR'], 0)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg['MODEL.DEVICE'] == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['MODEL.DEVICE_ID']  # new add by gu
    print('curr device: ', torch.cuda.current_device())
    cudnn.benchmark = True
    train(cfg)

if __name__ == '__main__':
    main()

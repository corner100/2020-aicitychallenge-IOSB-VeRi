import argparse
import os
import datetime
import sys
import torch
import numpy as np
print('curr device: ', torch.cuda.current_device())
import copy
from torch.backends import cudnn
import wandb
import pandas as pd

sys.path.append('.')
sys.path.append('..')
from config import cfg
from data import make_data_loader
from modeling import build_model
from engine.inference import predict

from utils.logger import setup_logger
from utils.random_seed import seed_everything
seed_everything()

def main():
    parser = argparse.ArgumentParser(description='cityAI Vehicle ReID')
    parser.add_argument('-u','--user', help='username', default='corner')
    parser.add_argument('-p','--project', help='project name', default='cityai2019')
    parser.add_argument('-r','--run_id', nargs='+', help='list of run ides, use -r xxx xyxy ...', default='6qpihpn8')
    args = parser.parse_args()

    api = wandb.Api()
    runs = []
    for run_id in args.run_id:
        runs.append(api.run(args.user+'/'+args.project+'/'+run_id))

    print('Das Skript nimmt die besten Models jedes runs und berechnet die mAP, Rank-1 usw..')

    cmcs=[]
    mAPs=[]
    for run in runs:
        if run.state != "finished":
            print("training didn't finish yet")

        cfg = copy.deepcopy(run.config)
        mAP_best = run.summary['mAP_best']
        epoch_best = run.summary['epoch_best']
        fold = 1#cfg['fold']
        train_loader, val_loader, num_query, num_classes = make_data_loader(cfg, fold)
        model = build_model(cfg, num_classes)
        weights_path = os.path.join(cfg['OUTPUT_DIR'],cfg['MODEL.NAME']+'_model_'+str(epoch_best)+'.pth')
        model.load_param(weights_path)
        cmc, mAP = predict(cfg, model, val_loader, num_query)
        cmcs.append(cmc)
        mAPs.append(mAP)

    for run_id, cmc, mAP in zip(args.run_id, cmcs, mAPs):
        print('=======')
        print(run_id)
        print("mAP: {:.2%}".format(mAP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    print('')
    print('mAP, Average: {:.2%}'.format(np.mean(mAPs)))
    for r in [1, 5, 10]:
        print("Rank-{:<3}:{:.2%}".format(r, np.mean(np.array(cmcs)[:,r - 1])))




if __name__ == '__main__':
    main()

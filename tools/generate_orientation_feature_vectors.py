import argparse
import os
import sys
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.backends import cudnn
import numpy as np
from torch.utils.data import DataLoader
import wandb
import copy

sys.path.append('.')
sys.path.append('..')
from data.collate_batch import train_collate_fn, val_collate_fn, val_collate_fn_regression
from data.datasets import init_dataset, ImageDataset, ImageDatasetOrientation
from data.samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from data.transforms import build_transforms
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_regression_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
import albumentations
from ignite.engine import Engine, Events
from utils.logger import setup_logger
from ignite.metrics import Metric
from utils.config_utils import config_to_dict
from engine.inference import generate_image_dir_and_txt

class Score_feats(Metric):
    def __init__(self):
        super(Score_feats, self).__init__()

    def reset(self):
        self.scores = []
        self.labels = []
        self.pids = []
        self.camids = []

    def update(self, output):
        score, label, pid, camid = output
        self.scores.extend(np.asarray(score))
        self.labels.extend(np.asarray(label))
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        return self.scores, self.labels, self.pids, self.camids

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pid, camid = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            score, feat = model(data)
            return score.cpu(), feat.cpu(), pid, camid

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def main():
    parser = argparse.ArgumentParser(description='Vehicle orientation')
    parser.add_argument('-u','--user', help='username', default='corner')
    parser.add_argument('-p','--project', help='project name', default='cityai2020Orientation')
    parser.add_argument('-r','--run_id', help='run id', default='pe5y029c')
    traindata = False
    is_synthetic = False
    is_track = True

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('wandb api')
    api = wandb.Api()
    run = api.run(args.user + '/' + args.project + '/' + args.run_id)
    print('copy wandb configs')
    cfg = copy.deepcopy(run.config)

    if cfg['MODEL.DEVICE'] == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['MODEL.DEVICE_ID']
    cfg['DATASETS.TRACKS_FILE'] = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/ImageFolderStatistics/Track3Files'
    cudnn.benchmark = True
    print('load dataset')
    if is_synthetic and traindata:
        cfg['DATASETS.SYNTHETIC'] = True
        cfg['DATASETS.SYNTHETIC_LOADER'] = 0
        cfg['DATASETS.SYNTHETIC_DIR'] = 'ai_city_challenge/2020/Track2/AIC20_track2_reid_simulation/AIC20_track2/AIC20_ReID_Simulation'
        dataset = init_dataset('AI_CITY2020_TEST_VAL', cfg=cfg,fold=1,eval_mode=False)
    else:
        if is_track:
            dataset = init_dataset('AI_CITY2020_TRACKS', cfg=cfg,fold=1,eval_mode=False)
        else:
            dataset = init_dataset('AI_CITY2020_TEST_VAL', cfg=cfg,fold=1,eval_mode=False)

    if traindata:
        if is_synthetic and traindata:
            dataset = [item[0] for item in dataset.train]
            dataset = dataset[36935:]
            dataset.sort()
            dataset = [[item, 0,0] for item in dataset]
            val_set = ImageDatasetOrientation(dataset, cfg, is_train=False, test=True)
        else:
            val_set = ImageDatasetOrientation(dataset.train, cfg, is_train=False, test=True)
    else:
        val_set = ImageDatasetOrientation(dataset.query+dataset.gallery, cfg, is_train=False, test=True)
    #
    val_loader = DataLoader(
        val_set, batch_size=cfg['TEST.IMS_PER_BATCH'], shuffle=False, num_workers = cfg['DATALOADER.NUM_WORKERS'],
        collate_fn=val_collate_fn
    )
    print('build model')
    model = build_regression_model(cfg)
    print('get last epoch')
    epoch_best = 10#run.summary['epoch']
    weights_path = os.path.join(cfg['OUTPUT_DIR'],cfg['MODEL.NAME']+'_model_'+str(epoch_best)+'.pth')
    print('load pretrained weights')
    model.load_param(weights_path)
    model.eval()

    evaluator = create_supervised_evaluator(model, metrics={'score_feat': Score_feats()}, device = cfg['MODEL.DEVICE'])
    print('run')
    evaluator.run(val_loader)
    scores,feats,pids,camids = evaluator.state.metrics['score_feat']
    feats = np.array(feats)
    scores = np.array(scores)
    print('save')
    if traindata:

        if is_track:
            feats_mean = []
            for item in dataset.train_tracks_vID:
                indis = np.array([int(jitem[:6]) - 1 for jitem in item[0]])
                feats_mean.append(np.mean(feats[indis], axis=0))
            feats_mean = np.array(feats_mean)

            scores_mean = []
            for item in dataset.train_tracks_vID:
                indis = np.array([int(jitem[:6]) - 1 for jitem in item[0]])
                scores_mean.append(np.mean(scores[indis], axis=0))

            np.save(os.path.join(cfg['OUTPUT_DIR'], 'feats_train_track.npy'), np.array(feats_mean))  # .npy extension is added if not given
            np.save(os.path.join(cfg['OUTPUT_DIR'], 'scores_train_track.npy'), np.array(scores_mean))  # .npy extension is added if not given
        else:
            if is_synthetic and traindata:
                np.save(os.path.join(cfg['OUTPUT_DIR'], 'feats_train_synthetic.npy'), np.array(feats))  # .npy extension is added if not given
                np.save(os.path.join(cfg['OUTPUT_DIR'], 'scores_train_synthetic.npy'), np.array(scores))  # .npy extension is added if not given
            else:
                np.save(os.path.join(cfg['OUTPUT_DIR'], 'feats_train.npy'), np.array(feats))  # .npy extension is added if not given
                np.save(os.path.join(cfg['OUTPUT_DIR'], 'scores_train.npy'), np.array(scores))  # .npy extension is added if not given
    else:
        if is_track:
            feats_mean = []
            for feat in feats[:1052]:
                feats_mean.append(feat)
            for item in dataset.test_tracks_vID:
                indis = np.array([int(jitem[:6]) - 1 for jitem in item[0]])
                feats_mean.append(np.mean(feats[1052:][indis], axis=0))
            feats_mean = np.array(feats_mean)

            scores_mean = []
            for score in scores[:1052]:
                scores_mean.append(score)
            for item in dataset.test_tracks_vID:
                indis = np.array([int(jitem[:6]) - 1 for jitem in item[0]])
                scores_mean.append(np.mean(scores[1052:][indis], axis=0))
            np.save(os.path.join(cfg['OUTPUT_DIR'], 'feats_query_gal_track.npy'), np.array(feats_mean))  # .npy extension is added if not given
            np.save(os.path.join(cfg['OUTPUT_DIR'], 'scores_query_gal_track.npy'), np.array(scores_mean))  # .npy extension is added if not given
        else:
            np.save(os.path.join(cfg['OUTPUT_DIR'], 'feats_query_gal.npy'), np.array(feats))  # .npy extension is added if not given
            np.save(os.path.join(cfg['OUTPUT_DIR'], 'scores_query_gal.npy'), np.array(scores))  # .npy extension is added if not given
        print(cfg['OUTPUT_DIR'])
        print()
        txt_dir='dist_orient'
        num_query = 1052
        all_mAP = np.zeros(num_query)


        statistic_name ='feats'
        feats = torch.from_numpy(feats).float().to('cuda')
        feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats_normed[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        g_camids = np.ones_like(g_camids)
        g_pids = np.ones_like(g_pids)
        generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, g_pids, q_pids, g_camids, q_camids, all_mAP,
                                   statistic_name=statistic_name, max_rank=100)

        statistic_name ='xyz'
        feats = torch.from_numpy(scores).float().to('cuda')
        feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats_normed[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        g_camids = np.ones_like(g_camids)
        g_pids = np.ones_like(g_pids)
        generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, g_pids, q_pids, g_camids, q_camids, all_mAP,
                                   statistic_name=statistic_name, max_rank=100)

        statistic_name ='xy'
        scores_curr = scores[:,0:2]
        feats = torch.from_numpy(scores_curr).float().to('cuda')
        feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats_normed[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        g_camids = np.ones_like(g_camids)
        g_pids = np.ones_like(g_pids)
        generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, g_pids, q_pids, g_camids, q_camids, all_mAP,
                                   statistic_name=statistic_name, max_rank=100)


        statistic_name ='x'
        scores_curr = scores[:, 0:1]
        feats = torch.from_numpy(scores_curr).float().to('cuda')
        feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats_normed[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        g_camids = np.ones_like(g_camids)
        g_pids = np.ones_like(g_pids)
        generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, g_pids, q_pids, g_camids, q_camids, all_mAP,
                                   statistic_name=statistic_name, max_rank=100)
        statistic_name ='y'
        scores_curr = scores[:, 1:2]
        feats = torch.from_numpy(scores_curr).float().to('cuda')
        feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats_normed[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        g_camids = np.ones_like(g_camids)
        g_pids = np.ones_like(g_pids)
        generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, g_pids, q_pids, g_camids, q_camids, all_mAP,
                                   statistic_name=statistic_name, max_rank=100)
        statistic_name ='z'
        scores_curr = scores[:, 2:3]
        feats = torch.from_numpy(scores_curr).float().to('cuda')
        feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats_normed[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        g_camids = np.ones_like(g_camids)
        g_pids = np.ones_like(g_pids)
        generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, g_pids, q_pids, g_camids, q_camids, all_mAP,
                                   statistic_name=statistic_name, max_rank=100)


if __name__ == '__main__':
    main()

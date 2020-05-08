# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from data.datasets import init_dataset

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=100, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP, all_mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        distmat_rerank = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc_rerank, mAP_rerank, all_mAP_rerank = eval_func(distmat_rerank, q_pids, g_pids, q_camids, g_camids,
                                                           max_rank=self.max_rank)
        return cmc, mAP, mAP_rerank


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=100, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP, all_mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class GiveMeTheFeaturesMetric(Metric): # ich habe die Klasse erstellt, damit beim submitten ich einfach die gesamten Features des testsets so erhalte
    def __init__(self,dataset, num_query, max_rank=100, feat_norm='yes'):
        super(GiveMeTheFeaturesMetric, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.dataset = dataset

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        return feats, self.pids, self.camids, distmat, self.num_query

class R1_mAP_train(Metric):
    def __init__(self, max_rank=100, feat_norm='yes'):
        super(R1_mAP_train, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats
        q_pids = np.asarray(self.pids)
        q_camids = np.asarray(self.camids)
        # gallery
        gf = feats
        g_pids = np.asarray(self.pids)
        g_camids = np.asarray(self.camids)
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP, all_mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class GiveMeTheFeaturesMetric_train(Metric):
    def __init__(self, max_rank=100, feat_norm='yes'):
        super(GiveMeTheFeaturesMetric_train, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # gallery
        f = feats

        m, n = f.shape[0], f.shape[0]
        distmat = torch.pow(f, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(f, 2).sum(dim=1, keepdim=True).expand(n, m).t() # jedes q hoch 2 plus jedes g hoch 2
        distmat.addmm_(1, -2, f, f.t()) #
        distmat = distmat.cpu().numpy()
        return self.feats, self.pids, self.camids, distmat

class R1_mAP_Tracks(Metric):
    def __init__(self,dataset, num_query, max_rank=100, feat_norm='yes'):
        super(R1_mAP_Tracks, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.dataset = dataset

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        test_name_track_indice = list(self.dataset.test_track_indice_from_test_name.items())
        test_name_track_indice.sort()
        track_indice = np.array([item[1] for item in test_name_track_indice])
        gf_tracks = feats[self.num_query:]
        gf = gf_tracks[track_indice]
        g_pids_tracks = np.asarray(self.pids[self.num_query:])
        g_pids = g_pids_tracks[track_indice]
        g_camids_tracks = np.asarray(self.camids[self.num_query:])
        g_camids = g_camids_tracks[track_indice]
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP, all_mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        distmat_rerank_track = re_ranking(qf, gf_tracks, k1=6, k2=3, lambda_value=0.3)
        distmat_rerank = distmat_rerank_track[:, track_indice]
        cmc_rerank, mAP_rerank, all_mAP_rerank = eval_func(distmat_rerank, q_pids, g_pids, q_camids, g_camids, max_rank=self.max_rank)
        return cmc, mAP, mAP_rerank


class GiveMeTheFeaturesMetricForTracks(Metric): # ich habe die Klasse erstellt, damit beim submitten ich einfach die gesamten Features des testsets so erhalte
    def __init__(self, num_query, max_rank=100, feat_norm='yes'):
        super(GiveMeTheFeaturesMetricForTracks, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        #self.dataset = dataset
        #self.dataset_tracks = init_dataset('AI_CITY2020_TRACKS', cfg=cfg, fold=1, eval_mode=True)

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t() # jedes q hoch 2 plus jedes g hoch 2
        distmat.addmm_(1, -2, qf, gf.t()) #
        distmat = distmat.cpu().numpy()
        return feats, self.pids, self.camids, distmat, self.num_query
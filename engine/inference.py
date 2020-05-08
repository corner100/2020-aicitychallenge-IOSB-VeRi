# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine
import os
from utils.reid_metric import GiveMeTheFeaturesMetric, GiveMeTheFeaturesMetricForTracks
from utils.re_ranking import re_ranking

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
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def submit_val(
        cfg,
        model,
        val_loader,
        num_query,
        directory,
        txt_dir,
        dataset=None
):
    device = cfg['MODEL.DEVICE']
    print("Create evaluator")
    max_rank = 100

    if cfg['DATASETS.TRACKS']:
            evaluator = create_supervised_evaluator(model, metrics={
                'submit_metric': GiveMeTheFeaturesMetricForTracks(num_query, max_rank=max_rank,
                                                                  feat_norm=cfg['TEST.FEAT_NORM']),
            },
                                                    device=device)

    else:
            evaluator = create_supervised_evaluator(model, metrics={
                'submit_metric': GiveMeTheFeaturesMetric(dataset, num_query, max_rank=max_rank,
                                                         feat_norm=cfg['TEST.FEAT_NORM']),
            },
                                                    device=device)

    print('run')
    evaluator.run(val_loader)
    print('end run')

    feats, _, _, distmat, num_query = evaluator.state.metrics['submit_metric']
    generate_submit_and_visualize_files(cfg, txt_dir, feats, num_query, dataset,directory, feat_norm=cfg['TEST.FEAT_NORM'])

def generate_submit_and_visualize_files(cfg, txt_dir, feats, num_query, dataset,directory, feat_norm, max_rank=100):
    dist_directory = os.path.join(directory, 'generate_feats_for_visual')
    if not os.path.exists(dist_directory):
        os.makedirs(dist_directory)
    if cfg['DATASETS.TRACKS']:
        # feats = torch.cat(feats, dim=0)
        if feat_norm == 'yes':
            print("The test feature is normalized")
            feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        # gallery
        ## für tracks
        test_name_track_indice = list(dataset.test_track_indice_from_test_name.items())
        test_name_track_indice.sort()
        track_indice = np.array([item[1] for item in test_name_track_indice])
        gf_tracks = feats_normed[num_query:]
        gf = gf_tracks[track_indice]
        ##
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        print(123)
    else:
        # feats = torch.cat(feats, dim=0)
        if feat_norm == 'yes':
            print("The test feature is normalized")
            feats_normed = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats_normed[:num_query]
        # gallery
        gf = feats_normed[num_query:]
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

    if cfg['DATASETS.TRACKS']:
        statistic_name = 'direct_track'
    else:
        statistic_name = 'direct'

    generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, statistic_name=statistic_name, dist_directory=dist_directory, max_rank=100)
    np.save(os.path.join(dist_directory, 'feat' + statistic_name + '.npy'), np.array(feats.cpu()))

    if cfg['DATASETS.TRACKS']:
        distmat_rerank_track = re_ranking(qf, gf_tracks, k1=6, k2=3, lambda_value=0.3)
        statistic_name = 'rerank_track'
        distmat_rerank = distmat_rerank_track[:, track_indice]
    else:
        distmat_rerank = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        statistic_name = 'rerank'
    generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat_rerank, statistic_name=statistic_name, dist_directory=dist_directory, max_rank=100)



def generate_image_dir_and_txt(cfg, dataset, txt_dir, distmat, statistic_name, dist_directory, max_rank=100):

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    queries = dataset.query
    galleries = dataset.gallery
    galleries = [[os.path.basename(os.path.splitext(item[0])[0]), item[1], item[2]] for item in galleries]

    queryset = [[os.path.basename(os.path.splitext(item[0])[0]), item[1], item[2]] for item in queries]
    result = {}
    for i_query, query in enumerate(queryset):
        list_of_reids = []
        for i_indice, indice in enumerate(indices[i_query,
                                          :max_rank + 1]):
            list_of_reids.append(galleries[indice][0] + ' ' + str(i_indice) + ' ' + str(
                distmat[i_query, indice]))
        result[query[0]] = list_of_reids

    dist_dir = os.path.join(dist_directory, txt_dir + '_' + statistic_name)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    query_names = list(result.keys())
    query_names.sort()
    with open(os.path.join(dist_directory, txt_dir + '_' + statistic_name + '.txt'), 'w') as f_submit:
        for i_query_name, query_name in enumerate(query_names):
            with open(os.path.join(dist_dir, query_name + '.txt'), 'w') as f:
                for item in result[query_name]:
                    f.write("%s\n" % item)
                for item in result[query_name][:100]:
                    f_submit.write(item[:7])
            f_submit.write('\n')

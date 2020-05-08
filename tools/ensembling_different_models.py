import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import copy
from utils.re_ranking import re_ranking
import os
import wandb
from data import make_data_loader

def generate_submit_txt_neu(dist_dir, txt_dir, distmat, statistic_name):
    dist_directory = dist_dir
    indices = np.argsort(distmat, axis=1)

    dist_dir = os.path.join(dist_directory, txt_dir + '_' + statistic_name)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    with open(os.path.join(dist_dir, txt_dir + '_' + statistic_name + '.txt'), 'w') as f_submit:
        for i_query_name, curr_query in enumerate(indices):
            for item in curr_query[:100]:
                f_submit.write(str(item + 1).zfill(6) + " ")
            f_submit.write('\n')

def main():
    wandb.init()
    cfg = wandb.config
    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)

    num_query = 1052
    feat_track_path1 = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/logs/cityai/2020/wandb/run-20200330_132414-zfwusjgf_my_logs_justfold1/featdirect_track.npy'
    feat_track_path3 = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/logs/cityai/2020/wandb/run-20200408_154344-xao5kw7i_my_logs_justfold1/featdirect_track.npy'
    feat_path_orient = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/logs/cityai/2020/Orientation/wandb/run-20200406_052703-pe5y029c_my_logs_Regression/feats.npy'
    feat_path_orient = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/logs/cityai/2020/Orientation/wandb/run-20200406_052703-pe5y029c_my_logs_Regression/scores.npy'
    track_indice_path = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/logs/cityai/2020/wandb/run-20200330_132414-zfwusjgf_my_logs_justfold1/track_indice.npy'

    feats_track1 = np.load(feat_track_path1)
    feats_track3 = np.load(feat_track_path3)
    feats_track1 = torch.tensor(feats_track1)
    feats_track3 = torch.tensor(feats_track3)
    track_indice = np.load(track_indice_path)
    orient_feats = np.load(feat_path_orient)
    orient_feats = torch.tensor(orient_feats)

    feats_normed_track = torch.nn.functional.normalize(feats_track1, dim=1, p=2)
    qf = feats_normed_track[:num_query]
    gf_tracks = feats_normed_track[num_query:]
    gf = gf_tracks[track_indice]
    ##
    m, n = qf.shape[0], gf.shape[0]
    distmat_track = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat_track.addmm_(1, -2, qf, gf.t())
    distmat_track = distmat_track.cpu().numpy()
    distmat_rerank_track = re_ranking(qf, gf_tracks, k1=6, k2=3, lambda_value=0.3)
    distmat_rerank_track = distmat_rerank_track[:, track_indice]
    distmat_rerank = distmat_rerank_track

    ###########

    feats_normed_track = torch.nn.functional.normalize(feats_track3, dim=1, p=2)
    qf = feats_normed_track[:num_query]
    gf_tracks = feats_normed_track[num_query:]
    gf = gf_tracks[track_indice]
    ##
    m, n = qf.shape[0], gf.shape[0]
    distmat_track = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat_track.addmm_(1, -2, qf, gf.t())
    distmat_track3 = distmat_track.cpu().numpy()
    distmat_rerank_track3 = re_ranking(qf, gf_tracks, k1=6, k2=3, lambda_value=0.3)
    distmat_rerank_track3 = distmat_rerank_track3[:, track_indice]
    distmat_rerank3 = distmat_rerank_track3
    ###########

    # orient
    orient_feats_normed = torch.nn.functional.normalize(orient_feats, dim=1, p=2)
    orient_qf = orient_feats_normed[:num_query]
    orient_gf = orient_feats_normed[num_query:]
    distmat_orients = torch.pow(orient_qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(orient_gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat_orients.addmm_(1, -2, orient_qf, orient_gf.t())
    distmat_orients = distmat_orients.cpu().numpy()

    ## orient min
    distmat_orients_min = copy.deepcopy(distmat_orients)
    for item in dataset.test_tracks_vID:
        indis = np.array([int(jitem[:6]) - 1 for jitem in item[0]])
        distmat_orients_min[:, indis] = np.repeat(np.min(distmat_orients_min[:, indis], axis=1)[..., np.newaxis],
                                                  repeats=len(indis),
                                                  axis=1)
    distmat_new = copy.deepcopy(distmat_rerank + distmat_rerank3)
    mask = (distmat_orients_min < 0.03) & (distmat_new > 0.3)
    distmat_new[mask] = distmat_new[mask] + 0.5

    statistic_name = 'orient_ensemble_final'
    txt_dir = 'dist_test'
    dist_dir = os.path.join(cfg['OUTPUT_DIR'],"ensembling_final")
    generate_submit_txt_neu(dist_dir, txt_dir, distmat_new, statistic_name + '_eigene')

if __name__ == '__main__':
    main()

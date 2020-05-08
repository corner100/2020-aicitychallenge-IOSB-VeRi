import argparse
import os
import sys
import torch
print('curr device: ', torch.cuda.current_device())
import copy
import wandb

sys.path.append('.')
sys.path.append('..')
from data import make_data_loader
from modeling import build_model
from engine.inference import submit_val
from utils.random_seed import seed_everything

seed_everything()
def update_cfg(cfg, at_home=False):
    ## test
    if cfg['DATASETS.TRACKS']:
        print('tracks')
        cfg['TEST.IMS_PER_BATCH'] = 1
        if 'DATASETS.SYNTHETIC_TRACKS' not in cfg.keys():
            cfg['DATASETS.SYNTHETIC_TRACKS']=False
    else:
        cfg['DATASETS.NAMES'] = 'AI_CITY2020_TEST_VAL'
        print('no tracks')
        # cfg['TEST.IMS_PER_BATCH'] = 1
        if 'MODEL.FEAT_SIZE' not in cfg.keys():
            cfg['MODEL.FEAT_SIZE'] = 512
            cfg['MODEL.FEAT_SIZE'] = 2048
        if 'MODEL.FEAT_FC' not in cfg.keys():
            cfg['MODEL.FEAT_FC'] = False
        cfg['TEST.IMS_PER_BATCH'] = 128
    if 'DATASETS.SYNTHETIC' not in cfg.keys():
        cfg['DATASETS.SYNTHETIC'] = False
    if 'TRANSFORM.RANDOM_ROTATE' not in cfg.keys():
        cfg['TRANSFORM.RANDOM_ROTATE'] = False
    if 'TRANSFORM.PADDING_SIZE' not in cfg.keys():
        cfg['TRANSFORM.PADDING_SIZE'] = 10
    if 'MODEL.BINARY' not in cfg.keys():
        cfg['MODEL.BINARY'] = False
    if 'TRANSFORM.RANDOM_SHIFTSCALEROTATE' not in cfg.keys():
        cfg['TRANSFORM.RANDOM_SHIFTSCALEROTATE'] = False
    if 'DATASETS.DATASET_DIR' not in cfg.keys():
        if at_home:
            cfg['DATASETS.DATSET_DIR'] = "aic19-track2-reid"
            cfg['DATASETS.DATASET_DIR'] = "aic19-track2-reid"
        else:
            cfg['DATASETS.DATASET_DIR'] = "ai_city_challenge/2020/Track2/AIC20_track2_reid/AIC20_track2/AIC20_ReID"
    # cfg['DATASETS.DATASET_DIR']="ai_city_challenge/2020/Track2/AIC20_track2_reid/AIC20_track2/Cropped"
    if 'DATASETS.DATASET_TRACK_TXT_DIR' not in cfg.keys():
        if at_home:
            cfg['DATASETS.DATSET_TRACK_TXT_DIR'] = "/home/goat/Projects/ImageFolderStatistics/Track3Files"
        else:
            cfg[
                'DATASETS.DATSET_TRACK_TXT_DIR'] = "/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/ImageFolderStatistics/Track3Files"
        if at_home:
            cfg['DATASETS.DATASET_TRACK_TXT_DIR'] = "/home/goat/Projects/ImageFolderStatistics/Track3Files"
        else:
            cfg[
                'DATASETS.DATASET_TRACK_TXT_DIR'] = "/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/ImageFolderStatistics/Track3Files"
    if 'DATASETS.SYNTHETIC_DIR' not in cfg.keys():
        cfg['DATASETS.SYNTHETIC_DIR'] = "AIC20_track2_reid_simulation/AIC20_track2/SyntheticTracks"
    if 'DATASETS.TRACKS_FILE' not in cfg.keys():
        if at_home:
            cfg['DATASETS.TRACKS_FILE'] = '/home/goat/Projects/ImageFolderStatistics/Track3Files'
        else:
            cfg[
                'DATASETS.TRACKS_FILE'] = '/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/ImageFolderStatistics/Track3Files'
    if 'DATASETS.TRACKS_TRAIN_SAMPLER' not in cfg.keys():
        cfg['DATASETS.TRACKS_TRAIN_SAMPLER'] = 'randomly-consecutive'
    if 'DATASETS.TRACKS_TEST_SAMPLER' not in cfg.keys():
        cfg['DATASETS.TRACKS_TEST_SAMPLER'] = 'all'
    if 'MODEL.USE_SEGMENTATION' not in cfg.keys():
        cfg['MODEL.USE_SEGMENTATION'] = False

    return cfg

def generate_dist_txt_from_cfg(cfg, path_to_weights,path_to_results):
    #cfg = update_cfg(cfg, at_home=False)
    train_loader, val_loader, num_query, num_classes, dataset = make_data_loader(cfg)
    num_classes = cfg['numTrainIDs']
    model = build_model(cfg, num_classes)
    model.load_param(path_to_weights)
    submit_val(cfg, model, val_loader, num_query,directory=path_to_results,txt_dir='dist_test',dataset=dataset)
    print()
    print('test done')
    print()

def main():
    parser = argparse.ArgumentParser(description='cityAI')
    parser.add_argument('-wp', '--weights_path', help='path to weights')
    parser.add_argument('-rp', '--results_path', help='path to weights')
    args = parser.parse_args()

    wandb.init()
    cfg = wandb.config
    generate_dist_txt_from_cfg(cfg, path_to_weights=args.weights_path,path_to_results=args.results_path)

if __name__ == '__main__':
    main()

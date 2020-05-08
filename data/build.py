# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader
from .collate_batch import train_collate_fn, val_collate_fn,train_track_collate_fn,val_track_collate_fn,val_collate_fn_regression,train_collate_fn_regression
from .datasets import init_dataset, ImageDataset, ImageDatasetTracks, ImageDatasetOrientation
from .samplers import RandomIdentitySampler

def make_data_loader(cfg):
    num_workers = cfg['DATALOADER.NUM_WORKERS']
    dataset = init_dataset(cfg['DATASETS.NAMES'], cfg=cfg)
    num_classes = dataset.num_train_pids
    if cfg['DATASETS.TRACKS']:
        train_set = ImageDatasetTracks(dataset_train=dataset.train,dataset_query=dataset.query,dataset_gallery=dataset.gallery, tracks_vID=dataset.train_tracks_vID,tracks_from_vID=dataset.train_tracks_from_vID,path_by_name1=dataset.train_path_by_name,path_by_name2=None, cfg=cfg, is_train=True)
        collate_train_fn = train_track_collate_fn
        val_set = ImageDatasetTracks(dataset_train=dataset.train, dataset_query=dataset.query,
                                     dataset_gallery=dataset.gallery, tracks_vID=dataset.test_tracks_vID,
                                     tracks_from_vID=dataset.test_tracks_from_vID,
                                     path_by_name1=dataset.gallery_path_by_name,
                                     path_by_name2=dataset.query_path_by_name, cfg=cfg, is_train=False)
        collate_val_fn = val_track_collate_fn
        if cfg['DATALOADER.SAMPLER'] == 'softmax':
            train_loader = DataLoader(train_set, batch_size=cfg['SOLVER.IMS_PER_BATCH'], shuffle=True,
                                      num_workers=num_workers, collate_fn=collate_train_fn)
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg['SOLVER.IMS_PER_BATCH'],
                sampler=RandomIdentitySampler(dataset.train_tracks_vID, cfg['SOLVER.IMS_PER_BATCH'],
                                              cfg['DATALOADER.NUM_INSTANCE']),
                num_workers=num_workers, collate_fn=collate_train_fn
            )
        val_loader = DataLoader(val_set, batch_size=cfg['TEST.IMS_PER_BATCH'], shuffle=False, num_workers=num_workers,collate_fn=collate_val_fn)
    else:
        train_set = ImageDataset(dataset=dataset.train,cfg=cfg, is_train=True)
        collate_train_fn = train_collate_fn
        if cfg['DATALOADER.SAMPLER'] == 'softmax':
            train_loader = DataLoader(
                train_set, batch_size=cfg['SOLVER.IMS_PER_BATCH'], shuffle=True, num_workers=num_workers,
                collate_fn=collate_train_fn
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg['SOLVER.IMS_PER_BATCH'],
                sampler=RandomIdentitySampler(dataset.train, cfg['SOLVER.IMS_PER_BATCH'], cfg['DATALOADER.NUM_INSTANCE']),
                num_workers=num_workers, collate_fn=collate_train_fn
            )
        val_set = ImageDataset(dataset=dataset.query + dataset.gallery,cfg=cfg, is_train=False)
        collate_val_fn = val_collate_fn
        val_loader = DataLoader(
            val_set, batch_size=cfg['TEST.IMS_PER_BATCH'], shuffle=False, num_workers=num_workers,
            collate_fn=collate_val_fn
        )
    return train_loader, val_loader, len(dataset.query), num_classes, dataset

def make_data_loader_for_regression(cfg):
    num_workers = cfg['DATALOADER.NUM_WORKERS']
    dataset = init_dataset(cfg['DATASETS.NAMES'], cfg=cfg)
    train_set = ImageDatasetOrientation(dataset=dataset.train, cfg=cfg, is_train=True)
    collate_train_fn = train_collate_fn_regression
    train_loader = DataLoader(
        train_set, batch_size=cfg['SOLVER.IMS_PER_BATCH'], shuffle=True, num_workers=num_workers,
        collate_fn=collate_train_fn
    )
    val_set = ImageDatasetOrientation(dataset=dataset.val,cfg=cfg, is_train=False)
    collate_val_fn = val_collate_fn_regression
    val_loader = DataLoader(
        val_set, batch_size=cfg['TEST.IMS_PER_BATCH'], shuffle=False, num_workers=num_workers,
        collate_fn=collate_val_fn
    )
    return train_loader, val_loader, dataset

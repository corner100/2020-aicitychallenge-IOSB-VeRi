# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch

def train_collate_fn(batch):
    imgs, pids, camids, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def train_track_collate_fn(batch):
    track_imgs, label, camids, _, = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(track_imgs, dim=0), label, camids

def val_track_collate_fn(batch):
    track_imgs, label, camids, _ = zip(*batch)
    label = torch.tensor(label, dtype=torch.int64)
    return torch.stack(track_imgs, dim=0), label, camids

def train_collate_fn_regression(batch):
    imgs, x,y,z = zip(*batch)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=1)
    z = torch.tensor(z, dtype=torch.float32).unsqueeze(dim=1)
    return torch.stack(imgs, dim=0), torch.stack([x,y,z], dim=2)

def val_collate_fn_regression(batch):
    imgs, x,y,z = zip(*batch)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=1)
    z = torch.tensor(z, dtype=torch.float32).unsqueeze(dim=1)
    return torch.stack(imgs, dim=0), torch.stack([x,y,z], dim=2)
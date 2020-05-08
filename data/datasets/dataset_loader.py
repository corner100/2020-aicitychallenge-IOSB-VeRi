# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import cv2
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from ..transforms.transforms import RandomErasing
import albumentations
import sys
sys.path.append('..')

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ToTensorV2(albumentations.core.transforms_interface.BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg, is_train=True):
        self.dataset = dataset
        self.cfg = cfg
        self.is_train = is_train
        self.normalize_transform = albumentations.Normalize(mean=cfg['INPUT.PIXEL_MEAN'], std=cfg['INPUT.PIXEL_STD'])
        # self.normalize_transform_keypoints = T.Normalize(mean=[0.,0.,0.,0.], std=[255.,255.,255.,255.])
        if self.is_train:
            self.tranform_list_img0_both = []
            self.tranform_list_img1_rgb = []
            self.tranform_list_img2_both = []
            self.tranform_list_img3_rgb = []
            self.tranform_list_img4_mask = []
            self.tranform_list_img5_rgb = []
            # both
            self.tranform_list_img0_both.append(
                albumentations.Resize(height=cfg['INPUT.SIZE_TRAIN'][1], width=cfg['INPUT.SIZE_TRAIN'][0]))

            # rgb
            if self.cfg['TRANSFORM.RANDOM_BLUR']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.Blur())
            if self.cfg['TRANSFORM.RANDOM_CLAHE']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.CLAHE())
            if self.cfg['TRANSFORM.RANDOM_CONTRAST']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RandomContrast())
            if self.cfg['TRANSFORM.RANDOM_HUE_SATURATION_VALUE']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.HueSaturationValue())
            if self.cfg['TRANSFORM.RGB_SHIFT']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RGBShift())
            if self.cfg['TRANSFORM.RANDOM_BRIGHTNESS']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RandomBrightness())

            # both
            if self.cfg['TRANSFORM.RANDOM_HORIZONTAL_FLIP']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.HorizontalFlip(p=0.5))
            if 'TRANSFORM.RANDOM_VERTICAL_FLIP' in self.cfg.keys():
                if self.cfg['TRANSFORM.RANDOM_VERTICAL_FLIP']:
                    self.tranform_list_img2_both.append(
                        albumentations.augmentations.transforms.VerticalFlip(p=0.5))

            if self.cfg['TRANSFORM.PAD']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.PadIfNeeded(
                        min_height=cfg['INPUT.SIZE_TRAIN'][1] + cfg['TRANSFORM.PADDING_SIZE'] * 2,
                        min_width=cfg['INPUT.SIZE_TRAIN'][0] + cfg['TRANSFORM.PADDING_SIZE'] * 2, p=1, value=0,
                        mask_value=0, border_mode=cv2.BORDER_CONSTANT))
            if self.cfg['TRANSFORM.RANDOM_CROP']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.RandomCrop(height=cfg['INPUT.SIZE_TRAIN'][1],
                                                                       width=cfg['INPUT.SIZE_TRAIN'][0], p=1))
            if self.cfg['TRANSFORM.RANDOM_ROTATE']:
                self.tranform_list_img2_both.append(albumentations.augmentations.transforms.Rotate(
                    limit=(self.cfg['TRANSFORM.ROTATE_FACTOR1'], self.cfg['TRANSFORM.ROTATE_FACTOR1']), p=0.5))
            if self.cfg['TRANSFORM.RANDOM_SHIFTSCALEROTATE']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                                                             rotate_limit=self.cfg[
                                                                                 'TRANSFORM.ROTATE_FACTOR2'], p=0.5))
            # rgb
            self.tranform_list_img3_rgb.append(self.normalize_transform)
            # mask
            self.tranform_list_img4_mask.append(
                albumentations.Resize(height=64, width=64))
            # rgb
            self.tranform_list_img5_rgb.append(ToTensorV2())
            # both
            if self.cfg['TRANSFORM.CUTOUT']:
                self.aug_erase = RandomErasing(probability=self.cfg['TRANSFORM.CUTOUT_PROB'],
                                               mean=self.cfg['INPUT.PIXEL_MEAN'])
            else:
                self.aug_erase = None


        else:
            self.tranform_list_img0_both = []
            self.tranform_list_img1_rgb = []
            self.tranform_list_img2_both = []
            self.tranform_list_img3_rgb = []
            self.tranform_list_img4_mask = []
            self.tranform_list_img5_rgb = []

            # both
            self.tranform_list_img0_both.append(
                albumentations.Resize(height=cfg['INPUT.SIZE_TEST'][1], width=cfg['INPUT.SIZE_TEST'][0]))

            # rgb
            self.tranform_list_img3_rgb.append(self.normalize_transform)

            # mask
            self.tranform_list_img4_mask.append(
                albumentations.Resize(height=64, width=64))

            # rgb
            self.tranform_list_img5_rgb.append(ToTensorV2())

            # both
            self.aug_erase = None

        self.tranform_compose_img0_both = albumentations.Compose(self.tranform_list_img0_both)
        self.tranform_compose_img1_rgb = albumentations.Compose(self.tranform_list_img1_rgb)
        self.tranform_compose_img2_both = albumentations.Compose(self.tranform_list_img2_both)
        self.tranform_compose_img3_rgb = albumentations.Compose(self.tranform_list_img3_rgb)
        self.tranform_compose_img4_mask = albumentations.Compose(self.tranform_list_img4_mask)
        self.tranform_compose_img5_rgb = albumentations.Compose(self.tranform_list_img5_rgb)

    def __len__(self):
        return len(self.dataset)

    def transform(self, image):
        augmented1 = self.tranform_compose_img0_both(image=image)
        augmented2 = self.tranform_compose_img1_rgb(image=augmented1['image'])
        augmented3 = self.tranform_compose_img2_both(image=augmented2['image'])
        augmented4 = self.tranform_compose_img3_rgb(image=augmented3['image'])
        augmented5 = self.tranform_compose_img5_rgb(image=augmented4['image'])
        image = augmented5['image']
        if self.aug_erase is not None:
            image = self.aug_erase(img=image)
        return image

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img, pid, camid, img_path


class ImageDatasetTracks(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset_train, dataset_query, dataset_gallery, tracks_vID, tracks_from_vID, path_by_name1,
                 path_by_name2, cfg, is_train=True, use_train_as_val=False):
        self.dataset_train = dataset_train
        self.use_train_as_val = use_train_as_val
        if self.use_train_as_val:
            self.dataset_qg = dataset_train
        else:
            self.dataset_query = dataset_query
            self.dataset_gallery = dataset_gallery
            self.dataset_qg = self.dataset_query + self.dataset_gallery
        self.tracks_vID = tracks_vID
        self.tracks_from_vID = tracks_from_vID
        self.path_by_name1 = path_by_name1
        self.path_by_name2 = path_by_name2
        self.cfg = cfg
        self.is_train = is_train
        self.seq_len = cfg['DATASETS.TRACKS_LENGTH']
        self.normalize_transform = albumentations.Normalize(mean=cfg['INPUT.PIXEL_MEAN'], std=cfg['INPUT.PIXEL_STD'])
        if self.is_train and self.use_train_as_val == False:
            self.tracks_sampler = cfg[
                'DATASETS.TRACKS_TRAIN_SAMPLER']  # Randomly sample seq_len 'randomly-consecutive', or not randomly 'evenly'
            self.tranform_list_img0_both = []
            self.tranform_list_img1_rgb = []
            self.tranform_list_img2_both_horizontal_flip = []
            self.tranform_list_img2_both = []
            self.tranform_list_img3_rgb = []
            self.tranform_list_img4_mask = []
            self.tranform_list_img5_rgb = []
            # both
            self.tranform_list_img0_both.append(
                albumentations.Resize(height=cfg['INPUT.SIZE_TRAIN'][1], width=cfg['INPUT.SIZE_TRAIN'][0]))

            # rgb
            if self.cfg['TRANSFORM.RANDOM_BLUR']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.Blur())
            if self.cfg['TRANSFORM.RANDOM_CLAHE']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.CLAHE())
            if self.cfg['TRANSFORM.RANDOM_CONTRAST']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RandomContrast())
            if self.cfg['TRANSFORM.RANDOM_HUE_SATURATION_VALUE']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.HueSaturationValue())
            if self.cfg['TRANSFORM.RGB_SHIFT']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RGBShift())
            if self.cfg['TRANSFORM.RANDOM_BRIGHTNESS']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RandomBrightness())

            # both
            if self.cfg['TRANSFORM.RANDOM_HORIZONTAL_FLIP']:
                self.tranform_list_img2_both_horizontal_flip.append(
                    albumentations.augmentations.transforms.HorizontalFlip(p=1.0))

            if self.cfg['TRANSFORM.PAD']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.PadIfNeeded(
                        min_height=cfg['INPUT.SIZE_TRAIN'][1] + cfg['TRANSFORM.PADDING_SIZE'] * 2,
                        min_width=cfg['INPUT.SIZE_TRAIN'][0] + cfg['TRANSFORM.PADDING_SIZE'] * 2, p=1, value=0,
                        mask_value=0,
                        border_mode=cv2.BORDER_CONSTANT))
            if self.cfg['TRANSFORM.RANDOM_CROP']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.RandomCrop(height=cfg['INPUT.SIZE_TRAIN'][1],
                                                                       width=cfg['INPUT.SIZE_TRAIN'][0], p=1))
            # rgb
            self.tranform_list_img3_rgb.append(self.normalize_transform)
            # mask
            self.tranform_list_img4_mask.append(
                albumentations.Resize(height=64, width=64))
            # rgb
            self.tranform_list_img5_rgb.append(ToTensorV2())
            # both
            if self.cfg['TRANSFORM.CUTOUT']:
                self.aug_erase = RandomErasing(probability=self.cfg['TRANSFORM.CUTOUT_PROB'],
                                               mean=self.cfg['INPUT.PIXEL_MEAN'])
            else:
                self.aug_erase = None


        else:
            self.tracks_sampler = cfg[
                'DATASETS.TRACKS_TEST_SAMPLER']  # 'evenly' so first and last and some between. or 'all'
            self.tranform_list_img0_both = []
            self.tranform_list_img1_rgb = []
            self.tranform_list_img2_both_horizontal_flip = []
            self.tranform_list_img2_both = []
            self.tranform_list_img3_rgb = []
            self.tranform_list_img4_mask = []
            self.tranform_list_img5_rgb = []

            # both
            self.tranform_list_img0_both.append(
                albumentations.Resize(height=cfg['INPUT.SIZE_TEST'][1], width=cfg['INPUT.SIZE_TEST'][0]))

            # rgb
            self.tranform_list_img3_rgb.append(self.normalize_transform)

            # mask
            self.tranform_list_img4_mask.append(
                albumentations.Resize(height=64, width=64))

            # rgb
            self.tranform_list_img5_rgb.append(ToTensorV2())

            # both
            self.aug_erase = None

        self.tranform_compose_img0_both = albumentations.Compose(self.tranform_list_img0_both)
        self.tranform_compose_img1_rgb = albumentations.Compose(self.tranform_list_img1_rgb)
        self.tranform_compose_img2_both_horizontal_flip = albumentations.Compose(
            self.tranform_list_img2_both_horizontal_flip)
        self.tranform_compose_img2_both = albumentations.Compose(self.tranform_list_img2_both)
        self.tranform_compose_img3_rgb = albumentations.Compose(self.tranform_list_img3_rgb)
        self.tranform_compose_img4_mask = albumentations.Compose(self.tranform_list_img4_mask)
        self.tranform_compose_img5_rgb = albumentations.Compose(self.tranform_list_img5_rgb)

    def __len__(self):
        if self.is_train:
            return len(self.tracks_vID)
        else:
            if self.use_train_as_val:
                return len(self.tracks_vID)
            else:
                return len(self.dataset_query) + len(self.tracks_vID)

    def transform(self, image, is_h_flip=False):
        augmented1 = self.tranform_compose_img0_both(image=image)
        augmented2 = self.tranform_compose_img1_rgb(image=augmented1['image'])
        if is_h_flip:
            augmented2_h_flip = self.tranform_compose_img2_both_horizontal_flip(image=augmented2['image'])
        else:
            augmented2_h_flip = augmented2
        augmented3 = self.tranform_compose_img2_both(image=augmented2_h_flip['image'])
        augmented4 = self.tranform_compose_img3_rgb(image=augmented3['image'])
        augmented5 = self.tranform_compose_img5_rgb(image=augmented4['image'])
        image = augmented5['image']
        if self.aug_erase is not None:
            image = self.aug_erase(img=image)
        return image

    def __getitem__(self, index):
        if self.is_train or self.use_train_as_val:
            curr_track, vid, camid = self.tracks_vID[index]
        else:
            if index < len(self.dataset_query):
                img_path, vid, camid = self.dataset_query[index]
                curr_track = [img_path]
            else:
                curr_track, vid, camid = self.tracks_vID[index - len(self.dataset_query)]
        if self.is_train:
            if len(curr_track) < self.seq_len:
                n_start = int(np.floor((self.seq_len - len(curr_track)) / 2))
                n_end = int(np.ceil((self.seq_len - len(curr_track)) / 2))
                curr_track_new = [curr_track[0]] * n_start + curr_track + [curr_track[-1]] * n_end
            else:
                if self.tracks_sampler == 'evenly':
                    idx = np.round(np.linspace(0, len(curr_track) - 1, self.seq_len)).astype(int)
                    curr_track_new = list(np.array(curr_track)[idx])
                else:
                    n_start = random.randint(0, len(curr_track) - self.seq_len)
                    curr_track_new = curr_track[n_start:n_start + self.seq_len]
        else:
            if self.tracks_sampler == 'evenly':
                idx = np.round(np.linspace(0, len(curr_track) - 1, self.seq_len)).astype(int)
                curr_track_new = list(np.array(curr_track)[idx])
            else:
                n_start = int(np.floor(self.seq_len / 2))
                n_end = int(np.ceil(self.seq_len / 2))
                curr_track_new = [curr_track[0]] * n_start + curr_track + [curr_track[-1]] * n_end
        is_h_flip = bool(random.getrandbits(1))
        track_imgs = []
        for img_name in curr_track_new:
            if self.is_train or self.use_train_as_val:
                img_path = self.path_by_name1[img_name]
            else:
                if index < len(self.dataset_query):
                    img_path = img_name
                else:
                    img_path = self.path_by_name1[img_name]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.is_train:
                img = self.transform(img, is_h_flip=is_h_flip)
            else:
                img = self.transform(img)
            track_imgs.append(img)
        track_imgs = torch.stack(track_imgs, dim=0)

        return track_imgs, vid, camid, img_path

class ImageDatasetOrientation(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, cfg, is_train=True, test=False):
        self.dataset = dataset
        self.cfg = cfg
        self.is_train = is_train
        self.test = test
        self.normalize_transform = albumentations.Normalize(mean=cfg['INPUT.PIXEL_MEAN'], std=cfg['INPUT.PIXEL_STD'])

        if self.is_train:
            self.tranform_list_img0_both = []
            self.tranform_list_img1_rgb = []
            self.tranform_list_img2_both = []
            self.tranform_list_img3_rgb = []
            self.tranform_list_img4_mask = []
            self.tranform_list_img5_rgb = []
            # both
            self.tranform_list_img0_both.append(
                albumentations.Resize(height=cfg['INPUT.SIZE_TRAIN'][1], width=cfg['INPUT.SIZE_TRAIN'][0]))

            # rgb
            if self.cfg['TRANSFORM.RANDOM_BLUR']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.Blur())
            if self.cfg['TRANSFORM.RANDOM_CLAHE']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.CLAHE())
            if self.cfg['TRANSFORM.RANDOM_CONTRAST']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RandomContrast())
            if self.cfg['TRANSFORM.RANDOM_HUE_SATURATION_VALUE']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.HueSaturationValue())
            if self.cfg['TRANSFORM.RGB_SHIFT']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RGBShift())
            if self.cfg['TRANSFORM.RANDOM_BRIGHTNESS']:
                self.tranform_list_img1_rgb.append(albumentations.augmentations.transforms.RandomBrightness())

            # both
            if self.cfg['TRANSFORM.RANDOM_HORIZONTAL_FLIP']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.HorizontalFlip(p=0.5))

            if self.cfg['TRANSFORM.PAD']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.PadIfNeeded(
                        min_height=cfg['INPUT.SIZE_TRAIN'][1] + cfg['TRANSFORM.PADDING_SIZE'] * 2,
                        min_width=cfg['INPUT.SIZE_TRAIN'][0] + cfg['TRANSFORM.PADDING_SIZE'] * 2, p=1, value=0,
                        mask_value=0,
                        border_mode=cv2.BORDER_CONSTANT))
            if self.cfg['TRANSFORM.RANDOM_CROP']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.RandomCrop(height=cfg['INPUT.SIZE_TRAIN'][1],
                                                                       width=cfg['INPUT.SIZE_TRAIN'][0], p=1))
            if self.cfg['TRANSFORM.RANDOM_ROTATE']:
                self.tranform_list_img2_both.append(albumentations.augmentations.transforms.Rotate(
                    limit=(self.cfg['TRANSFORM.ROTATE_FACTOR1'], self.cfg['TRANSFORM.ROTATE_FACTOR1']), p=0.5))
            if self.cfg['TRANSFORM.RANDOM_SHIFTSCALEROTATE']:
                self.tranform_list_img2_both.append(
                    albumentations.augmentations.transforms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                                                             rotate_limit=self.cfg[
                                                                                 'TRANSFORM.ROTATE_FACTOR2'], p=0.5))
            # rgb
            self.tranform_list_img3_rgb.append(self.normalize_transform)
            # mask
            self.tranform_list_img4_mask.append(
                albumentations.Resize(height=64, width=64))
            # rgb
            self.tranform_list_img5_rgb.append(ToTensorV2())
            # both
            if self.cfg['TRANSFORM.CUTOUT']:
                self.aug_erase = RandomErasing(probability=self.cfg['TRANSFORM.CUTOUT_PROB'],
                                               mean=self.cfg['INPUT.PIXEL_MEAN'])
            else:
                self.aug_erase = None


        else:
            self.tranform_list_img0_both = []
            self.tranform_list_img1_rgb = []
            self.tranform_list_img2_both = []
            self.tranform_list_img3_rgb = []
            self.tranform_list_img4_mask = []
            self.tranform_list_img5_rgb = []

            # both
            self.tranform_list_img0_both.append(
                albumentations.Resize(height=cfg['INPUT.SIZE_TEST'][1], width=cfg['INPUT.SIZE_TEST'][0]))

            # rgb
            self.tranform_list_img3_rgb.append(self.normalize_transform)

            # mask
            self.tranform_list_img4_mask.append(
                albumentations.Resize(height=64, width=64))

            # rgb
            self.tranform_list_img5_rgb.append(ToTensorV2())

            # both
            self.aug_erase = None

        self.tranform_compose_img0_both = albumentations.Compose(self.tranform_list_img0_both)
        self.tranform_compose_img1_rgb = albumentations.Compose(self.tranform_list_img1_rgb)
        self.tranform_compose_img2_both = albumentations.Compose(self.tranform_list_img2_both)
        self.tranform_compose_img3_rgb = albumentations.Compose(self.tranform_list_img3_rgb)
        self.tranform_compose_img4_mask = albumentations.Compose(self.tranform_list_img4_mask)
        self.tranform_compose_img5_rgb = albumentations.Compose(self.tranform_list_img5_rgb)

    def __len__(self):
        return len(self.dataset)

    def transform(self, image):
        augmented1 = self.tranform_compose_img0_both(image=image)
        augmented2 = self.tranform_compose_img1_rgb(image=augmented1['image'])
        augmented3 = self.tranform_compose_img2_both(image=augmented2['image'])
        augmented4 = self.tranform_compose_img3_rgb(image=augmented3['image'])
        augmented5 = self.tranform_compose_img5_rgb(image=augmented4['image'])
        image = augmented5['image']
        if self.aug_erase is not None:
            image = self.aug_erase(img=image)
        return image

    def __getitem__(self, index):
        if self.test:
            img_path, _, _ = self.dataset[index]
            x = 0
            y = 0
            z = 0
        else:
            img_path, x, y, z,alpha,beta = self.dataset[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)
        return img, x, y, z
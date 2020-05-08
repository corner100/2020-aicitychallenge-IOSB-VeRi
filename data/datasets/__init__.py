# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dataset_loader import ImageDataset, ImageDatasetTracks,ImageDatasetOrientation
from .ai_city2020_test_val import AI_CITY2020_TEST_VAL
from .ai_city2020_tracks import AI_CITY2020_TRACKS
from .ai_city2020_orientation import AI_CITY2020_ORIENTATION

__factory = {
    'AI_CITY2020_TEST_VAL': AI_CITY2020_TEST_VAL,
    'AI_CITY2020_TRACKS': AI_CITY2020_TRACKS,
    'AI_CITY2020_ORIENTATION': AI_CITY2020_ORIENTATION,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

import glob
import re
import os
import random
from utils.random_seed import seed_everything
seed_everything()
from .bases import BaseImageDataset
import wandb
from collections import defaultdict

class AI_CITY2020_TRACKS(BaseImageDataset):
    """
       AI_CITY2020 by viktor
       """
    def __init__(self, cfg, **kwargs):
        super(AI_CITY2020_TRACKS, self).__init__()
        root = cfg['DATASETS.ROOT_DIR']
        self.cfg = cfg
        self.dataset_dir = cfg['DATASETS.DATASET_DIR']
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'image_train')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'image_test')
        self.train_track_path = os.path.join(self.dataset_dir, 'train_track_id.txt')
        self.test_track_path = os.path.join(self.dataset_dir, 'test_track_id.txt')
        self.train_camid_label_path = os.path.join(self.dataset_dir, 'train_label.xml')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True, track_path = self.train_track_path, train_camid_label_path= self.train_camid_label_path)
        # alle trainings daten aus dem datensatz extrahieren
        train_labels = [item[1] for item in train]
        train_labels_unique = list(set(train_labels))
        train_labels_unique.sort()
        pid2label_train = {pid: label for label, pid in enumerate(train_labels_unique)}
        for i, item in enumerate(train):
            train[i][1] = pid2label_train[item[1]]

        self.train = train
        self.query = self.read_real_test_dataset(self.query_dir)
        self.gallery = self.read_real_test_dataset(self.gallery_dir)

        self.train_path_by_name = {}
        for item in self.train:
            self.train_path_by_name[os.path.basename(item[0])] = item[0]

        path_train_tracks = os.path.join(self.dataset_dir,
            'train_track_id.txt')
        path_test_tracks = os.path.join(self.dataset_dir,
            'test_track_id.txt')

        f = open(path_train_tracks, 'r')
        txtTrackList = f.read().splitlines()
        trainTrackListAsList = [element.split() for element in txtTrackList if len(element.split()) > 0]
        for i,elem in enumerate(trainTrackListAsList):
            for j,el in enumerate(elem):
                trainTrackListAsList[i][j]=el.zfill(6)+".jpg"
        f.close()

        f = open(path_test_tracks, 'r')
        txtTrackList = f.read().splitlines()
        testTrackListAsList = [element.split() for element in txtTrackList if len(element.split()) > 0]
        for i,elem in enumerate(testTrackListAsList):
            for j,el in enumerate(elem):
                testTrackListAsList[i][j]=el.zfill(6)+".jpg"
        f.close()

        train_name_to_vID = {}
        train_name_to_camID = {}
        for i, item in enumerate(train):
            train_name_to_vID[os.path.basename(item[0])] = item[1]
            train_name_to_camID[os.path.basename(item[0])] = item[2]

        test_name_to_vID = {}
        test_name_to_camID = {}
        for i, item in enumerate(self.gallery):
            test_name_to_vID[os.path.basename(item[0])] = item[1]
            test_name_to_camID[os.path.basename(item[0])] = item[2]

        self.train_tracks_vID = [[item,train_name_to_vID[item[0].zfill(6)],train_name_to_camID[item[0].zfill(6)]]for item in trainTrackListAsList]
        self.test_tracks_vID = [[item,test_name_to_vID[item[0].zfill(6)],test_name_to_camID[item[0].zfill(6)]]for item in testTrackListAsList]

        self.train_tracks_from_vID = defaultdict(list)
        for item in self.train_tracks_vID:
            self.train_tracks_from_vID[item[1]].append(item[0])

        self.test_tracks_from_vID = {}
        for item in self.train_tracks_vID:
            self.test_tracks_from_vID[item[1]] = []
        for item in self.train_tracks_vID:
            self.test_tracks_from_vID[item[1]].append(item[0])

        self.gallery_path_by_name = {}
        for item in self.gallery:
            self.gallery_path_by_name[os.path.basename(item[0])] = item[0]

        self.query_path_by_name = {}
        for item in self.query:
            self.query_path_by_name[os.path.basename(item[0])] = item[0]

        self.test_track_indice_from_test_name = {}
        for i,item in enumerate(testTrackListAsList):
            for test_name in item:
                self.test_track_indice_from_test_name[test_name.zfill(6)]=i

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        if wandb.run is not None:
            wandb.config.update({'numTrainIDs': self.num_train_pids, 'numTrainImages': self.num_train_imgs, 'numTrainCams': self.num_train_cams})
            wandb.config.update({'numQueryIDs': self.num_query_pids, 'numQueryImages': self.num_query_imgs, 'numQueryCams': self.num_query_cams})
            wandb.config.update({'numGalleryIDs': self.num_gallery_pids, 'numGalleryImages': self.num_gallery_imgs, 'numGalleryCams': self.num_gallery_cams})

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def read_real_test_dataset(self, dir):
        image_paths = glob.glob(os.path.join(dir, '*.jpg'))
        image_paths.sort()
        dataset = [[item, None, None] for item in image_paths]
        return dataset

    def _process_dir(self, dir_path,track_path, relabel=False, train_camid_label_path=None):
        if train_camid_label_path is not None:
            with open(train_camid_label_path, 'r') as f:
                xml = f.readlines()

            curr_dict = {}
            vIDs_list = []
            camIDs_list = []
            for line in xml[3:-2]:
                result = re.search('Item imageName="(.*)" vehicleID="', line)
                image_file_name = result.group(1)
                result = re.search('vehicleID="(.*)" cameraID="', line)
                vehicleID = result.group(1)
                vIDs_list.append(vehicleID)
                result = re.search('cameraID="(.*)" />', line)
                cameraID = result.group(1)
                camIDs_list.append(cameraID)
                curr_dict[image_file_name] = [vehicleID,cameraID]
        dataset = []

        for line in xml[3:-2]:
            result = re.search('Item imageName="(.*)" vehicleID="', line)
            image_file_name = result.group(1)
            result = re.search('vehicleID="(.*)" cameraID="', line)
            vehicleID = result.group(1)
            vIDs_list.append(vehicleID)
            result = re.search('cameraID="(.*)" />', line)
            cameraID = result.group(1)
            curr_dict[image_file_name] = [vehicleID, cameraID]

            vehicle_image_file_path = os.path.join(dir_path, image_file_name)
            dataset.append([vehicle_image_file_path, vehicleID, cameraID])

        return dataset
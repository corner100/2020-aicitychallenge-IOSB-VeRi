import re
import glob
import os
from utils.random_seed import seed_everything

seed_everything()
from .bases import BaseImageDataset
import wandb


class AI_CITY2020_TEST_VAL(BaseImageDataset):
    """
       AI_CITY2020 by viktor
       """

    def __init__(self, cfg, **kwargs):
        super(AI_CITY2020_TEST_VAL, self).__init__()
        root = cfg['DATASETS.ROOT_DIR']
        self.cfg = cfg
        self.dataset_dir = cfg['DATASETS.DATASET_DIR']
        self.synthetic_dir = cfg['DATASETS.SYNTHETIC_DIR']
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'image_train')
        self.query_dir = os.path.join(self.dataset_dir, 'image_query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'image_test')
        self.train_track_path = os.path.join(self.dataset_dir, 'train_track_id.txt')
        self.test_track_path = os.path.join(self.dataset_dir, 'test_track_id.txt')
        self.train_camid_label_path = os.path.join(self.dataset_dir, 'train_label.xml')
        self.synthetic_dataset_path = os.path.join(root, self.synthetic_dir)

        self._check_before_run()

        train = self.read_real_dataset()

        # try:
        if cfg['DATASETS.SYNTHETIC']:
            train_synthetic = self.read_synthetic_dataset()
            train += train_synthetic

        self.train = train
        self.query = self.read_real_test_dataset(self.query_dir)
        self.gallery = self.read_real_test_dataset(self.gallery_dir)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        if wandb.run is not None:
            wandb.config.update({'numTrainIDs': self.num_train_pids, 'numTrainImages': self.num_train_imgs,
                                 'numTrainCams': self.num_train_cams})

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, train_camid_label_path=None):

        if train_camid_label_path is not None:
            with open(train_camid_label_path, 'r') as f:
                xml = f.readlines()

            curr_dict = {}
            vIDs_list = []  # nur zum gucken erstellt, ob wirklich nur 333 ids existieren. len(set(vIDs_list)) ist 333
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
                curr_dict[image_file_name] = [vehicleID, cameraID]
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

    def read_real_test_dataset(self, dir):
        image_paths = glob.glob(os.path.join(dir, '*.jpg'))
        image_paths.sort()
        dataset = [[item, None, None] for item in image_paths]
        return dataset

    def read_real_dataset(self):
        dir_path = os.path.join(self.dataset_dir, 'image_train')
        label_path = os.path.join(self.dataset_dir, 'train_label.xml')
        with open(label_path, 'r') as f:
            xml = f.readlines()

        curr_dict = {}
        vIDs_list = []  # nur zum gucken erstellt, ob wirklich nur 333 ids existieren. len(set(vIDs_list)) ist 333
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
            curr_dict[image_file_name] = [vehicleID, cameraID]
        vid2label = {pid: label + 333 for label, pid in enumerate(set(vIDs_list))}
        camID2label = {camID: label + 40 for label, camID in enumerate(set(camIDs_list))}
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

            # pid = pid2label[vehicleID]
            # camid = camID2label[cameraID]
            vehicle_image_file_path = os.path.join(dir_path, image_file_name)
            dataset.append([vehicle_image_file_path, vehicleID, cameraID])

        train_labels_unique = list(set(vIDs_list))
        train_labels_unique.sort()
        vid2label_train = {pid: label for label, pid in enumerate(train_labels_unique)}
        for i, item in enumerate(dataset):
            dataset[i][1] = vid2label_train[item[1]]
            dataset[i][2] = camID2label[item[2]]
        return dataset

    def read_synthetic_dataset(self):
        dir_path = os.path.join(self.synthetic_dataset_path, 'image_train')
        label_path = os.path.join(self.synthetic_dataset_path, 'train_label.xml')
        with open(label_path, 'r') as f:
            xml = f.readlines()

        curr_dict = {}
        vIDs_list = []  # nur zum gucken erstellt, ob wirklich nur 333 ids existieren. len(set(vIDs_list)) ist 333
        camIDs_list = []
        for line in xml[3:-2]:
            result = re.search('Item imageName="(.*)" vehicleID="', line)
            image_file_name = result.group(1)
            result = re.search('vehicleID="(.*)" cameraID="', line)
            vehicleID = result.group(1)
            vIDs_list.append(vehicleID)
            result = re.search('cameraID="(.*)" colorID', line)
            cameraID = result.group(1)
            camIDs_list.append(cameraID)
            curr_dict[image_file_name] = [vehicleID, cameraID]
        vid2label = {pid: label + 333 for label, pid in enumerate(set(vIDs_list))}
        camID2label = {camID: label + 40 for label, camID in enumerate(set(camIDs_list))}
        dataset = []

        for line in xml[3:-2]:
            result = re.search('Item imageName="(.*)" vehicleID="', line)
            image_file_name = result.group(1)
            result = re.search('vehicleID="(.*)" cameraID="', line)
            vehicleID = result.group(1)
            vIDs_list.append(vehicleID)
            result = re.search('cameraID="(.*)" colorID', line)
            cameraID = result.group(1)
            curr_dict[image_file_name] = [vehicleID, cameraID]

            # pid = pid2label[vehicleID]
            # camid = camID2label[cameraID]
            vehicle_image_file_path = os.path.join(dir_path, image_file_name)
            dataset.append([vehicle_image_file_path, vehicleID, cameraID])

        train_labels_unique = list(set(vIDs_list))
        train_labels_unique.sort()
        vid2label_train = {pid: label + 333 for label, pid in enumerate(train_labels_unique)}
        for i, item in enumerate(dataset):
            dataset[i][1] = vid2label_train[item[1]]
            dataset[i][2] = camID2label[item[2]]
        return dataset

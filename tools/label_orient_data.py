import argparse
import os
import sys
import torch
from torch.backends import cudnn
import numpy as np
import wandb
import glob
sys.path.append('.')
sys.path.append('..')
from data.datasets import init_dataset, ImageDatasetOrientation
from modeling import build_regression_model
import cv2

parser = argparse.ArgumentParser(description='cityAI')
parser.add_argument('-wp', '--weights_path', help='path to weights')
parser.add_argument('-rp', '--results_path', help='path to weights')
args = parser.parse_args()

train_test_query = 'image_train'

label_dest_dir = '.../OrientationReal/'+train_test_query
label_paths = glob.glob(os.path.join(label_dest_dir,'*'))
label_names = [os.path.basename(path).split('.')[0] for path in label_paths]

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
wandb.init()
cfg = wandb.config

if cfg['MODEL.DEVICE'] == "cuda":
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['MODEL.DEVICE_ID']
cudnn.benchmark = True
print('load dataset')
dataset = init_dataset('AI_CITY2020_TEST_VAL', cfg=cfg,fold=1,eval_mode=False)
val_set = ImageDatasetOrientation(dataset.train, cfg,is_train=False, test=True)

print('build model')
model = build_regression_model(cfg)
weights_path = os.path.join(args.weights_path)
print('load pretrained weights')
model.load_param(weights_path)
model.eval()

dataset_synthetic = init_dataset('AI_CITY2020_ORIENTATION', cfg=cfg,fold=1,eval_mode=False)
val_set_synthetic = ImageDatasetOrientation(dataset_synthetic.train, cfg,is_train=False, test=False)
train_feats_real = np.load(os.path.join(cfg['OUTPUT_DIR'], 'feats_train.npy'))

real_paths = np.array([val_set.dataset[i][0] for i in range(len(val_set.dataset))])
names = np.array([os.path.basename(val_set.dataset[i][0]).split('.')[0] for i in range(len(val_set.dataset))])
names_found = [os.path.basename(path).split('.')[0] for i,path in enumerate(real_paths) if os.path.basename(path).split('.')[0] in label_names]
names_not_found = list(set(names) -set(names_found))
names_not_found.sort()
names_not_found_int = [int(item)-1 for item in names_not_found]
names_found_int = [int(item)-1 for item in names_found]
names_int = [int(item)-1 for item in names]
remain = np.array([i for i,path in enumerate(real_paths) if os.path.basename(path).split('.')[0] not in label_names])
real_paths_remain = real_paths[remain]
train_feats_real_remain = train_feats_real[remain]

synth_image_num = 1000
image_num = 0
img_synth = cv2.imread(val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][0], 1)
img_synth_for_torch = cv2.cvtColor(img_synth, cv2.COLOR_BGR2RGB)
while True:
    img_synth_torch = val_set_synthetic.transform(image=img_synth_for_torch)
    with torch.no_grad():
        score_synth, feat_synth = model(img_synth_torch.unsqueeze(dim=0))
    dist_array_remain = np.sum((train_feats_real_remain - feat_synth.cpu().numpy()) ** 2, axis=1)
    dist_indices_remain = np.argsort(dist_array_remain)
    path_real = real_paths_remain[dist_indices_remain[image_num]]
    print(path_real)
    print(dist_array_remain[:5])
    print(dist_indices_remain[:5])
    img = cv2.imread(path_real, 1)
    img = cv2.resize(img, (640, 480))
    img_synth = cv2.resize(img_synth, (640, 480))
    res = np.concatenate((img_synth, img), axis=1)
    cv2.imshow('image', res)
    print('remaining ', len(real_paths_remain),' von ', len(real_paths))
    print()
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('a'):
        synth_image_num -= 1
        print(synth_image_num % len(val_set_synthetic.dataset))
        img_synth = cv2.imread(val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][0], 1)
        img_synth_for_torch = cv2.cvtColor(img_synth, cv2.COLOR_BGR2RGB)
        image_num = 0
    elif k == ord('d'):
        synth_image_num += 1
        print(synth_image_num % len(val_set_synthetic.dataset))
        img_synth = cv2.imread(val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][0], 1)
        img_synth_for_torch = cv2.cvtColor(img_synth, cv2.COLOR_BGR2RGB)
        image_num = 0
    elif k == ord('n'):
        image_num += 1
    elif k == ord('b'):
        image_num -= 1
    elif k == ord('o'):
        x = val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][1]
        y = val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][2]
        z = val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][3]
        alpha = val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][4]
        beta = val_set_synthetic.dataset[synth_image_num % len(val_set_synthetic.dataset)][5]
        print('x: ',x)
        print('y: ',y)
        print('z: ',z)
        print('alpha: ',alpha)
        print('beta: ',beta)
        print(path_real)
        label_name = os.path.basename(path_real).split('.')[0]
        names_found.append(label_name)
        names_found_int.append(int(label_name)-1)
        names_not_found_int = list(set(names_int)-set(names_found_int))
        names_not_found_int.sort()
        names_not_found_int = np.array(names_not_found_int)
        label_path = os.path.join(label_dest_dir,label_name)
        np.save(label_path,np.array([float(alpha),float(beta)]))
        label_names.append(label_name)
        remain = names_not_found_int
        real_paths_remain = real_paths[remain]
        train_feats_real_remain = train_feats_real[remain]
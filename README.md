## Description
Implementation of the [cityAI CVPR 2020 paper](https://drive.google.com/open?id=1Cm25GNSpBQ11t4jujq6W8wA5JrxLDB2L)<br />
Our Code is based on the [bags of tricks repo](https://github.com/michuanhaohao/reid-strong-baseline). 
## Table of contents

- [Description](#description)
- [Orientation dataset generation](#dataset-generation)
- [Dataset download](#dataset-download )
- [Training the models from the paper](#training-the-models-from-the-paper)
- [Download the models from the paper](#download-the-models-from-the-paper)
- [Generate submission  and files for visualisation](#generate-submission-and-files-for-visualisation)
- [Citation](#citation)

## Orientation dataset generation (optional)
To generate the orientation dataset download the [vehicleX code](https://github.com/yorkeyao/VehicleX)<br />
Move the script unity3D/GenerateOrientationDataWithDistractors.cs in the project and change the paths to the background images you want to use.<br />
Click hold and move the script on the camera. Delete all opject including the lights in the scene. Run.<br />
After generating the data use the vehicleX SPGAN to domain adapt the orientation data.<br />
Or you can download the dataset.<br />
## Dataset download 
Download the cityAI datasets and the synthetic orient data from [link](https://drive.google.com/open?id=1huWCKzluNBwxz9D2pGqcJoz1wbJyVu-i) and place it in data/data/cityAI.<br />
The labels for the orientation data is [here](https://drive.google.com/open?id=1yLbbWKH-Q-rrtSCeFDmJ7MnrSQ1jEZua).
```
data
--- data
--- --- cityAI
--- --- --- real
--- --- --- synthetic
--- --- orient
--- --- --- synthetic
--- --- --- real
```
## Training the models from the paper (optional)
To train the I2I 256x256 run
```
2020-aicitychallenge-IOSB-VeRi/logs$ WANDB_MODE=dryrun  WANDB_CONFIG_PATHS=../config/config_singlenet_256x256.yaml CUDA_VISIBLE_DEVICES=0 taskset -c 6-10 python3 /net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/2020-aicitychallenge-IOSB-VeRi/tools/train.py
```
Similar with the I2I 320x320 model and with the V2V models.
```
2020-aicitychallenge-IOSB-VeRi/logs$ WANDB_MODE=dryrun  WANDB_CONFIG_PATHS=../config/resnet152_i2i_320x320.yaml CUDA_VISIBLE_DEVICES=0 taskset -c 6-10 python3 /net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/2020-aicitychallenge-IOSB-VeRi/tools/train.py
2020-aicitychallenge-IOSB-VeRi/logs$ WANDB_MODE=dryrun  WANDB_CONFIG_PATHS=../config/resnet152_v2v_256x256.yaml CUDA_VISIBLE_DEVICES=0 taskset -c 6-10 python3 /net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/2020-aicitychallenge-IOSB-VeRi/tools/train.py
2020-aicitychallenge-IOSB-VeRi/logs$ WANDB_MODE=dryrun  WANDB_CONFIG_PATHS=../config/resnet152_v2v_320x320.yaml CUDA_VISIBLE_DEVICES=0 taskset -c 6-10 python3 /net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/2020-aicitychallenge-IOSB-VeRi/tools/train.py
```
To train the orientation model
```
2020-aicitychallenge-IOSB-VeRi/logs$ WANDB_MODE=dryrun  WANDB_CONFIG_PATHS=../config/orient_net.yaml CUDA_VISIBLE_DEVICES=0 taskset -c 6-10 python3 /net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/2020-aicitychallenge-IOSB-VeRi/tools/train_regression.py
```

the results will be saved in logs/wandb/
Or you can download the pretrained models.<br />
## Download the pretrained models from the paper
Download the pretrained models [link](https://drive.google.com/open?id=1mQLlwE173bKt9UrKd_HU8tLWADcv6aKu) and move them to the directory pretrained.
## Generate submission and files for visualisation
To generate the submit.txt and the files for visualisation run<br />
Before ensembling
```
CUDA_VISIBLE_DEVICES=3 WANDB_CONFIG_PATHS=2020-aicitychallenge-IOSB-VeRi/config/config_singlenet_256x256.yaml WANDB_MODE=dryrun python3 2020-aicitychallenge-IOSB-VeRi/tools/generate_txt_files_for_visualisation.py -wp
/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/VehicleReID/logs/cityai/2020/wandb/run-20200325_181358-7zlgl6yc_my_logs_justfold1/resnet152_model_63.pth
-rp
/net/merkur/storage/deeplearning/users/eckvik/WorkspaceNeu/VReID/2020-aicitychallenge-IOSB-VeRi/logs/wandb/dryrun-20200507_112944-ctijgkkg_my_logs_justfold1
```
After ensembling
```
CUDA_VISIBLE_DEVICES=3 WANDB_CONFIG_PATHS=2020-aicitychallenge-IOSB-VeRi/config/config_singlenet_256x256.yaml WANDB_MODE=dryrun python3 2020-aicitychallenge-IOSB-VeRi/tools/ensembling_different_models.py
```
ensembling_different_models.py generates the submit file

## Citation

```
@InProceedings{Eckstein_2020_CVPR_Workshops,
author = {Eckstein, Viktor and Schumann, Arne and Specker, Andreas},
title = {Large Scale Vehicle Re-Identification by Knowledge Transfer From Simulated Data and Temporal Attention},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
} 
```
Bags of tricks
```
@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM, 
author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}}, 
journal={IEEE Transactions on Multimedia}, 
title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification}, 
year={2019}, 
pages={1-1}, 
doi={10.1109/TMM.2019.2958756}, 
ISSN={1941-0077}, 
}
```
VehicleX
```
@article{yao2019simulating,
  title={Simulating Content Consistent Vehicle Datasets with Attribute Descent},
  author={Yao, Yue and Zheng, Liang and Yang, Xiaodong and Naphade, Milind and Gedeon, Tom},
  journal={arXiv preprint arXiv:1912.08855},
  year={2019}
}
@inproceedings{tang2019pamtri,
  title={Pamtri: Pose-aware multi-task learning for vehicle re-identification using highly randomized synthetic data},
  author={Tang, Zheng and Naphade, Milind and Birchfield, Stan and Tremblay, Jonathan and Hodge, William and Kumar, Ratnesh and Wang, Shuo and Yang, Xiaodong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={211--220},
  year={2019}
}
@inproceedings{image-image18,
 author    = {Weijian Deng and
              Liang Zheng and
              Qixiang Ye and
              Guoliang Kang and
              Yi Yang and
              Jianbin Jiao},
 title     = {Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity
              for Person Re-identification},
 booktitle = {CVPR},
 year      = {2018},
}
```

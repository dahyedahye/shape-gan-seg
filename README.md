# Unsupervised Segmentation incorporating Shape Prior via Generative Adversarial Networks (ICCV 2021)

This repository contains the official PyTorch implementation of:

**Unsupervised Segmentation incorporating Shape Prior
via Generative Adversarial Networks**

[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Unsupervised_Segmentation_Incorporating_Shape_Prior_via_Generative_Adversarial_Networks_ICCV_2021_paper.pdf) | [project webpage](https://dahyedahye.github.io/shape-gan-seg/) | [supplementary](https://drive.google.com/file/d/1p-Ej0vqYA6hb8ciumQcMosNdW4wkYne6/view?usp=sharing)

## Setup
* The code is tested with Python 3.7-3.8.8 and PyTorch 1.6-1.9.1
  
## Demo
* [LSUN car sample data](https://github.com/dahyedahye/shape-gan-seg/blob/main/demo_shape_seg.ipynb)
## Train/Inference
* The shell script files in `./scripts` contain commands with argument settings to run training/inference.
* Each LSUN data class uses different values for the arguments `num_train_split` and `num_val_split` in the shell script files as follows:
  | class    | `num_train_split` | `num_val_split` |
  |:--------:|:-----------------:|:---------------:|
  | airplane | 71,590            | 7,954           |
  | boat     | 49,642            | 5,516           |
  | car      | 75,973            | 8,441           |
  | chair    | 60,606            | 6,734           |
### Segmentation with shape incorporation (Sec. 3.2. of the paper)
**Training**
* How to run:
  1. Set the configuration for training in `./scripts/train_shape_seg.sh`.
  2. Run the following commands:
     ```bash
     $ cd ./scripts
     $ chmod +x train_shape_seg.sh
     $ ./train_shape_seg.sh
     ```
**Inference**
* How to run:
  1. Set the configuration for training in `./scripts/inference_shape_seg.sh`.
  2. Run the following commands:
     ```bash
     $ cd ./scripts
     $ chmod +x inference_shape_seg.sh
     $ ./inference_shape_seg.sh
     ```
### Intrinsic decomposition (Sec. 3.1. of the paper)
* How to run:
  1. Set the configuration for training in `./scripts/train_intrinsic.sh`.
  2. Run the following commands:
     ```bash
     $ cd ./scripts
     $ chmod +x train_intrinsic.sh
     $ ./train_intrinsic.sh
     ```
## Pretrained models
**Segmentation with shape incorporation via GAN on original images**
  * [LSUN airplane](https://github.com/dahyedahye/shape-gan-seg/blob/main/pretrained/lsun/shape_seg/lsun_airplane/trained_model_by_best_val_iou.pth)
  * [LSUN boat](https://github.com/dahyedahye/shape-gan-seg/blob/main/pretrained/lsun/shape_seg/lsun_boat/trained_model_by_best_val_iou.pth)
  * [LSUN car](https://github.com/dahyedahye/shape-gan-seg/blob/main/pretrained/lsun/shape_seg/lsun_car/trained_model_by_best_val_iou.pth)
## Dataset
**LSUN dataset with pseudo labels for segmentation**
  * [download link via google drive](https://drive.google.com/file/d/1y_b0DIrECcNTrUgi-YoHyn0901bmyimK/view?usp=sharing) (file size - compressed/original: 8.75G/30G)

**Shape prior**
  * [download link via google drive](https://drive.google.com/file/d/1kVlhiqyE-GnoZRYDYYkyF8FHcehFUtmG/view?usp=sharing) (file size - compressed/original: 104.9M/9.4G)
  
## Update Plan
1. Source code to construct LSUN dataset with pseudo labels for segmentation (byNov 2021)
2. Source code to create synthetic dataset (by Nov 2021)
3. Pretrained models for intrinsic decomposition (by Nov 2021)
4. Source code to train segmentation with shape incorporation using intrinsic representation (by Nov 2021)
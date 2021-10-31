# Unsupervised Segmentation incorporating Shape Prior via Generative Adversarial Networks (ICCV 2021)

This repository contains the official PyTorch implementation of:

**Unsupervised Segmentation incorporating Shape Prior
via Generative Adversarial Networks**

[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Unsupervised_Segmentation_Incorporating_Shape_Prior_via_Generative_Adversarial_Networks_ICCV_2021_paper.pdf) | [project webpage](https://dahyedahye.github.io/shape-gan-seg/)


## News
* Codes for intrinsic decomposition has been uploaded. (29.10.2021)
* Codes to inference segmentation has been uploaded. (31.10.2021)

## Setup
* The code is tested with Python 3.7-3.8.8 and PyTorch 1.6-1.9.1
  
## Currently Available
### Source code for segmentation with shape incorporation (Sec. 3.2. of the paper)
**Training**
* The shell script `./scripts/train_shape_seg.sh` contains commands to train the segmentation network with shape incorporation.
* We will be sharing the dataset used in the code soon (by Nov 2021).
<!-- ```bash
$ cd ./scripts
$ chmod +x train_shape_seg.sh
$ ./train_shape_seg.sh
``` -->
**Inference**
* The shell script `./scripts/inference_shape_seg.sh` contains commands to infer segmentation.
### Source code to train intrinsic decomposition (Sec. 3.1. of the paper)
* The shell script `./scripts/train_intrinsic.sh` contains commands to train the intrinsic decomposition network.

## Update Plan
1. LSUN dataset with pseudo labels for segmentation (by Nov 2021)
2. Demo codes (by Nov 2021)
3. Source code to construct LSUN dataset with pseudo labels for segmentation (byNov 2021)
4. Source code to create synthetic dataset (by Nov 2021)
5. Supplementary material with experimental details & network architecture (by Nov 2021)
6. Pretrained models (by Nov 2021)
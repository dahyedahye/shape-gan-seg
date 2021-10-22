# Unsupervised Segmentation incorporating Shape Prior via Generative Adversarial Networks (ICCV 2021)

This repository contains the official PyTorch implementation of:

**Unsupervised Segmentation incorporating Shape Prior
via Generative Adversarial Networks**

[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Unsupervised_Segmentation_Incorporating_Shape_Prior_via_Generative_Adversarial_Networks_ICCV_2021_paper.pdf)|[project webpage](https://dahyedahye.github.io/shape-gan-seg/)

Currently, only the **segmentation with shape prior via GAN** part is available. The **full source code** will be released by **the end of October 2021**! :construction: :wrench: :woman_technologist:

## Setup
* The code is tested with Python 3.7-3.8.8 and PyTorch 1.6-1.9.1
  
## Currently Available
### Source code to train segmentation with shape incorporation (Sec. 3.2. of the paper)
* The shell script `./scripts/train_shape_seg.sh` contains commands to train the segmentation network with shape incorporation.
* We will be sharing the dataset used in the code soon (by Oct 2021).
<!-- ```bash
$ cd ./scripts
$ chmod +x train_shape_seg.sh
$ ./train_shape_seg.sh
``` -->

## Update Plan
1. Source code to train intrinsic decomposition (by Oct 2021)
2. LSUN dataset with pseudo labels for segmentation (by Oct 2021)
3. Source code to construct LSUN dataset with pseudo labels for segmentation (by Oct 2021)
4. Source code to create synthetic dataset (by Oct 2021)
5. Supplementary material with experimental details & network architecture (by Oct 2021)
6. Pretrained models (by Oct 2021)
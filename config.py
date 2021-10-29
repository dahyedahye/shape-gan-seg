import argparse
import os
import numpy as np
import torch

def str2bool(v):
    return v.lower() in ('true', '1')

d = os.path.dirname
parser = argparse.ArgumentParser(description='Unsupervised Segmentation incorporating Shape Prior via GANs')
path_arg = parser.add_argument_group('Experiment Config')
path_arg = parser.add_argument_group('Data Config')
path_arg = parser.add_argument_group('Networks Config')
path_arg = parser.add_argument_group('Training Environment Config')
path_arg = parser.add_argument_group('Coefficient Config')
path_arg = parser.add_argument_group('Optimization Config')

path_arg = parser.add_argument_group('Experiment Config')
path_arg.add_argument('--output_dir', type=str, default='./results',
    help='Directory where the experiment results will be stored')
path_arg.add_argument('--monitor_interval', type=int, default=200,
    help='Interval to monitor training results')
path_arg.add_argument('--num_plot_img', type=int, default=64,
    help='Number of sample images to plot per batch')

path_arg = parser.add_argument_group('Data Config')
path_arg.add_argument('--height', type=int, default=64,
    help='training image height to be resized by')
path_arg.add_argument('--width', type=int, default=64,
    help='training image width to be resized by')
path_arg.add_argument('--segment_data_train_dir', type=str, default='../data/occluded/occ_noisy_64_train/data',
    help='Directory of segmentation data to be used for training')
path_arg.add_argument('--segment_data_val_dir', type=str, default='../data/occluded/occ_noisy_64_train/data',
    help='Directory of segmentation data to be used for validation')
path_arg.add_argument('--segment_data_test_dir', type=str, default='../data/occluded/occ_noisy_64_test/data',
    help='Directory of segmentation data to be used for testing')
path_arg.add_argument('--prior_data_dir', type=str, default='../data/occluded/clean_square_64/',
    help='Directory of prior data to be used')
path_arg.add_argument('--num_train_split', type=int, default=50000,
    help='the number of train split from training data')
path_arg.add_argument('--num_val_split', type=int, default=10000,
    help='the number of validation split from training data')
path_arg.add_argument('--min_scale', type=float, default=0.85,
    help='Min size to scale in data transformation for prior data')

path_arg = parser.add_argument_group('Networks Config')
path_arg.add_argument('--trained_ckpt_path', type=str, default=None,
    help='Path of trained model checkpoint to be loaded')
path_arg.add_argument('--trained_intrinsic_ckpt_path', type=str, default=None,
    help='Path of trained intrinsic decomposition model checkpoint to be loaded')
path_arg.add_argument('--num_in_channel', type=int, default=3,
    help='Number of channel of input')
path_arg.add_argument('--num_out_channel', type=int, default=3,
    help='Number of channel of output')
path_arg.add_argument('--network_intrinsic', type=str, default='CNN',
    help='Network architecture to be used for intrinsic decomposition')
path_arg.add_argument('--network_segment_generator', type=str, default='CNN',
    help='Network architecture to be used as a generator that obtains segmentation proposals')
path_arg.add_argument('--network_discriminator', type=str, default='CNN',
    help='Network architecture to be used as a discriminator')

path_arg = parser.add_argument_group('Training Environment Config')
path_arg.add_argument('--num_workers', type=int, default=16,
    help='Number of subprocesses to use for data loading for training')
path_arg.add_argument('--multi_gpu', type=str2bool, default=True,
    help='Decide whether to use multiple GPUs')
path_arg.add_argument('--num_gpu', type=int, default=4,
    help='Number of GPUs to be used')
path_arg.add_argument('--cuda_id', type=str, default='cuda:0',
    help='GPU ID to be used')

path_arg = parser.add_argument_group('Coefficient Config')
path_arg.add_argument('--lambda_tv_intrin', type=float, default=1.0,
    help='Lambda in Eq.15: Coefficient of total variation regularization for intrinsic function')
path_arg.add_argument('--alpha_l2_bias', type=float, default=1.0,
    help='Alpha in Eq.15: Coefficient of l2 regularization for bias field function')
path_arg.add_argument('--beta_l2_bias_log', type=float, default=1.0,
    help='Beta in Eq.15: Coefficient of l2 regularization for logarithm of bias field function')
path_arg.add_argument('--gamma1_tv_seg', type=float, default=1.0,
    help='Gamma_1 in Eq.18: Coefficient of total variation regularization for segmenting function')
path_arg.add_argument('--gamma2_tv_region', type=float, default=1.0,
    help='Gamma_2 in Eq.18: Coefficient of total variation regularization for region of interest and background')
path_arg.add_argument('--k_r1', type=int, default=10,
    help='k in Eq.24: Control parameter for the r1 regularization for GAN')

path_arg = parser.add_argument_group('Optimization Config')
path_arg.add_argument('--num_epoch', type=int, default=5,
    help='Number of epochs to train for')
path_arg.add_argument('--train_batch_size', type=int, default=64,
    help='Batch size for training')
path_arg.add_argument('--val_batch_size', type=int, default=64,
    help='Batch size for validation')
path_arg.add_argument('--test_batch_size', type=int, default=64,
    help='Batch size for testing')
path_arg.add_argument('--init_lr_intrinsic', type=float, default=1e-3,
    help='Initial learning rate value for intrinsic decomposition network')
path_arg.add_argument('--top_lr_intrinsic', type=float, default=1e-1,
    help='Top learning rate value for intrinsic decomposition network')
path_arg.add_argument('--final_lr_intrinsic', type=float, default=1e-4,
    help='Final learning rate value for intrinsic decomposition network')
path_arg.add_argument('--init_lr_seg', type=float, default=1e-3,
    help='Initial learning rate value for segmentation generator')
path_arg.add_argument('--top_lr_seg', type=float, default=1e-1,
    help='Top learning rate value for segmentation generator')
path_arg.add_argument('--final_lr_seg', type=float, default=1e-4,
    help='Final learning rate value for segmentation generator')
path_arg.add_argument('--init_lr_discri', type=float, default=1e-3,
    help='Initial learning rate value for discriminator')
path_arg.add_argument('--top_lr_discri', type=float, default=1e-1,
    help='Top learning rate value for discriminator')
path_arg.add_argument('--final_lr_discri', type=float, default=1e-4,
    help='Final learning rate value for discriminator')
path_arg.add_argument('--momentum', type=float, default=0.9,
    help='Momentum value for stochastic gradient descent')
path_arg.add_argument('--weight_decay', type=float, default=0.0005,
    help='weight decay value for stochastic gradient descent')
path_arg.add_argument('--beta1_intrinsic', type=float, default=0.5,
    help='Beta1 hyperparameter for Adam optimizers for intrinsic decomposition network')
path_arg.add_argument('--beta1_discri', type=float, default=0.5,
    help='Beta1 hyperparameter for Adam optimizers for discriminator')
path_arg.add_argument('--beta1_generator', type=float, default=0.5,
    help='Beta1 hyperparameter for Adam optimizers for segmentation generator')
path_arg.add_argument('--num_discri', type=int, default=5,
    help='Number of discriminator steps before one generator step')

def get_config():
    config = parser.parse_args()
    print('[*] Configuration')
    print(config)
    return config
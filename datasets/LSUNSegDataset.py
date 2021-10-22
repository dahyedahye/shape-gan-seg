import argparse
import os
import random
import datetime
import glob
from time import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torchvision.transforms as transforms

from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


# to load dataset written in npy
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

class LSUNSegDataset(Dataset):
    """LSUN Segmentation Dataset"""
#     def __init__(self, root, transform=None):
    def __init__(self, root_dir, transform=None, transform_gt=None):
        """
        Args:
            np_file (string): Path to the np file of input image and its ground truth.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.transform_gt = transform_gt
        time_total_start = time()

        self.file_list = [f for f in glob.glob("{}/*.npy".format(self.root_dir))]

        total_time = time() - time_total_start
        total_hours, _total_rest = divmod(total_time, 3600)
        total_mins, total_secs = divmod(_total_rest, 60)
        print('#-LSUN Data-# Total Data Path Loading Time: {}h {}m {}s'.format(total_hours, total_mins, total_secs))


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        loaded_dict = np.load(self.file_list[idx], allow_pickle=True).item()
        # occluded = torch.from_numpy(loaded_dict['data'])
        # ground_truth = torch.from_numpy(loaded_dict['masks'])
        image = loaded_dict['data']
        ground_truth = loaded_dict['masks']

        # if self.transform_occluded:
        image = self.transform(image)

        if self.transform_gt:
            ground_truth = self.transform_gt(ground_truth)

        return image, ground_truth


class LSUNSegTrainDataset(Dataset):
    """LSUN Segmentation Dataset"""
#     def __init__(self, root, transform=None):
    def __init__(self, root_dir, num_split, transform=None, transform_gt=None):
        """
        Args:
            np_file (string): Path to the np file of input image and its ground truth.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.num_split = num_split
        self.transform = transform
        self.transform_gt = transform_gt

        # # search all file name
        # all_filename_list = []
        # for ...:
        #     all_filename_list.append(...)

        time_total_start = time()
        print('#-LSUN Train Data-# Start Load LSUN Train Data')
        self.file_list_all = [f for f in glob.glob("{}/*.npy".format(self.root_dir))]
        self.file_list = self.file_list_all[:num_split]


        total_time = time() - time_total_start
        total_hours, _total_rest = divmod(total_time, 3600)
        total_mins, total_secs = divmod(_total_rest, 60)
        print('#-LSUN Train Data-# Total Data Path Loading Time: {}h {}m {}s'.format(total_hours, total_mins, total_secs))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        loaded_dict = np.load(self.file_list[idx], allow_pickle=True).item()
        # occluded = torch.from_numpy(loaded_dict['data'])
        # ground_truth = torch.from_numpy(loaded_dict['masks'])
        image = loaded_dict['data']
        ground_truth = loaded_dict['masks']

        # if self.transform_occluded:
        image = self.transform(image)

        if self.transform_gt:
            ground_truth = self.transform_gt(ground_truth)

        return image, ground_truth

class LSUNSegValDataset(Dataset):
    """LSUN Segmentation Dataset"""
#     def __init__(self, root, transform=None):
    def __init__(self, root_dir, num_split, transform=None, transform_gt=None):
        """
        Args:
            np_file (string): Path to the np file of input image and its ground truth.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.num_split = num_split
        self.transform = transform
        self.transform_gt = transform_gt

        # # search all file name
        # all_filename_list = []
        # for ...:
        #     all_filename_list.append(...)
        time_total_start = time()
        print('#-LSUN Val Data-# Start Load LSUN Val Data')

        self.file_list_all = [f for f in glob.glob("{}/*.npy".format(self.root_dir))]
        self.file_list = self.file_list_all[-num_split:]


        total_time = time() - time_total_start
        total_hours, _total_rest = divmod(total_time, 3600)
        total_mins, total_secs = divmod(_total_rest, 60)
        print('#-LSUN Val Data-# Total Data Path Loading Time: {}h {}m {}s'.format(total_hours, total_mins, total_secs))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        loaded_dict = np.load(self.file_list[idx], allow_pickle=True).item()
        # occluded = torch.from_numpy(loaded_dict['data'])
        # ground_truth = torch.from_numpy(loaded_dict['masks'])
        image = loaded_dict['data']
        ground_truth = loaded_dict['masks']

        # if self.transform_occluded:
        image = self.transform(image)

        if self.transform_gt:
            ground_truth = self.transform_gt(ground_truth)

        return image, ground_truth

class LSUNSegTestDataset(Dataset):
    """LSUN Segmentation Dataset"""
#     def __init__(self, root, transform=None):
    def __init__(self, root_dir, transform=None, transform_gt=None):
        """
        Args:
            np_file (string): Path to the np file of input image and its ground truth.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.transform_gt = transform_gt

        # # search all file name
        # all_filename_list = []
        # for ...:
        #     all_filename_list.append(...)
        time_total_start = time()
        self.file_list = [f for f in glob.glob("{}/*.npy".format(self.root_dir))]
        

        total_time = time() - time_total_start
        total_hours, _total_rest = divmod(total_time, 3600)
        total_mins, total_secs = divmod(_total_rest, 60)
        print('#-Data-# Total Data Path Loading Time: {}h {}m {}s'.format(total_hours, total_mins, total_secs))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        loaded_dict = np.load(self.file_list[idx], allow_pickle=True).item()
        # occluded = torch.from_numpy(loaded_dict['data'])
        # ground_truth = torch.from_numpy(loaded_dict['masks'])
        image = loaded_dict['data']
        ground_truth = loaded_dict['masks']

        # if self.transform_occluded:
        image = self.transform(image)

        if self.transform_gt:
            ground_truth = self.transform_gt(ground_truth)

        return image, ground_truth
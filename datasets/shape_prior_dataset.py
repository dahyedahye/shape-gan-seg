import glob
import numpy as np

import torch
from torch.utils.data import Dataset

from time import time
import random


# to load dataset written in npy
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

class PriorDataset(Dataset):
    """Prior Dataset"""
#     def __init__(self, root, transform=None):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        print('#-Prior Data-# Start Load Prior Data')
        time_total_start = time()
        self.file_list = [f for f in glob.glob("{}/*.npy".format(self.root_dir))]

        total_time = time() - time_total_start
        total_hours, _total_rest = divmod(total_time, 3600)
        total_mins, total_secs = divmod(_total_rest, 60)
        print('#-Prior Data-# Total Data Path Loading Time: {}h {}m {}s'.format(total_hours, total_mins, total_secs))



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_data = np.array(np.load(self.file_list[idx]))

        if self.transform:
            input_data = self.transform(input_data)

        return input_data

class PriorDatasetSliced(Dataset):
    """Prior Dataset Sliced"""
#     def __init__(self, root, transform=None):
    def __init__(self, root_dir, num_split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_split = num_split
        self.file_list_all = [f for f in glob.glob("{}/*.npy".format(self.root_dir))]
        random.Random(7).shuffle(self.file_list_all)
        self.file_list = self.file_list_all[:num_split]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_data = np.array(np.load(self.file_list[idx]))

        if self.transform:
            input_data = self.transform(input_data)

        return input_data
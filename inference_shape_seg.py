"""
Codes for inference using unsupervised segmentation incorporating shape prior via GAN.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.utils as vutils

# ===================================
#         Import custom codes
# ===================================
import config

from datasets import LSUNSegDataset

from metrics import metric_seg

from networks import generator
from networks import discriminator

# ===================================
#             Get config
# ===================================
train_config = config.get_config()

# config - experiment
output_dir = train_config.output_dir

# config - data
height = train_config.height
width = train_config.width
segment_data_test_dir = train_config.segment_data_test_dir

# config - networks
trained_ckpt_path = train_config.trained_ckpt_path
num_in_channel = train_config.num_in_channel
num_out_channel = train_config.num_out_channel

# config - inference environment
test_batch_size = train_config.test_batch_size
num_workers = train_config.num_workers
multi_gpu = train_config.multi_gpu
num_gpu = train_config.num_gpu
cuda_id = train_config.cuda_id

# ===============================================
#     Set Path & Files to Save Inference Result
# ===============================================
# create output directory
try:
    os.mkdir(output_dir)
    print("Directory " , output_dir,  " Created ") 
except FileExistsError:
    print("Directory " , output_dir,  " already exists")

# file path to save .txt files which contain avg iou
txt_scores_test = '{}/avg_scores_test.txt'.format(output_dir)
with open(txt_scores_test, 'w', newline='') as f:
    f.write('init score txt file for test' + os.linesep)

# ===================================
#             Load Data
# ===================================
# transforms
transform_test = transforms.Compose([
                            transforms.ToTensor()
                            ])
transform_target = transforms.Compose([
                            transforms.ToTensor()
                            ]) 

# dataset
dataset_test = LSUNSegDataset.LSUNSegTestDataset(
    segment_data_test_dir,
    transform=transform_test,
    transform_gt=transform_target)

# dataloader
loader_test = torch.utils.data.DataLoader(
    dataset = dataset_test,
    batch_size=test_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False)

# ====================================
#           Set Inference Env
# ====================================
torch.cuda.set_device(cuda_id)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if multi_gpu:
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ngpu = num_gpu # should be modified to specify multiple gpu ids to be used
elif cuda_id:
    device = torch.device(cuda_id)
    ngpu = 1
else:
    device = torch.device('cpu')
    ngpu = 0

if multi_gpu == False:
    device = torch.device(cuda_id)
    num_gpu = 1

# ===================================
#              Set Model
# ===================================
model_segment_encoder = generator.Encoder(num_in_channel, num_out_channel).to(device)
model_segment_decoder_seg = generator.DecoderSeg(num_in_channel, num_out_channel).to(device)
model_segment_decoder_region = generator.DecoderRegion(num_in_channel, num_out_channel).to(device)

# if multiple gpus are used, set dataparallel
if multi_gpu == True:
    model_segment_encoder = nn.DataParallel(model_segment_encoder, list(range(num_gpu)))
    model_segment_decoder_seg = nn.DataParallel(model_segment_decoder_seg, list(range(num_gpu)))
    model_segment_decoder_region = nn.DataParallel(model_segment_decoder_region, list(range(num_gpu)))

# load trained model parameters
trained_model = torch.load(trained_ckpt_path, map_location='cpu')
model_segment_encoder.load_state_dict(trained_model['model_segment_encoder'], strict=False)
print("[*] Trained segment_encoder model has been loaded")
model_segment_decoder_seg.load_state_dict(trained_model['model_segment_decoder_seg'], strict=False)
print("[*] Trained segment_decoder_seg model has been loaded")
model_segment_decoder_region.load_state_dict(trained_model['model_segment_decoder_region'], strict=False)
print("[*] Trained segment_decoder_region model has been loaded")

# ================================================
#                    Inference
# ================================================
t = Variable(torch.Tensor([0.5])).to(device) # set threshold value

test_scores_sum_iou = 0
num_iters_test = 0

model_segment_encoder.eval()
model_segment_decoder_seg.eval()
model_segment_decoder_region.eval()

num_img = 0

for i, data_input in enumerate(loader_test, 0):
    imgs_input = Variable(data_input[0].type(Tensor)).to(device)
    imgs_gt = Variable(data_input[1].type(Tensor)).to(device)

    # get segmentation mask
    code = model_segment_encoder(imgs_input)
    imgs_seg = model_segment_decoder_seg(code)
    imgs_roi, imgs_bg = model_segment_decoder_region(code)

    num_iters_test += 1

    imgs_seg = (imgs_seg > t).float() * 1
    imgs_gt = (imgs_gt > t).float() * 1
    # flatten input and ground truth images
    metric_iou = metric_seg.evaluate_iou(imgs_seg, imgs_gt)

    # Collect score info
    test_scores_sum_iou += metric_iou        

# plot images
num_plot_img = 64

plot_segmented = imgs_seg.cpu().detach().numpy()
plot_gt = imgs_gt.cpu().detach().numpy()
plot_input = imgs_input.cpu().detach().numpy()
plot_roi = imgs_roi.cpu().detach().numpy()
plot_bg = imgs_bg.cpu().detach().numpy()

plot_segmented = vutils.make_grid(torch.from_numpy(plot_segmented[:num_plot_img]), padding=2, pad_value=1)
plot_gt = vutils.make_grid(torch.from_numpy(plot_gt[:num_plot_img]), padding=2, pad_value=1)
plot_input = vutils.make_grid(torch.from_numpy(plot_input[:num_plot_img]), padding=2, pad_value=1)
plot_roi = vutils.make_grid(torch.from_numpy(plot_roi[:num_plot_img]), padding=2, pad_value=1)
plot_bg = vutils.make_grid(torch.from_numpy(plot_bg[:num_plot_img]), padding=2, pad_value=1)

imgs = [[plot_segmented, plot_gt, plot_input, plot_roi, plot_bg]]
imgs_list = [plot_segmented, plot_gt, plot_input, plot_roi, plot_bg] # flat the list imgs
img_names = ['segmented', 'ground truth', 'input image', 'foreground', 'background']
fig, axes = plt.subplots(len(imgs), len(imgs[0]), figsize=(24,9))
for plot_i, ax in enumerate(axes.flat):
    ax.axis("off")
    ax.set_title(img_names[plot_i])
    ax.imshow(np.transpose(imgs_list[plot_i],(1,2,0)), vmin=0.0, vmax=1.0)
    if plot_i + 1 == len(imgs_list):
        break
plt.show()
file_name = '{}/results_test'.format(output_dir)
fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
plt.clf()
plt.close()

# get avg iou
test_avg_iou = test_scores_sum_iou/num_iters_test

print('[*TEST*] [Avg IoU: {}]'.format(test_avg_iou))

with open(txt_scores_test, 'r+') as f:
    f.write('************Test Average IoU]*************' + os.linesep)
    f.write('avg test iou : {}'.format(test_avg_iou) + os.linesep)            
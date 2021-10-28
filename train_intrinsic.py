"""
Codes for unsupervised intrinsic image decomposition
Optimized based on loss driven from an image model consisting of an additive noise and a multiplicative bias
"""

import os
import csv
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data

from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt


# ===================================
#         Import custom codes
# ===================================
import config
import lr_scheduler

from datasets import LSUNSegDataset

from losses import intrinsic_loss

from networks import intrinsic_autoencoder


# ===================================
#             Get config
# ===================================
train_config = config.get_config()

# config - experiment
output_dir = train_config.output_dir
monitor_interval = train_config.monitor_interval
num_plot_img = train_config.num_plot_img

# config - data
segment_data_train_dir = train_config.segment_data_train_dir
segment_data_val_dir = train_config.segment_data_val_dir
num_train_split = train_config.num_train_split
num_val_split = train_config.num_val_split
height = train_config.height
width = train_config.width

# config - networks
trained_ckpt_path = train_config.trained_ckpt_path
num_in_channel = train_config.num_in_channel
num_out_channel = train_config.num_out_channel

# config - training env
num_workers = train_config.num_workers
multi_gpu = train_config.multi_gpu
num_gpu = train_config.num_gpu
cuda_id = train_config.cuda_id

# config - coefficient
lambda_tv_intrin = train_config.lambda_tv_intrin
alpha_l2_bias = train_config.alpha_l2_bias
beta_l2_bias_log = train_config.beta_l2_bias_log

# config - optimization
num_epoch = train_config.num_epoch
train_batch_size = train_config.train_batch_size
val_batch_size = train_config.val_batch_size
init_lr_intrinsic = train_config.init_lr_intrinsic
top_lr_intrinsic = train_config.top_lr_intrinsic
final_lr_intrinsic = train_config.final_lr_intrinsic
momentum = train_config.momentum
weight_decay = train_config.weight_decay
beta1_intrinsic = train_config.beta1_intrinsic

# ================================================
#     Set Path & Files to Save Training Result
# ================================================
# create output directory
try:
    os.mkdir(output_dir)
    print("Directory " , output_dir,  " Created ") 
except FileExistsError:
    print("Directory " , output_dir,  " already exists")

# file path to save .csv which contain metrics and losses of every iteration
csv_loss_train = '{}/loss_train.csv'.format(output_dir)
csv_loss_val = '{}/loss_val.csv'.format(output_dir)
with open(csv_loss_train, 'w', newline='') as f_train:
    writer_train = csv.writer(f_train)
    writer_train.writerow(['epoch', 'iteration', 'loss'])
with open(csv_loss_val, 'w', newline='') as f_val:
    writer_val = csv.writer(f_val)
    writer_val.writerow(['epoch', 'iteration', 'loss'])

# file path to save .txt file which contains best scores info
txt_scores_train = '{}/best_scores_train.txt'.format(output_dir)
txt_scores_val = '{}/best_scores_val.txt'.format(output_dir)
with open(txt_scores_train, 'w', newline='') as f:
    f.write('init score txt file for train' + os.linesep)
with open(txt_scores_val, 'w', newline='') as f:
    f.write('init score txt file for val' + os.linesep)

# file path to save training models by the best score
dir_save_model = '{}/trained_model_by_best_val_loss.pth'.format(output_dir)

# ===================================
#             Load Data
# ===================================
# transforms
if num_in_channel == 3:
    transform_train = transforms.Compose([
                                transforms.ToTensor()
                                ])
    transform_test = transforms.Compose([
                                transforms.ToTensor()
                                ])      
    transform_target = transforms.Compose([
                                transforms.ToTensor()
                                ])   

elif num_in_channel == 1:
    transform_train = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.ToPILImage(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                ])
    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.ToPILImage(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                ])
    transform_target = transforms.Compose([
                                transforms.ToTensor()
                                ])   

# datasets
dataset_train = LSUNSegDataset.LSUNSegTrainDataset(
    segment_data_train_dir,
    num_train_split,
    transform=transform_train,
    transform_gt=transform_target)

dataset_val = LSUNSegDataset.LSUNSegValDataset(
    segment_data_val_dir,
    num_val_split,
    transform=transform_test,
    transform_gt=transform_target)

# dataloaders
loader_train = torch.utils.data.DataLoader(
    dataset = dataset_train,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

loader_val = torch.utils.data.DataLoader(
    dataset = dataset_val,
    batch_size=val_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False)

# ===================================
#           Set Train Env
# ===================================
torch.cuda.set_device(cuda_id)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if multi_gpu == True:
    print("use multiple GPUs")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ngpu = num_gpu # should be modified to specify multiple gpu ids to be used
elif cuda_id:
    device = torch.device(cuda_id)
    ngpu = 1
else:
    device = torch.device('cpu')
    ngpu = 0

# ===================================
#              Set Model
# ===================================
# init intrin model instance
model_intrinsic = intrinsic_autoencoder.IntrinsicResidualAutoencoder(num_in_channel, num_out_channel).to(device)

# decide whether to use multiple gpus
if multi_gpu == True:
    print("use multiple GPUs")
    model_intrinsic = nn.DataParallel(model_intrinsic, list(range(num_gpu)))

# optimizers
if(init_lr_intrinsic  == final_lr_intrinsic):
    optimizer_intrin   = optim.Adam(model_intrinsic.parameters(), lr = init_lr_intrinsic, betas=(beta1_intrinsic, 0.999))
else:
    optimizer_intrin   = optim.Adam(model_intrinsic.parameters(), lr = init_lr_intrinsic, betas=(beta1_intrinsic, 0.999))
 
scheduler_intrinsic   = lr_scheduler.scheduler_learning_rate_sigmoid_double(optimizer_intrin, lr_initial=init_lr_intrinsic, lr_top=top_lr_intrinsic, lr_final=final_lr_intrinsic, numberEpoch=num_epoch, ratio=0.25, alpha=10, beta=0, epoch=-1) 

# ==================================================
#      Init variables to save training results
# ==================================================
# init variables for training info
train_losses_I = []
train_epoch_avg_intrinsic_loss = []
train_best_epoch_avg_intrinsic_loss = [0, 0]

# init variables for validation info
val_losses_I = []
val_epoch_avg_intrinsic_loss=[]
val_best_epoch_avg_intrinsic_loss = [0, 0]

# ==============================
#         Start Training
# ==============================
for epoch in range(num_epoch):
    if epoch == 0:
        print('device:', device)
    print("start train epoch{}:".format(epoch))
    scheduler_intrinsic.step(epoch)
    sum_loss_i = 0
    num_iters_intrin = 0
    # ================================================
    #                     Train
    # ================================================
    model_intrinsic.train()
    for i, data_input in enumerate(loader_train, 0):
        imgs_input = Variable(data_input[0].type(Tensor)).to(device)

        # ---------------------
        #    Train Intrinsic
        # ---------------------
        optimizer_intrin.zero_grad()
        imgs_intrin, imgs_bias = model_intrinsic(imgs_input)
        loss_i, loss_terms_i = intrinsic_loss.intrinsic_loss(imgs_input, imgs_intrin, imgs_bias, lambda_tv_intrin, alpha_l2_bias, beta_l2_bias_log)
        loss_i.backward()
        optimizer_intrin.step()
        num_iters_intrin += 1

        # Collect intrinsic loss info
        train_losses_I.append(loss_i.item())
        sum_loss_i += loss_i.item()
        with open(csv_loss_train, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, i+1, loss_i.item()])
        
        if i % 10 == 0:
            print('[*TRAIN*] [Epoch {}/{}] [Batch {}/{}] [I loss: {}]'.format(epoch+1, num_epoch, i+1, len(loader_train), loss_i.item()))

        # -----------------------------------------
        #      Plot images to monitor training
        # -----------------------------------------
        if num_iters_intrin % monitor_interval == 0:
            # plot images
            plot_input = imgs_input.cpu().detach().numpy()
            plot_intrin = imgs_intrin.cpu().detach().numpy()
            plot_output_bias = imgs_bias.cpu().detach().numpy()

            plot_input = vutils.make_grid(torch.from_numpy(plot_input[:num_plot_img]), padding=2, pad_value=1)
            plot_intrin = vutils.make_grid(torch.from_numpy(plot_intrin[:num_plot_img]), padding=2, pad_value=1)
            plot_output_bias = vutils.make_grid(torch.from_numpy(plot_output_bias[:num_plot_img]), padding=2, pad_value=1)

            imgs = [[plot_input, plot_intrin, plot_output_bias]]
            imgs_list = [plot_input, plot_intrin, plot_output_bias] # flat the list imgs
            img_names = ['input', 'output intrinsic', 'output bias']
            fig, axes = plt.subplots(len(imgs), len(imgs[0]), figsize=(18,6))
            for plot_i, ax in enumerate(axes.flat):
                ax.axis("off")
                ax.set_title(img_names[plot_i])
                ax.imshow(np.transpose(imgs_list[plot_i],(1,2,0)), vmin=0.0, vmax=1.0)
                if plot_i + 1 == len(imgs_list):
                    break
            plt.show()
            file_name = '{}/results_train'.format(output_dir)
            fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
            plt.clf()
            plt.close()

            # Plot loss & metric score curves
            curve_titles = [
                "Intrinsic Loss",
            ]
            curve_data = [[train_losses_I]]
            curve_labels = [["I"]]
            curve_xlabels = ["iterations"]
            curve_ylabels = ["Loss"]
            curve_filenames = ["lr-curve-intrin-train-iter"]
            
            for i_curve, curve_data in enumerate(curve_data):
                plt.figure(figsize=(10,5))
                plt.title(curve_titles[i_curve])
                for i_curve_data, curve_data_item in enumerate(curve_data):
                    plt.plot(curve_data_item,label=curve_labels[i_curve][i_curve_data])
                plt.xlabel(curve_xlabels[i_curve])
                plt.ylabel(curve_ylabels[i_curve])
                plt.legend()
                file_name = '{}/{}'.format(output_dir, curve_filenames[i_curve])
                plt.show()
                plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()

    # get epoch avg intrinsic loss
    avg_epoch_intrinsic_loss = sum_loss_i/num_iters_intrin
    train_epoch_avg_intrinsic_loss.append(avg_epoch_intrinsic_loss)

    # track the best epoch intrinsic loss
    if epoch == 0:
        train_best_epoch_avg_intrinsic_loss = [avg_epoch_intrinsic_loss, epoch+1]
    else:
        if train_best_epoch_avg_intrinsic_loss[0] > avg_epoch_intrinsic_loss:
            train_best_epoch_avg_intrinsic_loss = [avg_epoch_intrinsic_loss, epoch+1]
    with open(txt_scores_train, 'r+') as f:
        f.write('*************[Train Epoch Average Best Score]*************' + os.linesep)
        f.write('best intrinsic loss: {} ([epoch]{})'.format(train_best_epoch_avg_intrinsic_loss[0], train_best_epoch_avg_intrinsic_loss[1]) + os.linesep)

    # plot ratio curve
    plt.figure(figsize=(10,5))
    plt.title("Metric Score - Train (Epoch Average)")
    plt.plot(train_epoch_avg_intrinsic_loss,label="loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    file_name = '{}/loss-epoch-avg-train'.format(output_dir)
    plt.show()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()


    # ================================================
    #                    Validation
    # ================================================
    val_sum_loss_i = 0
    num_iters_val = 0
    model_intrinsic.eval()
    time_val_epoch_start = time()
    with torch.no_grad():
        for i, data_input in enumerate(loader_val, 0):
            imgs_input = Variable(data_input[0].type(Tensor)).to(device)
            imgs_gt = Variable(data_input[1].type(Tensor)).to(device)
            imgs_intrin, imgs_bias = model_intrinsic(imgs_input)
            loss_i, loss_terms_i = intrinsic_loss.intrinsic_loss(imgs_input, imgs_intrin, imgs_bias, lambda_tv_intrin, alpha_l2_bias, beta_l2_bias_log)
            num_iters_val += 1

            # Collect intrinsic loss info
            val_losses_I.append(loss_i.item())
            val_sum_loss_i += loss_i.item()

            with open(csv_loss_val, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, i+1, loss_i.item()])
                
            if i % 10 == 0:
                print('[*VAL*] [Epoch {}/{}] [Batch {}/{}] [I loss: {}]'.format(epoch+1, num_epoch, i+1, len(loader_val), loss_i.item()))

            # -----------------------------------------
            #      Plot images to monitor training
            # -----------------------------------------
            if num_iters_val % (monitor_interval//2) == 0:
                # plot images
                plot_input = imgs_input.cpu().detach().numpy()
                plot_intrin = imgs_intrin.cpu().detach().numpy()
                plot_output_bias = imgs_bias.cpu().detach().numpy()

                plot_input = vutils.make_grid(torch.from_numpy(plot_input[:num_plot_img]), padding=2, pad_value=1)
                plot_intrin = vutils.make_grid(torch.from_numpy(plot_intrin[:num_plot_img]), padding=2, pad_value=1)
                plot_output_bias = vutils.make_grid(torch.from_numpy(plot_output_bias[:num_plot_img]), padding=2, pad_value=1)

                imgs = [[plot_input, plot_intrin, plot_output_bias]]
                imgs_list = [plot_input, plot_intrin, plot_output_bias] # flat the list imgs
                img_names = ['input', 'output intrin', 'output bias']
                fig, axes = plt.subplots(len(imgs), len(imgs[0]), figsize=(18,18))
                for plot_i, ax in enumerate(axes.flat):
                    ax.axis("off")
                    ax.set_title(img_names[plot_i])
                    ax.imshow(np.transpose(imgs_list[plot_i],(1,2,0)), vmin=0.0, vmax=1.0)
                    if plot_i + 1 == len(imgs_list):
                        break
                plt.show()
                file_name = '{}/results_val'.format(output_dir)
                fig.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()

                # Plot loss & metric score curves
                curve_titles = [
                    "Intrinsic Loss",
                ]
                curve_data = [[val_losses_I]]
                curve_labels = [["I"]]
                curve_xlabels = ["iterations"]
                curve_ylabels = ["Loss"]
                curve_filenames = ["lr-curve-intrin-val-iter"]
                
                for i_curve, curve_data in enumerate(curve_data):
                    plt.figure(figsize=(10,5))
                    plt.title(curve_titles[i_curve])
                    for i_curve_data, curve_data_item in enumerate(curve_data):
                        plt.plot(curve_data_item,label=curve_labels[i_curve][i_curve_data])
                    plt.xlabel(curve_xlabels[i_curve])
                    plt.ylabel(curve_ylabels[i_curve])
                    plt.legend()
                    file_name = '{}/{}'.format(output_dir, curve_filenames[i_curve])
                    plt.show()
                    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                    plt.clf()
                    plt.close()
        avg_epoch_intrinsic_loss = val_sum_loss_i/num_iters_val
        val_epoch_avg_intrinsic_loss.append(avg_epoch_intrinsic_loss)

        # plot ratio curve
        plt.figure(figsize=(10,5))
        plt.title("Metric Score - val (Epoch Average)")
        plt.plot(val_epoch_avg_intrinsic_loss,label="loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        file_name = '{}/loss-epoch-avg-val'.format(output_dir)
        plt.show()
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()

        # ===========================================================
        #      Save Model by the Best Validation Intrinsic Loss
        # ===========================================================
        if epoch == 0:
            val_best_epoch_avg_intrinsic_loss = [avg_epoch_intrinsic_loss, epoch+1]
        else:
            if val_best_epoch_avg_intrinsic_loss[0] > avg_epoch_intrinsic_loss:
                val_best_epoch_avg_intrinsic_loss = [avg_epoch_intrinsic_loss, epoch+1]
                torch.save({
                            'epoch': epoch,
                            'model_intrinsic': model_intrinsic.state_dict(),
                            }, dir_save_model)
                print('[*] model is saved by best loss')
            with open(txt_scores_val, 'r+') as f:
                f.write('*************[Val Epoch Average Best Score]*************' + os.linesep)
                f.write('best intrinsic loss: {} ([epoch]{})'.format(val_best_epoch_avg_intrinsic_loss[0], val_best_epoch_avg_intrinsic_loss[1]) + os.linesep)
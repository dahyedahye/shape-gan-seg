"""
Codes for training unsupervised segmentation incorporating shape prior via GAN.
Optimized based on Mumford-Shah functional & GAN loss in an adversarial way.
"""

import csv
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
import lr_scheduler

from datasets import LSUNSegDataset
from datasets import shape_prior_dataset

from metrics import metric_seg
from losses import gan_r1
from losses import ms_seg_loss

from networks import intrinsic_autoencoder
from networks import generator
from networks import discriminator

# ===================================
#             Get config
# ===================================
train_config = config.get_config()

# config - experiment
output_dir = train_config.output_dir
monitor_interval = train_config.monitor_interval
num_plot_img = train_config.num_plot_img

# config - data
height = train_config.height
width = train_config.width
segment_data_train_dir = train_config.segment_data_train_dir
segment_data_val_dir = train_config.segment_data_val_dir
prior_data_dir = train_config.prior_data_dir
num_train_split = train_config.num_train_split
num_val_split = train_config.num_val_split
min_scale = train_config.min_scale

# config - networks
trained_ckpt_path = train_config.trained_ckpt_path
trained_intrinsic_ckpt_path = train_config.trained_intrinsic_ckpt_path
num_in_channel = train_config.num_in_channel
num_out_channel = train_config.num_out_channel

# config - coefficient
gamma1_tv_seg = train_config.gamma1_tv_seg
gamma2_tv_region = train_config.gamma2_tv_region
k_r1 = train_config.k_r1

# config - optimization
num_epoch = train_config.num_epoch
train_batch_size = train_config.train_batch_size
val_batch_size = train_config.val_batch_size
init_lr_seg = train_config.init_lr_seg
top_lr_seg = train_config.top_lr_seg
final_lr_seg = train_config.final_lr_seg
init_lr_discri = train_config.init_lr_discri
top_lr_discri = train_config.top_lr_discri
final_lr_discri = train_config.final_lr_discri
momentum = train_config.momentum
weight_decay = train_config.weight_decay
beta1_discri = train_config.beta1_discri
beta1_generator = train_config.beta1_generator
num_discri = train_config.num_discri

# config - training environment
num_workers = train_config.num_workers
multi_gpu = train_config.multi_gpu
num_gpu = train_config.num_gpu
cuda_id = train_config.cuda_id

# ================================================
#     Set Path & Files to Save Training Result
# ================================================
# create output directory
try:
    os.mkdir(output_dir)
    print("Directory " , output_dir,  " Created ") 
except FileExistsError:
    print("Directory " , output_dir,  " already exists")

# file path to save .csv files which contain metrics and losses of every iteration
csv_metric_train = '{}/metric_train.csv'.format(output_dir)
csv_metric_val = '{}/metric_val.csv'.format(output_dir)
csv_losses = '{}/losses.csv'.format(output_dir)
with open(csv_metric_train, 'w', newline='') as f:
    writer_train = csv.writer(f)
    writer_train.writerow(['epoch', 'iteration', 'iou'])
with open(csv_metric_val, 'w', newline='') as f:
    writer_val = csv.writer(f)
    writer_val.writerow(['epoch', 'iteration', 'iou'])
with open(csv_losses, 'w', newline='') as f:
    writer_losses = csv.writer(f)
    writer_losses.writerow(['epoch', 'iteration', 'loss discri', 'loss seg'])

# file path to save .txt files which contain best scores info
txt_scores_train = '{}/best_scores_train.txt'.format(output_dir)
txt_scores_val = '{}/best_scores_val.txt'.format(output_dir)
with open(txt_scores_train, 'w', newline='') as f:
    f.write('init score txt file for train' + os.linesep)
with open(txt_scores_val, 'w', newline='') as f:
    f.write('init score txt file for val' + os.linesep)

# ===================================
#  define functions to help training
# ===================================
# weights initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ===================================
#             Load Data
# ===================================
# transforms
transform_prior = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(size=int(height*1.40625), scale=(0.8, 2.5), ratio=(0.2, 5)),
                            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(min_scale,1.6), shear=None),
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(size=height),
                            transforms.ToTensor()
])
transform_train = transforms.Compose([
                            transforms.ToTensor()
                            ])
transform_val = transforms.Compose([
                            transforms.ToTensor()
                            ])
transform_target = transforms.Compose([
                            transforms.ToTensor()
                            ]) 

# datasets
dataset_prior = shape_prior_dataset.PriorDataset(prior_data_dir,transform=transform_prior)

dataset_train = LSUNSegDataset.LSUNSegTrainDataset(
    segment_data_train_dir,
    num_train_split,
    transform=transform_train,
    transform_gt=transform_target)

dataset_val = LSUNSegDataset.LSUNSegValDataset(
    segment_data_val_dir,
    num_val_split,
    transform=transform_val,
    transform_gt=transform_target)

# dataloaders
loader_prior = torch.utils.data.DataLoader(
    dataset = dataset_prior,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)

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

if multi_gpu:
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
model_discriminator = discriminator.Discriminator().to(device)
model_segment_encoder = generator.Encoder(num_in_channel, num_out_channel).to(device)
model_segment_decoder_seg = generator.DecoderSeg(num_in_channel, num_out_channel).to(device)
model_segment_decoder_region = generator.DecoderRegion(num_in_channel, num_out_channel).to(device)

# load trained intrinsic decomposition network
model_intrinsic = intrinsic_autoencoder.IntrinsicResidualAutoencoder(num_in_channel, num_out_channel).to(device)
trained_intrinsic_model = torch.load(trained_intrinsic_ckpt_path, map_location='cpu')
model_intrinsic.load_state_dict(trained_intrinsic_model['model_intrinsic'], strict=False)
print("[*] Trained intrinsic model has been loaded")


# if multiple gpus are used, set dataparallel
if multi_gpu == True:
    model_discriminator = nn.DataParallel(model_discriminator, list(range(num_gpu)))
    model_segment_encoder = nn.DataParallel(model_segment_encoder, list(range(num_gpu)))
    model_segment_decoder_seg = nn.DataParallel(model_segment_decoder_seg, list(range(num_gpu)))
    model_segment_decoder_region = nn.DataParallel(model_segment_decoder_region, list(range(num_gpu)))
    model_intrinsic = nn.DataParallel(model_intrinsic, list(range(num_gpu)))

# init model parameters
model_discriminator.apply(weights_init)

# optimizers
optimizer_discriminator = optim.Adam(model_discriminator.parameters(), lr = init_lr_discri, betas=(beta1_discri, 0.999))
seg_parameters = list(model_segment_encoder.parameters()) + list(model_segment_decoder_seg.parameters()) + list(model_segment_decoder_region.parameters())
optimizer_segment   = optim.Adam(seg_parameters, lr = init_lr_seg, betas=(beta1_generator, 0.999))

scheduler_discriminator   = lr_scheduler.scheduler_learning_rate_sigmoid_double(optimizer_discriminator, lr_initial=init_lr_discri, lr_top=top_lr_discri, lr_final=final_lr_discri, numberEpoch=(num_epoch), ratio=0.25, alpha=10, beta=0, epoch=-1) 
scheduler_segment = lr_scheduler.scheduler_learning_rate_sigmoid_double(optimizer_segment, lr_initial=init_lr_seg, lr_top=top_lr_seg, lr_final=final_lr_seg, numberEpoch=(num_epoch), ratio=0.25, alpha=10, beta=0, epoch=-1)

# ==================================================
#      Init variables to save training results
# ==================================================
t = Variable(torch.Tensor([0.5])).to(device) # set threshold value
losses_discri = []
losses_seg = []
train_iou = []
train_epoch_avg_iou = []
train_best_epoch_avg_iou = [0, 0]
val_iou = []
val_epoch_avg_iou = []
val_best_epoch_avg_iou = [0, 0]

# ==============================
#         Start Training
# ==============================
model_intrinsic.eval()
for epoch in range(num_epoch):
    if epoch == 0:
        print('device:', device)
    print("start train epoch{}:".format(epoch))
    scheduler_discriminator.step(epoch)
    scheduler_segment.step(epoch)
    real_label = 1
    fake_label = 0
    scores_sum_iou = 0
    num_iters_discri = 0
    num_iters_seg = 0

    # =============================
    #             Train
    # =============================
    prior_dataloader_iterator = iter(loader_prior)
    for i, data_input in enumerate(loader_train, 0):
        try:
            data_prior = next(prior_dataloader_iterator)
        except StopIteration:
            prior_dataloader_iterator = iter(loader_prior)
            data_prior = next(prior_dataloader_iterator)
        imgs_prior = Variable(data_prior.type(Tensor)).to(device)
        imgs_input = Variable(data_input[0].type(Tensor)).to(device)
        imgs_gt = Variable(data_input[1].type(Tensor)).to(device)

        # get intrinsic representation of input images
        imgs_intrin, imgs_bias = model_intrinsic(imgs_input)

        # -----------------------------
        #      Train Discriminator
        # -----------------------------
        gan_r1.toggle_grad(model_segment_encoder, False)
        gan_r1.toggle_grad(model_segment_decoder_seg, False)
        gan_r1.toggle_grad(model_segment_decoder_region, False)
        gan_r1.toggle_grad(model_discriminator, True)
        # model_segment_decoder_seg.train()
        model_discriminator.train()
        optimizer_discriminator.zero_grad()

        # train discriminator with prior data
        imgs_prior.requires_grad_()
        validity_prior = model_discriminator(imgs_prior)
        loss_discri_real = gan_r1.compute_loss(validity_prior, 1)
        loss_discri_real.backward(retain_graph=True)
        reg = k_r1 * gan_r1.compute_grad2(validity_prior, imgs_prior).mean()
        reg.backward()

        # train discriminator with generated segmentation result
        with torch.no_grad():
            code = model_segment_encoder(imgs_input)
            imgs_seg = model_segment_decoder_seg(code)
            # imgs_roi, imgs_bg = model_segment_decoder_region(code)
        imgs_seg.requires_grad_()
        validity_seg = model_discriminator(imgs_seg)
        loss_discri_fake = gan_r1.compute_loss(validity_seg, 0)
        loss_discri_fake.backward()

        # full discriminator loss
        loss_discri = loss_discri_real + reg + loss_discri_fake

        # optimization
        optimizer_discriminator.step()
        num_iters_discri += 1

        if i % num_discri == 0:
            # ----------------------------------
            #    Train Segmentation Generator
            # ----------------------------------
            model_segment_encoder.train()
            model_segment_decoder_seg.train()
            model_segment_decoder_region.train()
            gan_r1.toggle_grad(model_segment_encoder, True)
            gan_r1.toggle_grad(model_segment_decoder_seg, True)
            gan_r1.toggle_grad(model_segment_decoder_region, True)
            gan_r1.toggle_grad(model_discriminator, False)
            optimizer_segment.zero_grad()
            
            # get segmentation result
            code = model_segment_encoder(imgs_intrin)
            imgs_seg = model_segment_decoder_seg(code)
            imgs_roi, imgs_bg = model_segment_decoder_region(code)
            validity_fake = model_discriminator(imgs_seg)
            loss_seg = ms_seg_loss.mumford_shah_seg_loss(imgs_roi, imgs_bg, imgs_seg, imgs_intrin, validity_fake, gamma1_tv_seg, gamma2_tv_region)
            loss_seg.backward()
            optimizer_segment.step()
            num_iters_seg += 1

            # threshold the segmentation result
            imgs_seg = (imgs_seg > t).float() * 1
            imgs_gt = (imgs_gt > t).float() * 1
            
            # ---------------------
            #     Get Results
            # ---------------------
            # Compute metric score
            metric_iou = metric_seg.evaluate_iou(imgs_seg, imgs_gt)
            print('[*TRAIN*][Epoch {}/{}] [Batch {}/{}] [Discri loss: {}] [Seg loss: {}] [IoU: {}]'.format(epoch+1, num_epoch, i+1, len(loader_train), loss_discri.item(), loss_seg.item(), metric_iou))

            losses_discri.append(loss_discri.item())
            losses_seg.append(loss_seg.item())
            train_iou.append(metric_iou)
            scores_sum_iou += metric_iou
            with open(csv_losses, 'a', newline='') as f:
                writer_losses = csv.writer(f)
                writer_losses.writerow([epoch+1, i+1, loss_discri.item(), loss_seg.item()])
            with open(csv_metric_train, 'a', newline='') as f:
                writer_train = csv.writer(f)
                writer_train.writerow([epoch+1, i+1, metric_iou])
            
            # -----------------------------------------
            #      Plot images to monitor training
            # -----------------------------------------
            if num_iters_seg == 0 or (num_iters_seg * num_discri) % monitor_interval == 0:
                # plot images

                plot_segmented = imgs_seg.cpu().detach().numpy()
                plot_gt = imgs_gt.cpu().detach().numpy()
                plot_input = imgs_input.cpu().detach().numpy()
                plot_intrin = imgs_intrin.cpu().detach().numpy()
                plot_bias = imgs_bias.cpu().detach().numpy()
                plot_prior = imgs_prior.cpu().detach().numpy()
                plot_roi = imgs_roi.cpu().detach().numpy()
                plot_bg = imgs_bg.cpu().detach().numpy()

                plot_segmented = vutils.make_grid(torch.from_numpy(plot_segmented[:num_plot_img]), padding=2, pad_value=1)
                plot_gt = vutils.make_grid(torch.from_numpy(plot_gt[:num_plot_img]), padding=2, pad_value=1)
                plot_input = vutils.make_grid(torch.from_numpy(plot_input[:num_plot_img]), padding=2, pad_value=1)
                plot_intrin = vutils.make_grid(torch.from_numpy(plot_intrin[:num_plot_img]), padding=2, pad_value=1)
                plot_bias = vutils.make_grid(torch.from_numpy(plot_bias[:num_plot_img]), padding=2, pad_value=1)
                plot_prior = vutils.make_grid(torch.from_numpy(plot_prior[:num_plot_img]), padding=2, pad_value=1)
                plot_roi = vutils.make_grid(torch.from_numpy(plot_roi[:num_plot_img]), padding=2, pad_value=1)
                plot_bg = vutils.make_grid(torch.from_numpy(plot_bg[:num_plot_img]), padding=2, pad_value=1)

                imgs = [[plot_segmented, plot_gt, plot_input, plot_intrin], [plot_bias, plot_prior, plot_roi, plot_bg]]
                imgs_list = [plot_segmented, plot_gt, plot_input, plot_intrin, plot_bias, plot_prior, plot_roi, plot_bg] # flat the list imgs
                img_names = ['segmented', 'ground truth', 'input image', 'intrinsic', 'bias', 'prior', 'foreground', 'background']
                fig, axes = plt.subplots(len(imgs), len(imgs[0]), figsize=(24,9))
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
                    "Discriminator Loss",
                    "Segmentation Loss",
                    "IoU - Train (Iteration)"
                ]
                curve_data = [[losses_discri], [losses_seg], [train_iou]]
                curve_labels = [["loss_discriminator"], ["loss_seg"], ["iou"]]
                curve_xlabels = ["iterations", "iterations", "iterations"]
                curve_ylabels = ["loss", "loss", "score"]
                curve_filenames = ["lr-curve-discri", "lr-curve-seg", "iou-train-iter"]

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

    # get epoch avg iou
    epoch_avg_iou = scores_sum_iou/num_iters_seg
    train_epoch_avg_iou.append(epoch_avg_iou)

    # track the best epoch avg iou
    if epoch == 0:
        train_best_epoch_avg_iou = [epoch_avg_iou, epoch + 1]
    else:
        if train_best_epoch_avg_iou[0] < epoch_avg_iou:
            train_best_epoch_avg_iou = [epoch_avg_iou, epoch + 1]
    with open(txt_scores_train, 'r+') as f:
        f.write('*************[Train Epoch Average Best IoU]*************' + os.linesep)
        f.write('best iou : {} ([epoch]{})'.format(train_best_epoch_avg_iou[0], train_best_epoch_avg_iou[1]) + os.linesep)

    # plot epoch avg iou
    plt.figure(figsize=(10,5))
    plt.title("IoU - Train (Epoch Avg)")
    plt.plot(train_epoch_avg_iou,label="iou")
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend()
    file_name = '{}/iou-train-epoch-avg'.format(output_dir)
    plt.show()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()

    # ================================================
    #                    Validation
    # ================================================
    val_scores_sum_iou = 0
    num_iters_val = 0
    model_discriminator.eval()
    model_segment_encoder.eval()
    model_segment_decoder_seg.eval()
    model_segment_decoder_region.eval()
    with torch.no_grad():
        for i, data_input in enumerate(loader_val, 0):
            imgs_input = Variable(data_input[0].type(Tensor)).to(device)
            imgs_gt = Variable(data_input[1].type(Tensor)).to(device)

            # get intrinsic representation of input images
            imgs_intrin, imgs_bias = model_intrinsic(imgs_input)
            
            # get segmentation mask
            code = model_segment_encoder(imgs_intrin)
            imgs_seg = model_segment_decoder_seg(code)
            imgs_roi, imgs_bg = model_segment_decoder_region(code)

            num_iters_val += 1

            imgs_seg = (imgs_seg > t).float() * 1
            imgs_gt = (imgs_gt > t).float() * 1
            # flatten input and grount truth images
            metric_iou = metric_seg.evaluate_iou(imgs_seg, imgs_gt)

            # Collect score info
            val_iou.append(metric_iou)
            with open(csv_metric_val, 'a', newline='') as f:
                writer_val = csv.writer(f)
                writer_val.writerow([epoch+1, i+1, metric_iou])
            
            val_scores_sum_iou += metric_iou

            # track the best validation iou
            if (epoch == 0) and (i == 0):
                val_best_score_iou = [metric_iou, epoch+1, i+1]          
            else:
                if val_best_score_iou[0] < metric_iou:
                    val_best_score_iou = [metric_iou, epoch+1, i+1]

            # -----------------------------------------
            #     Plot images to monitor validation
            # -----------------------------------------
            if num_iters_val == 0 or num_iters_val % monitor_interval == 0:
                print('[*VAL*][Epoch {}/{}] [Batch {}/{}] [IoU: {}]'.format(epoch+1, num_epoch, i+1, len(loader_val), metric_iou))

                # plot metric score curve
                plt.figure(figsize=(10,5))
                plt.title("IoU - Val (Iteration)")
                plt.plot(val_iou,label="iou")
                plt.xlabel("iterations")
                plt.ylabel("score")
                plt.legend()
                file_name = '{}/iou-val-iter'.format(output_dir)
                plt.show()
                plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
                plt.clf()
                plt.close()

    # get epoch avg iou
    val_avg_epoch_iou = val_scores_sum_iou/num_iters_val
    val_epoch_avg_iou.append(val_avg_epoch_iou)
    
    # plot epoch avg iou
    plt.figure(figsize=(10,5))
    plt.title("IoU - Val (Epoch Avg)")
    plt.plot(val_epoch_avg_iou,label="iou")
    plt.xlabel("epochs")
    plt.ylabel("score")
    plt.legend()
    file_name = '{}/iou-val-epoch-avg'.format(output_dir)
    plt.show()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()

    # ======================================================
    #      Save Model by the Best Validation IoU Score
    # ======================================================
    if epoch == 0:
        val_best_epoch_avg_iou = [val_avg_epoch_iou, epoch+1]
    else:
        if val_best_epoch_avg_iou[0] < val_avg_epoch_iou:
            val_best_epoch_avg_iou = [val_avg_epoch_iou, epoch+1]
            dir_save_model = '{}/trained_model_by_best_val_iou.pth'.format(output_dir)
            torch.save({
                        'epoch': epoch,
                        'model_discriminator': model_discriminator.state_dict(),
                        'model_segment_encoder': model_segment_encoder.state_dict(),
                        'model_segment_decoder_seg': model_segment_decoder_seg.state_dict(),
                        'model_segment_decoder_region': model_segment_decoder_region.state_dict(),
                        'model_intrinsic': model_intrinsic.state_dict()
                        }, dir_save_model)
            print('[*] model is saved by best val iou score')
            with open(txt_scores_val, 'r+') as f:
                f.write('*************[Val Epoch Average Best IoU]*************' + os.linesep)
                f.write('best iou : {} ([epoch]{})'.format(val_best_epoch_avg_iou[0], val_best_epoch_avg_iou[1]) + os.linesep)            



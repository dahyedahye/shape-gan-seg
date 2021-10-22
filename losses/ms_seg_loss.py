import torch
import torch.nn.functional as F

def mumford_shah_seg_loss(imgs_roi, imgs_bg, imgs_seg, imgs_input, validity_fake, gamma1_tv_seg, gamma2_tv_region):
    mask_bg =  torch.ones_like(imgs_input) - imgs_seg
    num_pixels = imgs_input.nelement()
    seg_roi = ((torch.sum(torch.mul(((imgs_input-imgs_roi)**2), imgs_seg)))/num_pixels)
    seg_bg = ((torch.sum(torch.mul(((imgs_input-imgs_bg)**2), mask_bg)))/num_pixels)
    loss_seg = seg_roi + seg_bg

    # compute L1 total variation for seg mask by forward 
    seg_mask_grad_w = imgs_seg[:,:,:,:-1] - imgs_seg[:,:,:,1:]
    seg_mask_grad_h = imgs_seg[:,:,:-1,:] - imgs_seg[:,:,1:,:]
    l1_seg_mask_grad = (seg_mask_grad_w.norm(p=1) + seg_mask_grad_h.norm(p=1)) / seg_mask_grad_w.nelement()

    # compute L1 total variation for intrinsic image by forward 
    roi_grad_w = imgs_roi[:,:,:,:-1] - imgs_roi[:,:,:,1:]
    roi_grad_h = imgs_roi[:,:,:-1,:] - imgs_roi[:,:,1:,:]
    l1_roi_grad = (roi_grad_w.norm(p=1) + roi_grad_h.norm(p=1)) / roi_grad_w.nelement()

    # compute L1 total variation for intrinsic image by forward 
    bg_grad_w = imgs_bg[:,:,:,:-1] - imgs_bg[:,:,:,1:]
    bg_grad_h = imgs_bg[:,:,:-1,:] - imgs_bg[:,:,1:,:]
    l1_bg_grad = (bg_grad_w.norm(p=1) + bg_grad_h.norm(p=1)) / bg_grad_w.nelement()

    # compute gan loss
    targets = validity_fake.new_full(size=validity_fake.size(), fill_value=1)
    loss_gan = F.binary_cross_entropy_with_logits(validity_fake, targets)

    loss_seg_mask = (gamma1_tv_seg * l1_seg_mask_grad)
    loss_roi = (gamma2_tv_region * l1_roi_grad)
    loss_bg = (gamma2_tv_region * l1_bg_grad)
    
    return (loss_seg + loss_seg_mask + loss_roi + loss_bg + loss_gan)
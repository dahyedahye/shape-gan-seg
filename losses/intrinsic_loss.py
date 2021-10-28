import torch
import torch.nn.functional as F

def intrinsic_loss(img_input, img_intrin, img_bias, w_l1_intrin_grad, w_l2_bias_grad, w_enforcing_bias_mean):
    total_num_pixel = img_intrin.nelement()
    noise = (img_input / img_bias+0.0000000001) - img_intrin
    # noise = img_input - torch.mul(img_bias,img_intrin)
    noise_means = torch.ones_like(noise) * 0.5
    distance = F.mse_loss(noise, noise_means)

    # compute L1 total variation for intrinsic image by forward 
    intrin_grad_w = img_intrin[:,:,:,:-1] - img_intrin[:,:,:,1:]
    intrin_grad_h = img_intrin[:,:,:-1,:] - img_intrin[:,:,1:,:]
    l1_intrin_grad = (intrin_grad_w.norm(p=1) + intrin_grad_h.norm(p=1)) / total_num_pixel
    l1_intrin_grad = w_l1_intrin_grad * l1_intrin_grad

    # compute L2 norm for bias
    squared_l2_bias_grad_w = torch.sum(torch.pow(img_bias[:,:,:,:-1] - img_bias[:,:,:,1:], 2))
    squared_l2_bias_grad_h = torch.sum(torch.pow(img_bias[:,:,:-1,:] - img_bias[:,:,1:,:], 2))
    squared_l2_bias_grad = (squared_l2_bias_grad_w + squared_l2_bias_grad_h) / total_num_pixel
    squared_l2_bias_grad = w_l2_bias_grad * squared_l2_bias_grad

    # induce bias mean value
    bias_means = torch.ones_like(img_bias) * 0.5
    mse_bias = F.mse_loss(img_bias, bias_means)
    mse_bias = w_enforcing_bias_mean * mse_bias

    loss = distance + l1_intrin_grad + squared_l2_bias_grad + mse_bias
    terms = [distance, l1_intrin_grad, squared_l2_bias_grad, mse_bias]

    return loss, terms
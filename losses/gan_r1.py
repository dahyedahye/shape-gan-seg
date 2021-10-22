"""
GAN loss with r1 regularization proposed by "Which Training Methods for GANs do actually Converge? (ICML 2018)"
* The code below is the official code from "https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py"
"""

import torch.nn.functional as F
import torch.autograd as autograd

def compute_loss(output_discriminator, target):
    targets = output_discriminator.new_full(size=output_discriminator.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(output_discriminator, targets)
    return loss

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
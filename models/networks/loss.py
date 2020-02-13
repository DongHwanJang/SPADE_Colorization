"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19BN
from util.WLSFilter import wls_filter
import util.util as util


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                # TODO: modify GANloss for generator
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    # simple wrapper around self.loss() method that handles batches
    # input = list of list of tensors
    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):

            loss = 0
            # iterate through each discriminator's output
            for pred_i in input:

                # take the last feature map only, since that is the final output of the discriminator
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)

                loss += new_loss

            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, opt, gpu_ids, vgg=None):
        super(VGGLoss, self).__init__()
        if vgg is not None:
            self.vgg = vgg
        else:
            self.vgg = VGG19BN().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):

        x_vgg, y_vgg = self.vgg(x, corr_feature=False), self.vgg(y, corr_feature=False)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class SmoothnessLoss(nn.Module):  # This does not nn.Module
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def get_near_by_coords(self, h, w, height, width):
        coords_candidates = [
            (h - 1, w - 1),
            (h - 1, w),
            (h - 1, w + 1),
            (h, w - 1),
            (h, w + 1),
            (h + 1, w - 1),
            (h + 1, w),
            (h + 1, w + 1)
        ]

        coords = []

        for (y, x) in coords_candidates:
            if x < 0 or y < 0:
                continue
            if x >= width or y >= height:
                continue
            coords.append((y, x))

        return coords

    def forward(self, x):
        height, width = x.shape[2:]
        error = 0

        for c in range(2):
            wls_weight = wls_filter(x[c, :, :])
            for h in range(height):
                for w in range(width):
                    coords = self.get_near_by_coords(h, w, height, width)
                    sum = 0
                    for (y, x) in coords:
                        sum += wls_weight[y, x] * x[c, y, x]
                    diff = x[c, h, w] - sum
                    error += diff

        return error / (height * width)

# source: https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, img):
        return ( torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
        torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, fake_LAB, reference_LAB):
        return self.loss(fake_LAB, reference_LAB)


# source: https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da
class ContextualLoss(nn.Module):
    def __init__(self):
        super(ContextualLoss, self).__init__()

    def forward(self, x, y, h=0.5):
        """Computes contextual loss between x and y.

            Args:
              x: features of shape (N, C, H, W).
              y: features of shape (N, C, H, W).

            Returns:
              cx_loss = contextual loss between x and y (Eq (1) in the paper)
            """
        assert x.size() == y.size()
        N, C, H, W = x.size()

        y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

        x_centered = x - y_mu
        y_centered = y - y_mu
        x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
        y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

        # FIXME: is it best using F.interpolate for scaling down?
        x_normalized = F.interpolate(x_normalized, (H // 4, W // 4), mode='bilinear',
                                     align_corners=False)  # (N, C, H/4, W/4)
        y_normalized = F.interpolate(y_normalized, (H // 4, W // 4), mode='bilinear',
                                     align_corners=False)  # (N, C, H/4, W/4)

        # The equation at the bottom of page 6 in the paper
        # Vectorized computation of cosine similarity for each pair of x_i and y_j
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H/4 * W/4)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H/4 * W/4)

        d = 1 - torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H/4 * W/4, H/4 * W/4)
        d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H/4 * W/4, 1)

        # Eq (2)
        d_tilde = d / (d_min + 1e-5)

        # Eq(3)
        w = torch.exp((1 - d_tilde) / h)

        # Eq(4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H/4 * W/4, H/4 * W/4)

        # Eq (1)
        cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
        cx_loss = torch.mean(-torch.log(cx + 1e-5))

        return cx_loss


class IndexLoss(nn.Module):
    def __init__(self):
        super(IndexLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, corr_map, index_map):
        # index_map: B x H_key x W_key, corr_map: B x C(=N_query) x H_key x W_key
        index_loss = self.loss(corr_map, index_map)
        return index_loss

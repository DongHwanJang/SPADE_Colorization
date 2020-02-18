"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
from util.img_loader import lab_deloader
import torch.nn.functional as F
from util import img_loader
from util.fid import FID
import numpy as np

class AffinModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netA = self.initialize_networks(opt)

        if opt.use_wandb and len(opt.gpu_ids) <= 1:
            opt.wandb.watch(self.netA, log="all")

        # set loss functions
        if opt.isTrain:
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

    # parses LAB into L and AB.
    # This shouldn't make new copies. It should also handle batch cases
    def parse_LAB(self, image_LAB):
        if len(image_LAB.size()) == 4:
            image_L = image_LAB[:, 0, :, :].unsqueeze(1)
            image_A = image_LAB[:, 1, :, :].unsqueeze(1)
            image_B = image_LAB[:, 2, :, :].unsqueeze(1)
            image_AB = torch.cat([image_A, image_B], 1)
            return image_L, image_AB

        elif len(image_LAB.size()) == 3:
            # It would be a tensor whose batch size = 1.
            # Then, unsqueeze to be 4D tensor?
            return self.parse_LAB(image_LAB.unsqueeze(0))

        else:
            raise ("Pass 3D or 4D tensor")

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, is_training=True):
        # main branch
        target_LAB = data["target_LAB"]
        target_L, _ = self.parse_LAB(target_LAB)

        g_loss, out_affin = self.compute_affinnet_loss(
            target_L, target_LAB)

        return g_loss, out_affin


    def create_optimizers(self, opt):
        A_params = list(self.netA.parameters())
        A_lr = opt.lr
        optimizer_A = torch.optim.Adam(A_params, lr=A_lr)

        return optimizer_A

    def save(self, epoch):
        util.save_network(self.netA, 'A', epoch, self.opt)


    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netA = networks.define_A(opt)

        if not opt.isTrain or opt.continue_train:
            netA = util.load_network(netA, 'A', opt.which_epoch, opt)

        return netA

    def compute_affinnet_loss(self, target_L, target_LAB):

        gt_affin = self._calc_color_affin_batch(target_LAB) # BxNxN
        out_affin = self.netA(target_L)
        B, N, N = gt_affin.size()

        G_loss = self.criterionSmoothL1(
            out_affin.view(B, -1), gt_affin.view(B, -1))

        return G_loss, out_affin

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

    def _calc_color_affin(self, img_lab):

        image_a = img_lab[1, :, :]
        image_b = img_lab[2, :, :]
        img_ab = torch.cat([image_a, image_b], 0)

        # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/4
        img_ab = torch.nn.functional.interpolate(img_ab, size=(64, 64), mode='bilinear')
        img_resize = img_ab.view(2, -1)\
            .permute(1, 0) # 2 x (64*64=4096) -> 4096 x 2

        x = img_resize
        y = img_resize

        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist = torch.clamp(dist, 0.0, np.inf)
        aff_matrix = torch.exp((-1)*dist)

        # img_resize1 = img_resize.unsquueze(1).expand(64 * 64, 64 * 64, 2)
        # img_resize2 = img_resize.unsquueze(0).expand(64 * 64, 64 * 64, 2)
        #
        # aff_matrix = torch.pow(img_resize1-img_resize2, 2).sum(2)

        return aff_matrix

    def _calc_color_affin_batch(self, img_lab):
        image_a = img_lab[:, 1, :, :]
        image_b = img_lab[:, 2, :, :]
        img_ab = torch.cat([image_a, image_b], 1)

        # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/4
        img_ab = torch.nn.functional.interpolate(img_ab, size=(64, 64), mode='bilinear')
        B, C, H, W = img_ab.size()
        img_resize = img_ab.view(B, C, H*W) \
            .permute(0, 2, 1)  # BxCxHW -> BxHWxC

        aff_matrix = util.calc_affin_batch(img_resize)
        return aff_matrix
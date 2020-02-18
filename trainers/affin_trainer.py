"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.affin_model import AffinModel
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from util import util
from torch import autograd

class AffinTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.affin_model = AffinModel(opt)
        if len(opt.gpu_ids) > 0:
            self.affin_model = DataParallelWithCallback(self.affin_model,
                                                          device_ids=opt.gpu_ids)
            self.affin_model_on_one_gpu = self.affin_model.module
        else:
            self.affin_model_on_one_gpu = self.affin_model

        self.attention= None
        self.data = None

        if opt.isTrain:
            self.optimizer_A = self.affin_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def get_latest_losses(self):
        return self.g_loss

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.affin_model_on_one_gpu.save(epoch)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_A = new_lr

            else:
                new_lr_A = new_lr / 2

            for param_group in self.optimizer_A.param_groups:
                param_group['lr'] = new_lr_A

            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def run_affinnet_one_step(self, data):
        self.optimizer_A.zero_grad()

        g_loss, out_affin = self.affin_model(data)

        g_loss.backward()
        self.optimizer_A.step()
        self.g_loss = g_loss
        self.attention = out_affin.detach().cpu()
        self.data = data

    def val_affinnet_one_step(self, data):
        with torch.no_grad():

            g_loss, affin = self.affin_model(data)

            self.g_loss = g_loss
            self.attention = affin.cpu()
            self.data = data

    def get_latest_losses(self):
        self.g_loss

    ###############################
    # Visualize Affinity
    ###############################

    def get_latest_attention(self):
        """
        get attention[0]'s attentions for points satisfying certain conditions(grid,top,rand)
        :return: (N, C, H, W) where N: the number of attention maps (NOT the batch size)
        """
        points = []

        # attention => query x key
        points += self.get_grid_points()
        points += self.get_top_conf_points()
        points += self.get_random_points()

        attention_visuals = []

        for point in points:
            point = tuple(np.uint8(point))
            attention_visuals.append(self.get_point_img_on_target(point))
            attention_visuals.append(self.get_attention_visual(point))

        return torch.stack(attention_visuals)

    def get_point_img_on_target(self, point, marker_size=9):
        target_L_gray_image = self.data['target_L_gray_image'][:1]

        target_L_gray_image = util.denormalize(target_L_gray_image.clone().detach())

        x, y = point
        H = target_L_gray_image.size()[-2]
        W = target_L_gray_image.size()[-1]
        conf_H = np.sqrt(self.attention.size()[-1]) # sqrt(N)
        assert isinstance(conf_H, int)
        scale = H/conf_H

        for j in range(np.max([0, int(x*scale) - marker_size // 2]), np.min([W, int(x*scale) + marker_size // 2])):
            for k in range(np.max([0, int(y*scale) - marker_size // 2]), np.min([H, int(y*scale) + marker_size // 2])):
                target_L_gray_image[:,1,j,k] = 1  # be careful for the indexing order # assign max A value

        return target_L_gray_image.squeeze(0)

    def get_grid_points(self, n_partition = 4):
        _, H_tgt, W_tgt, _, _ = self.attention.size()
        pts_lt = []

        for i in range(1, n_partition):
            for j in range(1, n_partition):
                pts_lt.append((H_tgt/4*i, W_tgt/4*j))

        return pts_lt

    def get_random_points(self, num_pts=3):
        pts_lt = []
        H = np.sqrt(self.attention.size()[1])

        for i in range(num_pts):
            pts_lt.append((np.random.randint(H), np.random.randint(H)))

        return pts_lt

    def get_attention_visual(self, point, overlay=False, heatmap_format=False):
        # Watch out for the indexing order (y first)
        pointwise_attention = self.attention[0][point[1]][point[0]].detach().cpu() # H_key x W_key

        # pointwise_attention : [0, 1]
        pointwise_attention = pointwise_attention.unsqueeze(0).unsqueeze(0)

        pointwise_attention = F.interpolate(pointwise_attention,
                                            size=self.data["reference_LAB"].size()[2:4], mode="bilinear") # 1x1xH_refxW_ref

        reference_LAB = self.data["reference_LAB"].clone().detach()


        if overlay:
            ## FIXME
            one_reference_LAB = util.denormalize(
                reference_LAB, mean=(50, 0, 0), std=(50, 128, 128))[0]  # CxHxW -128~128

            if heatmap_format:
                # TODO need to convert LAB to RGB
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * pointwise_attention[0][0]), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255 # HxWx3  [0,1]

                atten_on_img = heatmap + np.float32(reference_LAB)
                atten_on_img += np.min(atten_on_img)
                atten_on_img = atten_on_img / np.max(atten_on_img) # atten_on_img ~ [0, 1]
                atten_on_img = torch.Tensor(atten_on_img)

            else:
                atten_on_img = pointwise_attention[0].repeat(3, 1, 1) + one_reference_LAB
                atten_on_img = atten_on_img / torch.max(atten_on_img)
        else:
            if heatmap_format:
                #FIXME
                pass
            else:
                atten_on_img = pointwise_attention[0].repeat(3, 1, 1)

        # util.normalize(atten_on_img)

        return atten_on_img

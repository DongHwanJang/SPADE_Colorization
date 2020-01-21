"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
import numpy as np
import torch.nn.functional as F
import torch


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        self.attention= None
        self.conf_map = None
        self.data = None

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated, attention, conf_map = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated
        self.attention = attention
        self.conf_map = conf_map
        self.data = data

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def get_latest_conf(self):
        return self.conf_map.detach().cpu()

    def get_latest_attention(self):
        self.attention = self.attention.detach().cpu()
        self.conf_map = self.conf_map.detach().cpu()

        points = []

        # attention => query x key
        points += self.get_grid_points()
        points += self.get_top_conf_points()
        points += self.get_random_points()

        attention_visuals = []

        for point in points:
            attention_visuals.append(self.get_attention_visual(point))

        return attention_visuals

    def get_grid_points(self, n_partition = 4):
        _, H_tgt, W_tgt, _, _ = self.attention.size()
        pts_lt = []

        for i in range(1, n_partition):
            for j in range(n_partition):
                pts_lt.append((H_tgt/4*i, W_tgt/4*j))

        return pts_lt

    def get_top_conf_points(self, num_pts=3):
        temp_tensor = self.conf_map[0].clone()
        min_value = torch.min(temp_tensor)
        H, W =temp_tensor.size()[0], temp_tensor.size()[1]
        window_sz_h = H // (num_pts+1)
        window_sz_w = W // (num_pts + 1)

        pts_lt = []

        for i in range(num_pts):
            pt = torch.argmax(temp_tensor)
            x = pt%H
            y = pt//H
            pts_lt.append((x,y))

            for j in range(np.max([0,x-window_sz_h//2]), np.min([H, x + window_sz_h//2])):
                for k in range(np.max([0, y-window_sz_w//2]), np.min([W, y + window_sz_w//2])):
                    temp_tensor[j][k]=min_value

        return pts_lt

    def get_random_points(self, num_pts=3):
        pts_lt = []

        for i in range(num_pts):
            pts_lt.append(np.random.randint(self.conf_map.size()[1]), np.random.randint(self.conf_map.size()[2]))

        return pts_lt

    def get_attention_visual(self, point):
        pointwise_attention = self.attention[0][point[0]][point[1]] # H_key x W_key

        F.interpolate(pointwise_attention, size=(self.sh, self.sw))

        reference_LAB = self.data["reference_LAB"]
        target_LAB = self.data["target_LAB"]

        # TODO




    def get_latest_image(self):
        pass

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

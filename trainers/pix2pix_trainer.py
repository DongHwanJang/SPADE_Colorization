"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.pix2pix_model import Pix2PixModel
import numpy as np
import torch.nn.functional as F
import torch
import cv2
from util import util


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
        self.attention = attention.detach().cpu()
        self.conf_map = conf_map.detach().cpu()
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

    def get_latest_conf_map(self):
        # return self.conf_map.detach().cpu()
        return self.conf_map

    def get_warped_ref_img(self):
        ref_LAB = self.data["reference_LAB"][0].clone()  # 3xHxW
        B, H_query, W_query, H_key, W_key = self.attention.size()

        ref_LAB = F.interpolate(
            ref_LAB.unsqueeze(0), size=(H_key, W_key)) # 1 x 3 x H_key x W_key
        ref_LAB=ref_LAB.squeeze(0) # 3xH_keyxW_key
        ref_LAB = ref_LAB.view(3, -1) # 3 x N_key

        attention = self.attention[0].view(H_query, W_query, -1) # H_query x W_query x N_key
        attention = attention.view(-1, H_key*W_key) # N_query x N_key
        attention = attention.permute(1, 0) # N_key x N_query

        warped_img = torch.mm(ref_LAB, attention).view(3, H_query, W_query) # 3 x N_query

        return warped_img


    def get_latest_attention(self):
        """
        get attention[0]'s attentions for points satisfying certain conditions(grid,top,rand)
        :return: (N, C, H, W) where N: the number of attention maps
        """
        self.attention = self.attention.detach().cpu()
        self.conf_map = self.conf_map.detach().cpu()

        points = []

        # attention => query x key
        points += self.get_grid_points()
        points += self.get_top_conf_points()
        points += self.get_random_points()

        attention_visuals = []

        for point in points:
            point = tuple(np.uint8(point))
            attention_visuals.append(self.get_attention_visual(point))

        return torch.stack(attention_visuals)

    def get_grid_points(self, n_partition = 4):
        _, H_tgt, W_tgt, _, _ = self.attention.size()
        pts_lt = []

        for i in range(1, n_partition):
            for j in range(1, n_partition):
                pts_lt.append((H_tgt/4*i, W_tgt/4*j))

        return pts_lt

    def get_top_conf_points(self, num_pts=3):
        temp_tensor = self.conf_map[0][0].clone()
        min_value = torch.min(temp_tensor)
        H, W =temp_tensor.size()[0], temp_tensor.size()[1]
        window_sz_h = H // (num_pts+1)
        window_sz_w = W // (num_pts + 1)

        pts_lt = []

        for i in range(num_pts):
            pt = torch.argmax(temp_tensor)
            x = np.uint8(pt%H)
            y = np.uint8(pt//H)
            pts_lt.append((x,y))

            for j in range(np.max([0,x-window_sz_h//2]), np.min([H, x + window_sz_h//2])):
                for k in range(np.max([0, y-window_sz_w//2]), np.min([W, y + window_sz_w//2])):
                    temp_tensor[k][j]=min_value # becareful for the indexing order

        return pts_lt

    def get_random_points(self, num_pts=3):
        pts_lt = []

        for i in range(num_pts):
            pts_lt.append((np.random.randint(self.conf_map.size()[2]), np.random.randint(self.conf_map.size()[3])))

        return pts_lt

    def get_attention_visual(self, point, heatmap_format=False):
        # Watch out for indexing order (y first)
        pointwise_attention = self.attention[0][point[1]][point[0]].detach().cpu() # H_key x W_key

        # pointwise_attention : [0, 1]
        pointwise_attention = pointwise_attention.unsqueeze(0).unsqueeze(0)

        pointwise_attention = F.interpolate(pointwise_attention,
                                            size=self.data["reference_LAB"].size()[2:4]) # 1x1xH_keyxW_key

        reference_LAB = self.data["reference_LAB"].clone()
        one_reference_LAB = util.denormalize(reference_LAB)[0]  # CxHxW

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
            atten_on_img = pointwise_attention[0].expand(3,-1,-1) + one_reference_LAB
            atten_on_img = atten_on_img / torch.max(atten_on_img)

        util.normalize(atten_on_img)

        return atten_on_img

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

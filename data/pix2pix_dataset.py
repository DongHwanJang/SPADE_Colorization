"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from util.img_loader import lab_loader, rgb_loader, rgb_pil2l_as_rgb, rgb_pil2lab_tensor
import os
import numpy as np
from torchvision.transforms import functional as F
from util.subnet_train_helper import get_subnet_images, create_warpped_image
import torch

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.target_ref_dict = self.get_paths(opt)
        self.target_paths = list(self.target_ref_dict.keys())[:opt.max_dataset_size]
        self.dataset_size = len(self.target_paths)
        self.top_n_reference = opt.top_n_reference
        self.train_subnet_only = opt.train_subnet_only
        if not self.train_subnet_only:
            self.train_subnet = opt.train_subnet
            if self.train_subnet:
                self.train_subnet_period = opt.train_subnet_period

    # Must be over written.
    # return type: {target_path: [top1_reference_path, top2_reference_path, ...], ... }
    def get_paths(self, opt):
        target_ref_dict = {}

        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return target_ref_dict

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # params = get_params(self.opt, label.size)
        params = get_params(self.opt, (256, 256))  # FIXME: what is label.size?

        # input image (real images)
        target_path = self.target_paths[index]

        # randomly chose from top-n nearest reference
        similarity = np.random.choice(range(self.top_n_reference), 1)[0] + 1  # top-n starts from 1 (not 0)
        reference_path = target_path

        # target_LAB = lab_loader(self.opt, target_path, is_ref=False)
        # reference_LAB = lab_loader(self.opt, reference_path, is_ref=True)
        target_rgb_pil = rgb_loader(self.opt, target_path)
        reference_rgb_pil = rgb_loader(self.opt, reference_path)

        transform_image_rgb = get_transform(self.opt, params, normalize=False, toTensor=False)
        # transform_image_LAB = get_transform(self.opt, params, normalize=False, toTensor=False)

        target_rgb_pil = transform_image_rgb(target_rgb_pil)
        reference_rgb_pil = transform_image_rgb(reference_rgb_pil)

        ####### LAB
        target_lab = rgb_pil2lab_tensor(target_rgb_pil)
        reference_lab = rgb_pil2lab_tensor(reference_rgb_pil)

        ####### L in RGB
        target_L_gray_image = rgb_pil2l_as_rgb(target_rgb_pil, need_Tensor=True)
        reference_L_gray_image = rgb_pil2l_as_rgb(reference_rgb_pil, need_Tensor=True)

        ####### RGB
        target_rgb = F.normalize(F.to_tensor(target_rgb_pil), mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        reference_rgb = F.normalize(F.to_tensor(reference_rgb_pil), mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

        ####### subnet reconstruction loss
        subnet_target = None
        subnet_ref = None
        subnet_target_lab = None
        subnet_ref_lab = None
        subnet_target_L_gray_image = None
        subnet_ref_L_gray_image = None
        subnet_warped_LAB_gt_resized = None
        subnet_index_gt_resized = None

        train_subnet = False
        if self.train_subnet_only or (self.train_subnet and self.train_subnet_period % index == 0):
            self.opt.crop_to_target = True  # FIXME
            self.opt.flip_to_target = True  # FIXME

            (subnet_ref, ref_warp), (subnet_target, target_gt), (index_image, index_image_gt) =\
                get_subnet_images(self.opt, target_rgb_pil, self.opt.subnet_crop_size // 4)

            subnet_warped_RGB_gt_resized = create_warpped_image(index_image_gt, ref_warp, target_gt)
            subnet_warped_LAB_gt_resized = rgb_pil2lab_tensor(subnet_warped_RGB_gt_resized)

            subnet_target_lab = rgb_pil2lab_tensor(subnet_target)
            subnet_ref_lab = rgb_pil2lab_tensor(subnet_ref)

            subnet_target_L_gray_image = rgb_pil2l_as_rgb(subnet_target, need_Tensor=True)
            subnet_ref_L_gray_image = rgb_pil2l_as_rgb(subnet_ref, need_Tensor=True)

            index_gt_tensor = torch.from_numpy(np.array(index_image_gt).astype(np.int64))  # H x W x C
            subnet_index_gt_resized = index_gt_tensor[:, :, 1] * self.opt.subnet_crop_size +\
                                      index_gt_tensor[:, :, 0]  # H x W

            train_subnet = True

        input_dict = {'label': target_path,
                      'target_image': target_rgb,
                      'reference_image': reference_rgb,
                      'target_LAB': target_lab,
                      'reference_LAB': reference_lab,
                      'target_L_gray_image': target_L_gray_image,
                      'reference_L_gray_image': reference_L_gray_image,
                      'similarity': similarity,

                      # TODO : avoid duplicated dict
                      # 'subnet_target_image': subnet_target,
                      # 'subnet_ref_image': subnet_ref,
                      'subnet_target_LAB': subnet_target_lab,
                      'subnet_ref_LAB': subnet_ref_lab,
                      'subnet_target_L_gray_image': subnet_target_L_gray_image,
                      'subnet_ref_L_gray_image': subnet_ref_L_gray_image,

                      "subnet_warped_LAB_gt_resized": subnet_warped_LAB_gt_resized,
                      "subnet_index_gt_resized": subnet_index_gt_resized,
                      "is_train_subnet": train_subnet}

        return input_dict

    def __len__(self):
        return self.dataset_size






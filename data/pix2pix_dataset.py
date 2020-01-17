"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
# import util.util as util
from util.img_loader import pil_loader, rgb_loader
import os
import numpy

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
        similarity = numpy.random.choice(range(self.top_n_reference), 1)[0] + 1  # top-n starts from 1 (not 0)
        reference_path = self.target_ref_dict[target_path][similarity]

        target_LAB = pil_loader(self.opt, target_path, is_ref=False)
        reference_LAB = pil_loader(self.opt, reference_path, is_ref=True)
        target_rgb = rgb_loader(self.opt, target_path, is_ref=False)

        transform_image = get_transform(self.opt, params)

        target_LAB = transform_image(target_LAB)
        reference_LAB = transform_image(reference_LAB)
        target_rgb = transform_image(target_rgb)

        input_dict = {'label': target_path,
                      'image': target_rgb,
                      'target_LAB': target_LAB,
                      'reference_LAB': reference_LAB,
                      "similarity": similarity,
                      'is_reconstructing': False}

        return input_dict


    def __len__(self):
        return self.dataset_size

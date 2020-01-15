"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
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
        self.target_paths = self.target_ref_dict.keys()[:opt.max_dataset_size]
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
        params = get_params(self.opt, label.size)

        # input image (real images)
        target_path = self.target_paths[index]

        # randomly chose from top-n nearest reference
        similarity = numpy.random.choice(range(self.top_n_reference), 1)
        reference_path = self.target_ref_dict[target_path][similarity]

        target_LAB = Image.open(target_path).convert('LAB')


        reference_LAB = Image.open(reference_path).convert('LAB')

        transform_image = get_transform(self.opt, params)

        target_LAB = transform_image(target_LAB)
        reference_LAB = transform_image(reference_LAB)

        input_dict = {'target_LAB': target_LAB,
                      'reference_LAB': reference_LAB,
                      "similarity": similarity,
                      'is_reconstructing': False}

        return input_dict


    def __len__(self):
        return self.dataset_size

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class ImagenetDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataroot='/DATA1/hksong/imagenet/')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        # load_size = 286 if is_train else 256
        load_size = 256 if is_train else 256  # FIXME : need to concern about input size
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        # root = opt.dataroot  # TODO: unify dataroot to one directory
        root = "./pair_img/"
        phase = 'val' if opt.phase == 'test' else opt.phase

        pair_list = os.path.join(root, '%s.txt' % phase)
        pair_data = dict()

        with open(pair_list, mode='r') as f:
            data = [x.strip().split(" ") for x in f.readlines()]

        for line in data:
            ref_top_n = dict()
            for ref_score, (ref_name, ref_similarity) in enumerate(zip(line[1::2], line[2::2])):
                ref_top_n[ref_score + 1] = ref_name  # ref_score(top-n) needs to count from 1 (not 0)
                # TODO: use similarity (cosine distance) score for test result
                # ref_top_n[ref_score] = [ref_name, ref_similarity]
            pair_data[line[0]] = ref_top_n

        return pair_data

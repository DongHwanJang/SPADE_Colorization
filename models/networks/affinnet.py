"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
from models.networks.architecture import VGGFeatureExtractor
from util import util
from models.networks.base_network import BaseNetwork

class AffinNet(BaseNetwork):
    def __init__(self, opt):
        super(AffinNet, self).__init__()
        self.opt = opt

        # create vgg Model
        self.vgg_feature_extracter = VGGFeatureExtractor(opt)

    def forward(self, tgt_l):

        tgt_feature = self.vgg_feature_extracter(tgt_l, input_type='value', l_with_ab=False)

        return self._calc_aff(tgt_feature)

    def _calc_aff(self, tgt_feature):
        # tgt_feature : BxCxHxW
        B, C, H, W = tgt_feature.size()
        tgt_feature_BNC = tgt_feature.view(B, C, -1)\
            .permute(0, 2, 1) # BxCxN -> BxNxC

        if self.opt.no_radius:
            aff_matrix = util.calc_affin_batch(tgt_feature_BNC)  # BxNxN
        else:
            aff_matrix = util.calc_affin_batch_new(tgt_feature)  # BxMxN'

        return aff_matrix




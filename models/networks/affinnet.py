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
        ## FIXME Change input_type: None for the ablation
        tgt_feature = self.vgg_feature_extracter(tgt_l, l_with_ab=False, input_type='value')

        return self._calc_aff(tgt_feature)

    def _calc_aff(self, tgt_feature):
        # tgt_feature : BxHxWxC
        B, C, H, W = tgt_feature.size()
        tgt_feature = tgt_feature.view(B, C, -1)\
            .permute(0, 2, 1) # BxCxN -> BxNxC

        aff_matrix = util.calc_affin_batch(tgt_feature) # BxNxN

        return aff_matrix




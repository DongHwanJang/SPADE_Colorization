"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE
import util.util as util
import numpy as np
import PIL.Image as Image


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, conf_map):
        x_s = self.shortcut(x, seg, conf_map)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg, conf_map)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, conf_map)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg, conf_map):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, conf_map))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            norm_layer,
            activation,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            norm_layer,
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19BN(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19BN, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19_bn(pretrained=True).features
        self.slice1_corr = nn.Sequential()
        self.slice2_corr = nn.Sequential()
        self.slice3_corr = nn.Sequential()
        self.slice4_corr = nn.Sequential()
        self.slice5_corr = nn.Sequential()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(13):
            self.slice1_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 20):
            self.slice2_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 33):
            self.slice3_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(33, 46):
            self.slice4_corr.add_module(str(x), vgg_pretrained_features[x])

        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 10):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 30):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(30, 43):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, corr_feature=True):
        if corr_feature:
            h_relu1 = self.slice1_corr(x)
            h_relu2 = self.slice2_corr(h_relu1)
            h_relu3 = self.slice3_corr(h_relu2)
            h_relu4 = self.slice4_corr(h_relu3)

            out = [h_relu1, h_relu2, h_relu3, h_relu4]

        else:
            h_relu1 = self.slice1(x)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)

            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out


class VGG19BN_L(nn.Module):
    def __init__(self, checkpoint_dir, requires_grad=False):
        super(VGG19BN_L, self).__init__()
        vgg_pretrained_features = torch.load(checkpoint_dir).features
        self.slice1_corr = nn.Sequential()
        self.slice2_corr = nn.Sequential()
        self.slice3_corr = nn.Sequential()
        self.slice4_corr = nn.Sequential()
        self.slice5_corr = nn.Sequential()

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(13):
            self.slice1_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 20):
            self.slice2_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 33):
            self.slice3_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(33, 46):
            self.slice4_corr.add_module(str(x), vgg_pretrained_features[x])

        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 10):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 30):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(30, 43):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, corr_feature=True):
        if corr_feature:
            h_relu1 = self.slice1_corr(x)
            h_relu2 = self.slice2_corr(h_relu1)
            h_relu3 = self.slice3_corr(h_relu2)
            h_relu4 = self.slice4_corr(h_relu3)

            out = [h_relu1, h_relu2, h_relu3, h_relu4]

        else:
            h_relu1 = self.slice1(x)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)

            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out


class VGGFeatureExtractor(nn.Module):
    def __init__(self, opt):
        super(VGGFeatureExtractor, self).__init__()

        # create vgg Model
        self.opt = opt
        checkpoint_dir = "models/networks/checkpoint.pth.tar"
        self.vgg_l_as_rgb = VGG19BN_L(checkpoint_dir).cuda()
        self.vgg_lab_as_rgb = VGG19BN().cuda()

        # create conv layers
        self.conv_2_2_0 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_2_2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv_3_2_0 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_3_2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_4_2_0 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_4_2_1 = nn.ConvTranspose2d(256, 256,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.conv_5_2_0 = nn.ConvTranspose2d(512, 256,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.conv_5_2_1 = nn.ConvTranspose2d(256, 256,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)

        self.conv_2_2_0_lab = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_2_2_1_lab = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv_3_2_0_lab = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_3_2_1_lab = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_4_2_0_lab = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_4_2_1_lab = nn.ConvTranspose2d(256, 256,
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1)
        self.conv_5_2_0_lab = nn.ConvTranspose2d(512, 256,
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1)
        self.conv_5_2_1_lab = nn.ConvTranspose2d(256, 256,
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_2_2_0 = spectral_norm(self.conv_2_2_0)
            self.conv_2_2_1 = spectral_norm(self.conv_2_2_1)
            self.conv_3_2_0 = spectral_norm(self.conv_3_2_0)
            self.conv_3_2_1 = spectral_norm(self.conv_3_2_1)
            self.conv_4_2_0 = spectral_norm(self.conv_4_2_0)
            self.conv_4_2_1 = spectral_norm(self.conv_4_2_1)
            self.conv_5_2_0 = spectral_norm(self.conv_5_2_0)
            self.conv_5_2_1 = spectral_norm(self.conv_5_2_1)

            self.conv_2_2_0_lab = spectral_norm(self.conv_2_2_0_lab)
            self.conv_2_2_1_lab = spectral_norm(self.conv_2_2_1_lab)
            self.conv_3_2_0_lab = spectral_norm(self.conv_3_2_0_lab)
            self.conv_3_2_1_lab = spectral_norm(self.conv_3_2_1_lab)
            self.conv_4_2_0_lab = spectral_norm(self.conv_4_2_0_lab)
            self.conv_4_2_1_lab = spectral_norm(self.conv_4_2_1_lab)
            self.conv_5_2_0_lab = spectral_norm(self.conv_5_2_0_lab)
            self.conv_5_2_1_lab = spectral_norm(self.conv_5_2_1_lab)

        self.conv_concate = nn.Conv2d(256*4, 256, kernel_size=1)
        self.conv_concate_lab = nn.Conv2d(256 * 4, 256, kernel_size=1)

        self.resblock_0 = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_1 = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_2 = ResnetBlock(256, nn.InstanceNorm2d(256))

        self.resblock_0_ref = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_1_ref = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_2_ref = ResnetBlock(256, nn.InstanceNorm2d(256))

        self.resblock_0_val = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_1_val = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_2_val = ResnetBlock(256, nn.InstanceNorm2d(256))


    def forward(self, x, input_type, l_with_ab=False):

        if l_with_ab:
            vgg_feature = self.vgg_lab_as_rgb(x, corr_feature=True)
            vgg_feature[0] = self.conv_2_2_1_lab(self.actvn(self.conv_2_2_0_lab(vgg_feature[0])))
            vgg_feature[1] = self.conv_3_2_1_lab(self.actvn(self.conv_3_2_0_lab(vgg_feature[1])))
            vgg_feature[2] = self.conv_4_2_1_lab(self.actvn(self.conv_4_2_0_lab(vgg_feature[2])))
            vgg_feature[3] = self.conv_5_2_1_lab(self.actvn(self.conv_5_2_0_lab(vgg_feature[3])))

            x = torch.cat(vgg_feature, dim=1)
            x = self.conv_concate_lab(x)

        else:
            vgg_feature = self.vgg_l_as_rgb(x, corr_feature=True)
            vgg_feature[0] = self.conv_2_2_1(self.actvn(self.conv_2_2_0(vgg_feature[0])))
            vgg_feature[1] = self.conv_3_2_1(self.actvn(self.conv_3_2_0(vgg_feature[1])))
            vgg_feature[2] = self.conv_4_2_1(self.actvn(self.conv_4_2_0(vgg_feature[2])))
            vgg_feature[3] = self.conv_5_2_1(self.actvn(self.conv_5_2_0(vgg_feature[3])))

            x = torch.cat(vgg_feature, dim=1)
            x = self.conv_concate(x)

        if input_type == 'target' or (input_type == 'reference' and not l_with_ab):
            x = self.resblock_0(x)
            x = self.resblock_1(x)
            x = self.resblock_2(x)  # [B, 256, H/4, W/4]

        elif input_type == 'reference':
            x = self.resblock_0_ref(x)
            x = self.resblock_1_ref(x)
            x = self.resblock_2_ref(x)  # [B, 256, H/4, W/4]

        elif input_type == 'value':
            x = self.resblock_0_val(x)
            x = self.resblock_1_val(x)
            x = self.resblock_2_val(x)  # [B, 256, H/4, W/4]

        return x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class NonLocalBlock(nn.Module):
    def __init__(self, opt, in_dim, subnet_only=False):
        super(NonLocalBlock, self).__init__()

        self.register_buffer('tau', torch.FloatTensor([0.01]))
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        self.subnet_only = subnet_only

        if not self.subnet_only:
            self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

            self.use_gamma = opt.use_gamma
            if self.use_gamma:
                self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))



    """
    key = ref_feature
    query = tgt_feature
    value = ref_value (currently equal to ref_feature)
    """
    def forward(self, key, query, value=None, unit_mult=True, subnet_only=False):
        # B: mini batches, C: channels, W_key: width, H_key: height
        B, C_key, H_key, W_key = key.shape
        B, C_query, H_query, W_query = query.shape

        # B x C x H x W -> B x C x (H*W) -> B x N x C
        proj_query = self.query_conv(query).view(B, -1, W_query * H_query).permute(0, 2, 1)

        # B x C x H x W -> B X C x (W_key*H_key)
        proj_key = self.key_conv(key).view(B, -1, W_key * H_key)

        if unit_mult:
            proj_query = proj_query - torch.mean(proj_query, dim=1, keepdim=True)
            # F.normalize is safer because it handles the case when norm is closer to zero
            proj_query = F.normalize(proj_query, dim=2)
            proj_key = proj_key - torch.mean(proj_key, dim=2, keepdim=True)
            proj_key = F.normalize(proj_key, dim=1)

        corr_map = torch.bmm(proj_query, proj_key)  # transpose check | B x N_query x N_key
        conf_map = torch.max(corr_map, dim=2)[0]  # B x N_query
        conf_map = conf_map.view(-1, H_query, W_query).unsqueeze(1)

        conf_argmax = torch.max(corr_map, dim=2)[1]  # B x N_query (argmax)  # TODO: Not used now
        attention = self.softmax(corr_map / self.tau)  # B x (N_query) x (N_key)
        attention = attention.view(B, H_query, W_query, H_key, W_key)

        if subnet_only:
            # attention: B x H_query x W_query x H_key x W_key
            # corr_map: B x N_query x N_key
            corr_map = corr_map.view(B, H_query * W_query, H_key, W_key)  # B x N_query x H_key x W_key
            return attention, corr_map

        else:

            _, C_value, _, _ = value.shape

            proj_value = self.value_conv(value).view(B, -1, W_key * H_key)  # B X 256 X N_key

            out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x 256 x N_query
            out = out.view(B, C_value, H_query, W_query)

            if self.use_gamma:
                out = self.gamma * out + value

            return attention, conf_map, out

class CorrSubnet(nn.Module):
    def __init__(self, opt):
        super(CorrSubnet, self).__init__()

        # create vgg Model
        self.vgg_feature_extracter = VGGFeatureExtractor(opt)

        self.non_local_blk = NonLocalBlock(opt, 256)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, tgt, ref_rgb, ref_l=None, subnet_only=False):
        tgt_feature = self.vgg_feature_extracter(tgt, l_with_ab=False, input_type='target')
        if ref_l is not None:
            ref_feature = self.vgg_feature_extracter(ref_l, l_with_ab=False, input_type='reference')
        else:
            ref_feature = self.vgg_feature_extracter(ref_rgb, l_with_ab=True, input_type='reference')

        if subnet_only:
            attention, corr_map = self.non_local_blk(ref_feature, tgt_feature, subnet_only=subnet_only)
            return attention, corr_map
        else:
            ref_value = self.vgg_feature_extracter(ref_rgb, l_with_ab=True, input_type='value')
            attention, conf_map, out = self.non_local_blk(ref_feature, tgt_feature, value=ref_value, subnet_only=subnet_only)
            return attention, conf_map, out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
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
class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
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

        for x in range(9):
            self.slice1_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice2_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):
            self.slice3_corr.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):
            self.slice4_corr.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, corr_feature=True):
        if corr_feature:
            h_relu1 = self.slice1_corr(X)
            h_relu2 = self.slice2_corr(h_relu1)
            h_relu3 = self.slice3_corr(h_relu2)
            h_relu4 = self.slice4_corr(h_relu3)

            out = [h_relu1, h_relu2, h_relu3, h_relu4]

        else:
            h_relu1 = self.slice1(X)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)

            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out


class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = nn.parameter.Parameter(torch.Tensor(channels))
        self.bias = nn.parameter.Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
        nB = x.size(0)
        nC = x.size(1)
        nH = x.size(2)
        nW = x.size(3)
        x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + \
            self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x


# Ported from caffemodel. The original model is made from the author of 'image colorization'.
# Luminance-only feature extraction is available for this model.
# weight can get from the link 'https://www.dropbox.com/s/smfuhqremoxc0ro/gray_vgg19_bn.pth?dl=0'
class Vgg19BN(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19BN, self).__init__()

        self.layers = nn.ModuleList([
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.9, affine=False),
            Scale(channels=3),

            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False),
            nn.ReLU(inplace=True)
        ])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, corr_feature=True):
        features = []
        # pretend that it is used for correlation loss
        if corr_feature:
            for idx in range(15):
                x = self.layers[idx](x)
            features.append(x)  # relu2_2
            for idx in range(15, 22):
                x = self.layers[idx](x)
            features.append(x)  # relu3_2
            for idx in range(22, 35):
                x = self.layers[idx](x)
            features.append(x)  # relu4_2
            for idx in range(35, 48):
                x = self.layers[idx](x)
            features.append(x)  # relu5_2

        else:
            for idx in range(5):
                x = self.layers[idx](x)
            features.append(x)

            for idx in range(5, 12):
                x = self.layers[idx](x)
            features.append(x)

            for idx in range(12, 19):
                x = self.layers[idx](x)
            features.append(x)

            for idx in range(19, 32):
                x = self.layers[idx](x)
            features.append(x)

            for idx in range(32, 45):
                x = self.layers[idx](x)
            features.append(x)

        return features


class VGGFeatureExtractor(nn.Module):
    def __init__(self, opt):
        super(VGGFeatureExtractor, self).__init__()

        # create vgg Model
        self.opt = opt
        if self.opt.ref_type == 'l' or self.opt.ref_type == 'ab' or self.opt.ref_type == 'lab':
            self.vgg = Vgg19BN().cuda().eval()
            self.vgg.load_state_dict(torch.load(util.find_pretrained_weight(opt.weight_root, opt=opt)))
        else:
            self.vgg = VGG19().cuda()

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

        self.conv_concate = nn.Conv2d(256*4, 256, kernel_size=1)
        self.resblock_0 = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_1 = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_2 = ResnetBlock(256, nn.InstanceNorm2d(256))

        self.resblock_value_0 = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_value_1 = ResnetBlock(256, nn.InstanceNorm2d(256))
        self.resblock_value_2 = ResnetBlock(256, nn.InstanceNorm2d(256))


    def forward(self, x, isValue=False, is_ref=True):

        if is_ref:
            x = x[:, 0, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
        else:
            if self.opt.ref_type == 'l' and x.size()[1] == 1:
                x = x.expand(-1, 3, -1, -1)


        vgg_feature = self.vgg(x, corr_feature=True)
        vgg_feature[0] = self.conv_2_2_1(self.actvn(self.conv_2_2_0(vgg_feature[0])))
        vgg_feature[1] = self.conv_3_2_1(self.actvn(self.conv_3_2_0(vgg_feature[1])))
        vgg_feature[2] = self.conv_4_2_1(self.actvn(self.conv_4_2_0(vgg_feature[2])))
        vgg_feature[3] = self.conv_5_2_1(self.actvn(self.conv_5_2_0(vgg_feature[3])))

        x = torch.cat(vgg_feature, dim=1)
        x = self.conv_concate(x)

        if not isValue:
            x = self.resblock_0(x)
            x = self.resblock_1(x)
            x = self.resblock_2(x)  # [B, 256, H/4, W/4]
        else:
            x = self.resblock_value_0(x)
            x = self.resblock_value_1(x)
            x = self.resblock_value_2(x)  # [B, 256, H/4, W/4]

        return x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class NonLocalBlock(nn.Module):
    def __init__(self, in_dim):
        super(NonLocalBlock, self).__init__()

        self.register_buffer('tau', torch.FloatTensor([0.01]))
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, key, query, value, unit_mult=True):
        # B: mini batches, C: channels, W: width, H: height
        B, C, H, W = key.shape
        _, C_value, _, _ = value.shape
        proj_query = self.query_conv(query).view(B, -1, W * H).permute(0, 2, 1)  # B X CX(N) -> B x N x C
        proj_key = self.key_conv(key).view(B, -1, W * H)  # B X C x (*W*H)
        if unit_mult:
            proj_query = proj_query-torch.mean(proj_query, dim=2, keepdim=True)
            proj_query = proj_query/torch.norm(proj_query, dim=2, keepdim=True)
            proj_key = proj_key - torch.mean(proj_key, dim=1, keepdim=True)
            proj_key = proj_key / torch.norm(proj_key, dim=1, keepdim=True)

        corr_map = torch.bmm(proj_query, proj_key)  # transpose check
        conf_map = torch.max(corr_map, dim=2)[0]
        conf_map = conf_map.view(-1, H, W).unsqueeze(1)  # B x N -> B x C(=1) x H x W
        attention = self.softmax( corr_map / self.tau )  # BX (N_query) X (N_key)
        proj_value = self.value_conv(value).view(B, -1, W * H)  # B X 256 X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x 256 x N_query
        out = out.view(B, C_value, H, W)

        out = self.gamma * out + value

        return corr_map, conf_map, out

class CorrSubnet(nn.Module):
    def __init__(self, opt):
        super(CorrSubnet, self).__init__()

        # create vgg Model
        self.vgg_feature_extracter = VGGFeatureExtractor(opt)

        self.non_local_blk = NonLocalBlock(256)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, tgt, ref):
        tgt_feature = self.vgg_feature_extracter(tgt, is_ref=False)
        ref_feature = self.vgg_feature_extracter(ref, is_ref=True)

        ref_value = self.vgg_feature_extracter(ref, isValue=True, is_ref=True)

        corr_map, conf_map, out = self.non_local_blk(ref_feature, tgt_feature, ref_value)

        return corr_map, conf_map, out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
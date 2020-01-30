import torch
import torch.nn as nn
import os
from torch.nn.parameter import Parameter
from skimage import io, color
from skimage.transform import resize
import util.util as util
from util.pca import pca
import pickle
import numpy as np
import torchvision


class VGG19BN_LAB(nn.Module):
    def __init__(self, checkpoint, requires_grad=False):
        super(VGG19BN_LAB, self).__init__()
        vgg_pretrained = torchvision.models.vgg19_bn(pretrained=False)
        vgg_pretrained.load_state_dict(checkpoint)

        vgg_pretrained_features = vgg_pretrained.features

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
        self.slice6 = nn.Sequential()

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
        for x in range(43, 52):
            self.slice6.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)

        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6]

        return out[-1]


def pickle_save(model, target_folder, feature_folder):
    idx_folder = 0
    for (path, _, files) in os.walk(target_folder):
        idx_folder += 1
        if len(files) > 0:

            path_base = os.path.basename(path)
            if os.path.exists(feature_folder + path_base + '.pickle'):
                print(feature_folder + path_base + '.pickle', 'already exist')
                continue
            feature_dict = dict()

            len_files = len(files)
            idx = 0
            for file in sorted(files):
                feature_file = dict()
                idx += 1
                filename = os.path.join(path, file)
                pickle_key = os.path.join(path_base, file)

                rgb = resize(io.imread(filename), (256, 256))
                if len(rgb.shape) < 3:
                    continue
                else:
                    lab = torch.from_numpy(color.rgb2lab(rgb)[:, :, 0]).float()

                lab = torch.unsqueeze(lab, 0)

                # Block saving luminance image, because of shortage of SSD available
                # feature_file['luminance'] = lab

                lab = torch.unsqueeze(lab, 0)
                lab = lab.repeat(1, 3, 1, 1)  # assume image (3 * 224 * 224) whose batch size = 1

                outputs = model.forward(lab.cuda())

                feat_size = outputs.size()[-2:]
                feature_map = outputs.squeeze(0).view(512, -1)
                try:
                    local_feature = pca(feature_map.t(), k=64).t().view(64, *feat_size)
                    print(pickle_key, local_feature.size(), idx, '/', len_files, idx_folder)
                    feature_file['feature'] = local_feature.cpu().numpy()
                    feature_dict[pickle_key] = feature_file
                except:
                    pass

            with open(feature_folder + path_base + '.pickle', 'wb') as handle:
                pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':  # Test Mode

    model_dict = torch.load('models/networks/checkpoint.pth.tar')['state_dict']
    model = VGG19BN_LAB(model_dict)

    model.cuda()
    # torch.save(model, 'model.pth')

    target_folder = '../data/imagenet/train/'
    feature_folder = '../data/imagenet_feature/'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder, exist_ok=True)

    pickle_save(model, target_folder, feature_folder)

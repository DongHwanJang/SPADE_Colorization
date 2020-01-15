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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
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


class Vgg19BN(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19BN, self).__init__()

        self.slice_input = nn.Sequential(
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            Scale(channels=3)
        )
        self.slice2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.slice3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.slice4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.slice5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.slice6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.9, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True)
        )


        # for x in range(2):
        #     self.slice_input.add_module(str(x), self.model_dict[x])
        #
        # for x in range(2, 12):
        #     self.slice2.add_module(str(x), self.model_dict[x])
        #
        # for x in range(12, 19):
        #     self.slice3.add_module(str(x), self.model_dict[x])
        #
        # for x in range(19, 32):
        #     self.slice4.add_module(str(x), self.model_dict[x])
        #
        # for x in range(32, 45):
        #     self.slice5.add_module(str(x), self.model_dict[x])
        #
        # for x in range(45, len(self.model_dict) - 1):
        #     self.slice6.add_module(str(x), self.model_dict[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_input = self.slice_input(x)
        h_relu2 = self.slice2(h_input)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        # outputs = [h_relu2, h_relu3, h_relu4, h_relu5, h_relu6]
        return h_relu6


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
                    # expand grayscale image (channel = 1) into rgb format
                    rgb = np.broadcast_to(np.expand_dims(rgb, 2), (256, 256, 3))
                    # rgb2lab (because grayscale is not fit to Luminance map)
                    lab = torch.from_numpy(color.rgb2lab(rgb)[:, :, 0]).float()
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
                local_feature = pca(feature_map.t(), k=64).t().view(64, *feat_size)
                print(pickle_key, local_feature.size(), idx, '/', len_files, idx_folder)
                feature_file['feature'] = local_feature.cpu().numpy()
                feature_dict[pickle_key] = feature_file

            with open(feature_folder + path_base + '.pickle', 'wb') as handle:
                pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':  # Test Mode

    # weight_root = './models/networks/'
    # model_dict = torch.load('./models/networks/vgg19_bn_gray.pth')
    # model = Vgg19BN()
    # model.load_state_dict(model_dict)

    model_dict = torch.load('gray_vgg19_torch.pth')
    model = Vgg19BN()
    model.load_state_dict(model_dict)

    model.cuda()
    # torch.save(model, 'model.pth')

    target_folder = '../data/imagenet/train/'
    feature_folder = '../data/imagenet_feature/'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder, exist_ok=True)

    pickle_save(model, target_folder, feature_folder)

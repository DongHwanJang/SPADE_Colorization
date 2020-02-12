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
from models.networks.architecture import VGG19BN_L

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

    model = VGG19BN_L('models/networks/checkpoint.pth.tar', requires_grad=False)

    model.cuda()
    # torch.save(model, 'model.pth')

    target_folder = '/data1/imagenet/train/'
    feature_folder = '/data1/imagenet_feature/'
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder, exist_ok=True)

    pickle_save(model, target_folder, feature_folder)
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
from torch.utils.data import Dataset, DataLoader
from util.img_loader import lab_loader, rgb_loader, rgb_pil2l_as_rgb, rgb_pil2lab_tensor
from PIL import Image
import torchvision.transforms as transforms

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

class ImageNetGrayscale(Dataset):
    def __init__(self, imagenet_path):
        self.imagenet_path = imagenet_path
        self.image_paths = []

        for root_path, _, files in os.walk(target_folder):
            for file in files:
                self.image_paths.append(os.path.join(root_path, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        file_path = self.image_paths[idx]
        with open(file_path, 'rb') as f:
            with Image.open(f) as img:
                rgb_image = img.convert('RGB')
                transform = transforms.Resize([256,256], interpolation=Image.BICUBIC)
                rgb_image = transform(rgb_image)
                target_L_gray_image = rgb_pil2l_as_rgb(rgb_image, need_Tensor=True)

                splitted_path = os.path.normpath(file_path).split("/")

        return target_L_gray_image, splitted_path[0], splitted_path[1]

def pickle_save(model, loader, target_folder, feature_folder):
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


                # Block saving luminance image, because of shortage of SSD available
                # feature_file['luminance'] = lab

                lab = torch.unsqueeze(lab, 0)
                lab = lab.repeat(1, 3, 1, 1)  # assume image (3 * 224 * 224) whose batch size = 1

                outputs = model.forward(lab.cuda())

                print(outputs.size())

                feat_size = outputs.size()[-2:]
                feature_map = outputs.squeeze(0).view(512, -1)
                try:

                    local_feature = pca(feature_map.t(), k=64).t().view(64, *feat_size)
                    print(pickle_key, local_feature.size(), idx, '/', len_files, idx_folder)
                    feature_file['feature'] = local_feature.cpu().numpy()
                    feature_dict[pickle_key] = feature_file
                    print("I'm in the loop")
                except:
                    pass

            print(path_base)
            with open(feature_folder + path_base + '.pickle', 'wb') as handle:
                pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':  # Test Mode

    target_folder = '/DATA1/hksong/imagenet/train/'
    feature_folder = '../data/imagenet_feature/'
    base_path = "/DATA1/isaac/imagenet_features"
    model_dict = torch.load('models/networks/checkpoint.pth.tar').state_dict()
    model = VGG19BN_LAB(model_dict)

    model.cuda()

    dataset = ImageNetGrayscale(target_folder)
    loader = DataLoader(dataset, batch_size = 128, shuffle=False, num_workers=20, pin_memory=True)


    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder, exist_ok=True)

    print("starting inference")
    feature_dict={}
    for grayscales, class_names, file_names in loader:
        print("in the loop!")
        print(grayscales.size())
        feature_maps = model.forward(grayscales.cuda())

        for feature_map, class_name, file_name in zip(feature_maps, class_names, file_names):
            feat_size = feature_map.size()[-2:]
            feature_map = feature_map.squeeze(0).view(512, -1)
            local_feature = pca(feature_map.t(), k=64).t().view(64, *feat_size)
            feature_dict[class_name] = local_feature.cpu().numpy()


            file_name = os.path.splitext(file_name)
            with open(os.path.join(os.path.join(base_path, class_names), file_name + ".pickel"), 'wb') as handle:
                pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle_save(model, loader, target_folder, feature_folder)


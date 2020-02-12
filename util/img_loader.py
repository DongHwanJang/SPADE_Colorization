from PIL import Image, ImageCms
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision.transforms import functional as F
import os
import torch
from util.util import normalize

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")


def lab_loader(opt, path, is_ref=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    phase = 'val/' if opt.phase == 'test' else 'train/'
    filename = opt.dataroot + phase + path
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            # TODO: choose whether ref_img contains lab, or l
            return ImageCms.applyTransform(img, rgb2lab_transform)


def rgb_loader(opt, path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    phase = 'val/' if opt.phase == 'test' else 'train/'
    filename = os.path.join(opt.dataroot, phase, path)
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def lab_deloader(lab_np, np_output):
    """

    :param lab_np: lab numpy array with the values in range of [0~100, -128~128, -128~128]
    :param np_output:  T/F for output    pil if np_output==False else np (0~255)
    :return: rgb converted pil object
    """
    rgb_np = (lab2rgb(lab_np.astype(np.float32))*255).astype(np.uint8)

    if np_output:
        return rgb_np

    return Image.fromarray(rgb_np)


def rgb_pil2lab_tensor(rgb_pil):
    lab128_np = rgb2lab(np.array(rgb_pil) / 255.0)
    lab128_np = lab128_np.astype(np.float32)
    return F.normalize(F.to_tensor(lab128_np), mean=(50, 0, 0), std=(50, 128, 128))

def rgb_pil2l_as_rgb(rgb_pil, need_Tensor=False):
    # rgb -> lab128_np
    lab128_np = rgb2lab(np.array(rgb_pil) / 255.0)
    H, W = lab128_np.shape[:2]

    # erase AB
    lab128_np[:, :, 1] = np.zeros((H, W))
    lab128_np[:, :, 2] = np.zeros((H, W))
    rgb_with_l = lab2rgb(lab128_np)

    if not need_Tensor:
        rgb_with_l *= 255.0
        return rgb_with_l.astype(np.uint8)

    else:
        return F.normalize(F.to_tensor(rgb_with_l).float(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def torch_lab2rgb(lab_image, normalize=False):
    '''
    :param lab_image: [-1., 1.]  format: Bx3XHxW  L is scaled. AB is shifted
    :param batch_size:
    :return: rgb_image [0., 1.]
    '''

    EPSILON = 1e-5
    # rgb_image = torch.zeros(lab_image.size()).to(config.device)
    lab_image = lab_image/2. + 0.5
    lab_image.clamp_(min=EPSILON, max=1-EPSILON)

    rgb_image = torch.zeros(lab_image.size())

    l_s = lab_image[:, 0, :, :] * 100
    a_s = (lab_image[:, 1, :, :] * 255) - 128
    b_s = (lab_image[:, 2, :, :] * 255) - 128

    var_Y = (l_s + 16.0) / 116.
    var_X = a_s / 500. + var_Y
    var_Z = var_Y - b_s / 200.

    mask_Y = var_Y.abs().pow(3.0) > 0.008856
    mask_X = var_X.pow(3.0) > 0.008856
    mask_Z = var_Z.pow(3.0) > 0.008856

    Y_1 = var_Y.abs().pow(3.0) * mask_Y.float()
    Y_2 = (var_Y - 16. / 116.) / 7.787 * (~mask_Y).float()
    var_Y = Y_1 + Y_2

    X_1 = var_X.abs().pow(3.0) * mask_X.float()
    X_2 = (var_X - 16. / 116.) / 7.787 * (~mask_X).float()
    var_X = X_1 + X_2

    Z_1 = var_Z.abs().pow(3.0) * mask_Z.float()
    Z_2 = (var_Z - 16. / 116.) / 7.787 * (~mask_Z).float()
    var_Z = Z_1 + Z_2

    X = 0.95047 * var_X
    Y = 1.00000 * var_Y
    Z = 1.08883 * var_Z

    var_R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    var_G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    var_B = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    var_R.clamp_(min=EPSILON)
    var_G.clamp_(min=EPSILON)
    var_B.clamp_(min=EPSILON)

    mask_R = var_R > 0.0031308
    R_1 = (1.055 * var_R.abs().pow(1 / 2.4) - 0.055) * mask_R.float()
    R_2 = (12.92 * var_R) * (~mask_R).float()
    var_R = R_1 + R_2

    mask_G = var_G > 0.0031308
    G_1 = (1.055 * var_G.abs().pow(1 / 2.4) - 0.055) * mask_G.float()
    G_2 = (12.92 * var_G) * (~mask_G).float()
    var_G = G_1 + G_2

    mask_B = var_B > 0.0031308
    B_1 = (1.055 * var_B.abs().pow(1 / 2.4) - 0.055) * mask_B.float()
    #B_2 = (12.92 * temp_B) * (~mask_B).float()
    B_2 = (12.92 * var_B) * (~mask_B).float()
    var_B = B_1 + B_2

    out = torch.cat([var_R.unsqueeze(1),
                        var_G.unsqueeze(1),
                        var_B.unsqueeze(1)], dim=1).clamp(EPSILON, 1.-EPSILON)

    if normalize:
        out = normalize(out)

    assert not (torch.isnan(out.max()) or torch.isnan(out.min()))

    return out
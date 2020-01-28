from PIL import Image, ImageCms
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision.transforms import functional as F
import os

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
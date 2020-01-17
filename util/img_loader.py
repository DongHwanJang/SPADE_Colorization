from PIL import Image, ImageCms
import numpy as np

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
    filename = opt.dataroot + phase + path
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def lab_deloader(img):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    return ImageCms.applyTransform(img, lab2rgb_transform)

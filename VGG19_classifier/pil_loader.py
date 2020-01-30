from PIL import Image, ImageCms
import numpy as np

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")

def get_pil_loader(mode):
    if mode=="LAB":
        return LAB_pil_loader

    if mode=="L":
        return L_pil_loader

def LAB_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            return ImageCms.applyTransform(img, rgb2lab_transform)

def L_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            lab = ImageCms.applyTransform(img, rgb2lab_transform)
            L, _, _ = lab.split()
            return L
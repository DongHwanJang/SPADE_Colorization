import numpy as np
import PIL.Image as Image
from PIL import ImageCms

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")

def get_subnet_images(opt, image, subnet_image_size):
    """
    opt must have:
    - crop_size
    - crop_to_ref: use cropping from original image to create reference
    - crop_to_ref_size: size of cropping window
    - crop_to_target: crop from ref to create target
    - crop_to_target_size: size of cropping window
    - flip_to_target: perform horizontal flipping
    """
    height = width = opt.crop_size

    # creat ref from original.
    # allowed transforms: crop and resize (bicubic)
    center_width = image.size[0] // 2
    center_height = image.size[1] // 2
    ref = image.crop((center_width - width//2, center_height - height//2,
                        center_width + width//2, center_height + height//2))
    ref = ref.resize((width, height), Image.BICUBIC)

    # create target from ref
    target_crop_width = target_crop_height = opt.crop_to_target_size

    target = ref
    if opt.crop_to_target:
        center_x, center_y = get_valid_center_coord(width, height, opt.crop_to_target_size, opt.crop_to_target_size)

        target = target.crop((center_x - target_crop_width//2, center_y - target_crop_height//2,
                center_x + target_crop_width//2, center_y + target_crop_height//2))

    if opt.flip_to_target:
        is_flipping = np.random.choice([True, False])
        if is_flipping:
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
    target = target.resize((width, height), Image.BILINEAR)

    # create index image
    w = np.array(range(width))
    w = np.tile(w, (height, 1))

    h = np.array(range(height)).reshape(-1, 1)
    h = np.tile(h, (1, width))

    filler = np.zeros((width, height))

    index_array = np.stack([w, h, filler], 2)

    index_image = Image.fromarray(index_array.astype(np.uint8))

    # apply the same transformations as target to index array
    if opt.crop_to_target:
        index_image = index_image.crop((center_width - target_crop_width//2, center_height - target_crop_height//2,
                center_width + target_crop_width//2, center_height + target_crop_height//2))

    if opt.flip_to_target and is_flipping:
        index_image = index_image.transpose(Image.FLIP_LEFT_RIGHT)
    index_image = index_image.resize((width, height), Image.BILINEAR)

    # create target used as GT for discriminator, perceptual, ... by resizing to the same resolution as the output of
    # correspondence subnet. Resizing should use bilinear
    subnet_width = subnet_height = subnet_image_size
    ratio = width // subnet_width

    target_gt = target.resize((subnet_width, subnet_height), Image.BILINEAR)

    index_image_gt = index_image.resize((subnet_width, subnet_height), Image.BILINEAR)
    index_image_gt = Image.fromarray(np.array(index_image_gt).astype(np.uint8) // ratio)

    ref_warp = ref.resize((subnet_width, subnet_height), Image.BILINEAR)

    return (ref, ref_warp), (target,target_gt), (index_image, index_image_gt)

def get_valid_center_coord(img_width, img_height, crop_width, crop_height):
    width_room = (img_width - crop_width)
    height_room = (img_height - crop_height)
    center_x = np.randon.choice(range(width_room)) - crop_width // 2
    center_y = np.randon.choice(range(height_room)) - crop_height // 2
    return (center_x, center_y)

def create_warpped_image(index_image, warp_image, target_gt):
    # PIL.Image[width][height] ==> np.array[height][width][channel]
    # index_image[width][height][channel] this also holds when converting between arrays and images
    # L, A, B
    width, height = index_image.size
    index_array = np.array(index_image).astype(np.uint8)
    output_array = np.zeros((height, width, 3)).astype(np.uint8)
    warp_array = ImageCms.applyTransform(warp_image, rgb2lab_transform)
    warp_array = np.array(warp_array).astype(np.uint8)
    target_gt_array = np.array(target_gt).astype(np.uint8)

    for h in range(height):
        for w in range(width):
            x = index_array[h][w][0]
            y = index_array[h][w][1]

            A, B = warp_array[y][x][1], warp_array[y][x][2]

            output_array[h][w][0] = target_gt_array[h][w][0] # L
            output_array[h][w][1] = A # A
            output_array[h][w][2] = B # B

    output_image = Image.fromarray(output_array, mode="LAB")
    output_image = ImageCms.applyTransform(output_image, lab2rgb_transform)
    return output_image
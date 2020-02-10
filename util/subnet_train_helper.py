import numpy as np
import PIL.Image as Image

def get_subnet_images(opt, image):
    """
    opt must have:
    - input_image_size
    - subnet_image_size: output of subnet (default: 64x64)
    - crop_to_ref: use cropping from original image to create reference
    - crop_to_ref_size: size of cropping window
    - crop_to_target: crop from ref to create target
    - crop_to_target_size: size of cropping window
    - flip_to_target: perform horizontal flipping

    """
    height = width = opt.input_image_size

    # creat ref from original.
    # allowed transforms: crop and resize (bicubic)
    center_width = image.size[0] // 2
    center_height = image.size[1] // 2
    ref = image.crop((center_width - width//2, center_height - height//2,
                        center_width + width//2, center_height + height//2))
    ref = ref.resize((width, height), Image.BICUBIC)

    # create target from ref
    center_width = ref.size[0] // 2
    center_height = ref.size[1] // 2

    target_crop_width = target_crop_height = opt.crop_to_target_size

    target = ref
    if opt.crop_to_target:
        target = target.crop((center_width - target_crop_width//2, center_height - target_crop_height//2,
                center_width + target_crop_width//2, center_height + target_crop_height//2))
    if opt.flip_to_target:
        target = target.transpose(Image.FLIP_LEFT_RIGHT)
    target = target.resize((width, height), Image.BILINEAR)

    # create index image
    w = np.array(range(width))
    w = np.tile(w, (height, 1))

    h = np.array(range(height)).reshape(-1, 1)
    h = np.tile(h, (1, width))

    filler = np.zeros((256, 256))

    index_array = np.stack([w, h, filler], 2)

    index_image = Image.fromarray(index_array.astype(np.uint8))

    # apply the same transformations as target to index array
    if opt.crop_to_target:
        index_image = index_image.crop((center_width - target_crop_width//2, center_height - target_crop_height//2,
                center_width + target_crop_width//2, center_height + target_crop_height//2))
    if opt.flip_to_target:
        index_image = index_image.transpose(Image.FLIP_LEFT_RIGHT)
    index_image = index_image.resize((width, height), Image.BILINEAR)

    # create target used as GT for discriminator, perceptual, ... by resizing to the same resolution as the output of
    # correspondence subnet. Resizing should use bilinear
    subnet_width = subnet_height = opt.subnet_image_size
    ratio = width // subnet_width

    target_gt = target.resize((subnet_width, subnet_height), Image.BILINEAR)

    index_image_gt = index_image.resize((subnet_width, subnet_height), Image.BILINEAR)
    index_image_gt = Image.fromarray(np.array(index_image_gt).astype(np.uint8) // ratio)

    ref_warp = ref.resize((subnet_width, subnet_height), Image.BILINEAR)

    return (ref, ref_warp), (target,target_gt), (index_image, index_image_gt)

def create_warpped_image(index_image, warp_image):
    # index_image[width][height][channel] this also holds when converting between arrays and images

    width, height = index_image.size
    index_array = np.array(index_image).astype(np.uint8)
    output_array = np.zeros((width,height,3)).astype(np.uint8)
    warp_array = np.array(warp_image).astype(np.uint8)

    for h in range(height):
        for w in range(width):
            x = index_array[h][w][0]
            y = index_array[h][w][1]

            pixel = [warp_array[y][x][0], warp_array[y][x][1], warp_array[y][x][2]]

            output_array[h][w][0] = pixel[0]
            output_array[h][w][1] = pixel[1]
            output_array[h][w][2] = pixel[2]

    output_image = Image.fromarray(output_array)
    return output_image

if __name__ == '__main__':
    image = Image.open("/DATA1/hksong/imagenet/train/n01440764/n01440764_8878.JPEG")
    (ref, ref_warp), (target,target_gt), (index_image, index_image_gt) = get_subnet_images(None, image)

    generated_warpped_image = create_warpped_image(index_image_gt, ref_warp)

    out_path = "/home/minds/isaac/SPADE_Colorization/transform_images"

    ref.save(out_path + "/ref" + ".JPEG")
    ref_warp.save(out_path + "/ref_warp" + ".JPEG")
    target.save(out_path + "/target" + ".JPEG")
    target_gt.save(out_path + "/target_gt" + ".JPEG")
    generated_warpped_image.save(out_path + "/generated_warpped_image" + ".JPEG")
    index_image.save(out_path + "/index_image" + ".JPEG")
    index_image_gt.save(out_path + "/index_image_gt" + ".JPEG")
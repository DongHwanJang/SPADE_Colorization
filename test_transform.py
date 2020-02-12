from PIL import Image
from util.subnet_train_helper import create_warpped_image,get_subnet_images
from options.train_options import TrainOptions

opt = TrainOptions().parse()
opt.crop_to_target = True
opt.flip_to_target = True
opt.crop_to_ref = True

image = Image.open("/DATA1/hksong/imagenet/train/n01440764/n01440764_8878.JPEG")
(ref, ref_warp), (target, target_gt), (index_image, index_image_gt) = get_subnet_images(opt, image, 64)

generated_warpped_image = create_warpped_image(index_image_gt, ref_warp, target_gt)

out_path = "/home/minds/isaac/SPADE_Colorization/transform_images"

ref.save(out_path + "/ref" + ".JPEG")
ref_warp.save(out_path + "/ref_warp" + ".JPEG")
target.save(out_path + "/target" + ".JPEG")
target_gt.save(out_path + "/target_gt" + ".JPEG")
generated_warpped_image.save(out_path + "/generated_warpped_image" + ".JPEG")
index_image.save(out_path + "/index_image" + ".JPEG")
index_image_gt.save(out_path + "/index_image_gt" + ".JPEG")
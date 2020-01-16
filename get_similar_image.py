import pickle
import os
import numpy as np

from util.distance import nearest_neighbor, find_fast
import time
import PIL.Image as Image

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

# TODO: Luminance is too heavy, so you should import an image and convert it into 'LAB' format
#  using following snippet


"""""
with Image.open(f) as img:
        img = img.convert('RGB')
        lab = ImageCms.applyTransform(img, rgb2lab_transform)
        return np.broadcast_to(np.expand_dims(lab[0, :, :], axis=0), (3, 224, 224))
"""""


# def find_similar_image_original(feature_dict, target_key, k, stop_at):
#     target_feature = feature_dict[target_key]['feature']
#     c, h, w = target_feature.shape
#     similarity_score = list()
#
#     for idx, (key, value) in enumerate(feature_dict.items()):
#         if idx == stop_at:
#             break
#         # print(idx, key)
#         if key == target_key:
#             pass
#         else:
#             cos_dist = 0
#             lum_dist = 0
#
#             # local similarity
#             for i in range(h):
#                 for j in range(w):
#
#                     # get cosine distance with nearest neighbor pixel
#                     vec_p = target_feature[:, i, j]
#                     cos_pixel_score, q_h, q_w = nearest_neighbor(vec_p, value['feature'])
#                     cos_dist += cos_pixel_score
#
#                     # from nearest neighbor pixel, get corrcoef in luminance map
#                     # TODO: change the dimension of 'luminance' into 3
#                     target_lum = feature_dict[target_key]['luminance'][0, 0, i * 16:(i + 1) * 16,
#                                  j * 16:(j + 1) * 16].reshape(1, -1)
#                     ref_lum = feature_dict[key]['luminance'][0, 0, q_h * 16:(q_h + 1) * 16,
#                               q_w * 16:(q_w + 1) * 16].reshape(1, -1)
#                     lum_pixel_score = np.corrcoef(target_lum, ref_lum).squeeze()[0][1]
#                     lum_dist += lum_pixel_score
#
#             total_dist = lum_dist * 0.25 + cos_dist
#             # print(total_dist, key)
#             similarity_score.append([total_dist, key])
#
#     # export most similar image
#     return sorted(similarity_score, key=lambda elem: elem[0], reverse=True)[:k]


def find_similar_image_fast(feature_dict, target_key, k, stop_at):
    target_feature = feature_dict[target_key]['feature']
    cos_score = list()
    for idx, (key, value) in enumerate(feature_dict.items()):
        if idx == stop_at:
            break

        if key == target_key:
            pass
        else:
            cos_score_target = find_fast(target_feature, value['feature'], key)
            cos_score.append([cos_score_target, key])
    return sorted(cos_score, key=lambda elem: elem[0], reverse=True)[:k]


def find_similar_image(feature_dict, target_key, k=1, mode='original', stop_at=10):
    # if mode == 'original':
    #     return find_similar_image_original(feature_dict, target_key, k, stop_at=stop_at)
    # else:
        return find_similar_image_fast(feature_dict, target_key, k, stop_at=stop_at)


if __name__ == '__main__':

    # TODO: get dataroot from base_option (need to discuss about data path)
    # Suppose that folder.pickle = {filecode: {feature: relu5_4, luminance: l-map(256x256 resized)}}

    similar = []
    k = 3  # choose top-k
    stop_at = 10 # choose stop at
    data_root = '../data/imagenet_feature/'  # Here in absolute : /DATA1/hksong/imagenet_feature/
    pair_root = './pair_img/'

    if not os.path.exists(pair_root):
        os.makedirs(pair_root)

    for file in sorted(os.listdir(data_root))[:]:
        print("{:s} is in progress...".format(file))
        pickle_path = data_root + file
        pair_path = pair_root + os.path.splitext(file)[0] + '.txt'
        if os.path.exists(pair_path):
            print("{:s} already exist!!!".format(pair_path))
            continue
        print(pickle_path)
        feature_dict = load_pickle(pickle_path)

        with open(pair_path, mode='w') as text_file:
            for idx, (target_key, feature) in enumerate(feature_dict.items()):
                start_time = time.time()

                # print(target_key)
                ref_keys = find_similar_image(feature_dict, target_key, k=k, mode='fast', stop_at=stop_at)

                end_time = time.time()
                # print(end_time - start_time)

                text_file.write("{:s} ".format(target_key))
                text_file.write(" ".join(["{:s} {:f}".format(ref_key[1], ref_key[0]) for ref_key in ref_keys]))
                text_file.write("\n")
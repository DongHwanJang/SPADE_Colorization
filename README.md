This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

To reproduce the results reported in the paper, you would need an NVIDIA DGX1 machine with 8 V100 GPUs.

---

## Variable naming convention
Varible name should be a combination one or more parts below:
- `normalized` - normalized. If not indicated, then unnormalized
- `pil` or `image` - `PIL.Image` object. If not indicated, then `torch.Tensor`
- `subnet` - variable is for subnet, and should not be fed into the main network
- `resized` - resized from original image size (default:`256x256`) to the size of the correspondence matrix (default:`64x64`) with `bilinear`
- `warped` - warped output created according to the attention
- `gt` - should be used as ground truth when calculating loss

---

## Specifics

### Main network
- **VGG perceptual loss**
  - input: normalized RGB

- **discriminator**
  - input: LAB

- **smoothness loss**
  - input: LAB

- **FID**
  - input: RGB (not normalized)


### Subnet:
- subnet_transformer: 
  - input: `image` [wxh] PIL Image in RGB 
  - output: `(ref[256x256], ref_warp[64x64]),(target[256x256], target_gt[64x64]),(index_image[256x256], index_image_warp[64x64])` all of them are `PIL.Image` in RGB
    - we use indexes stored in `index_image_warp[64x64]` to index into `ref_warp[64x64]` and extract `AB` channels. For `L` channel, `L` from `target_gt[64x64]` is used.
    
- **subnet**
  - input:  `target_L`  & `reference_LAB` (256x256) normalized

- **discriminator**
  - input: `warped_LAB`, `target_LAB` (256x256) normalized

- **FID**
  - input: `warped_RGB` & `target_RGB` (not normalized)

- **smoothness**
  - input: `warped_LAB` (64x64)

- **softmax**
  - input: raw `attention` from subnet (64x64x64^2) & index tesnor (64x64)

- **VGG**
  - input: `target_RGB` & `generated_RGB` (64x64) (이미 네트워크의 인풋으로 normalized 를 넣어주는데 이때 genearted LAB 는 또 normalize 를 해줘야하나?)

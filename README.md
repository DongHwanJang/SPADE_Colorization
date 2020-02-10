This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

To reproduce the results reported in the paper, you would need an NVIDIA DGX1 machine with 8 V100 GPUs.

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

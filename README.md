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
- VGG perceptual loss
input: normalized RGB

- discriminator:
input: LAB

- smoothness loss:
input: LAB

- FID:
input: RGB (not normalized)


### Subnet:

- subnet:
input: LAB[256x256] (normalized?)

- discriminator:

- FID:

- smoothness:

- softmax:

- VGG:

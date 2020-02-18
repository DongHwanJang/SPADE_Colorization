# Provided from Bo Zhang
import torch.nn.functional as F

def find_local_patch(x, patch_size):
    # unfold the image
    N, C, H, W = x.shape
    x_unfold = F.unfold(x, kernel_size=(patch_size, patch_size), padding=(
        patch_size // 2, patch_size // 2), stride=(1, 1))
    out = x_unfold.view(N, x_unfold.shape[1], H, W)
    return out
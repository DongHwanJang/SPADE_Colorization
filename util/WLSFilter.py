# source: https://github.com/zhaoyin214/wls_filter/blob/master/wls/wls_filter.py
import numpy as np
from scipy.sparse import spdiags, linalg

EPS = 1e-4
DIM_X = 1
DIM_Y = 0

def wls_filter(luma, lambda_=1, alpha=1.2):
    """
    edge-preserving smoothing via weighted least squares (WLS)
        u = F_λ (g) = (I + λ L_g)^(-1) g
        L_g = D_x^T A_x D_x +D_y^T A_y D_y
    arguments:
        luma (2-dim array, required) - the input image luma
        lambda_ (float) - balance between the data term and
            the smoothness term
        alpha (float) - a degree of control over the affinities
            by non-lineary scaling the gradients
    return:
        out (2-dim array)
    """
    height, width = luma.shape[0 : 2]
    size = height * width
    log_luma = np.log(luma + EPS)

    # affinities between adjacent pixels based on gradients of luma
    # dy
    diff_log_luma_y = np.diff(a=log_luma, n=1, axis=DIM_Y)
    diff_log_luma_y = - lambda_ / (np.abs(diff_log_luma_y) ** alpha + EPS)
    diff_log_luma_y = np.pad(
        array=diff_log_luma_y, pad_width=((0, 1), (0, 0)),
        mode="constant"
    )
    diff_log_luma_y = diff_log_luma_y.ravel()

    # dx
    diff_log_luma_x = np.diff(a=log_luma, n=1, axis=DIM_X)
    diff_log_luma_x = - lambda_ / (np.abs(diff_log_luma_x) ** alpha + EPS)
    diff_log_luma_x = np.pad(
        array=diff_log_luma_x, pad_width=((0, 0), (0, 1)),
        mode="constant"
    )
    diff_log_luma_x = diff_log_luma_x.ravel()

    # construct a five-point spatially inhomogeneous Laplacian matrix
    diff_log_luma = np.vstack((diff_log_luma_y, diff_log_luma_x))
    smooth_weights = spdiags(data=diff_log_luma, diags=[-width, -1],
                             m=size, n=size)

    w = np.pad(array=diff_log_luma_y, pad_width=(width, 0), mode="constant")
    w = w[: -width]
    n = np.pad(array=diff_log_luma_x, pad_width=(1, 0), mode="constant")
    n = n[: -1]

    diag_data = 1 - (diff_log_luma_x + w + diff_log_luma_y + n)
    smooth_weights = smooth_weights + smooth_weights.transpose() + \
        spdiags(data=diag_data, diags=0, m=size, n=size)

    out, _ = linalg.cg(A=smooth_weights, b=luma.ravel())
    out = out.reshape((height, width))
    # out = np.clip(a=out, a_min=0, a_max=100)

    return out
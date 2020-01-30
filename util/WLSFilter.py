# source: https://github.com/zhaoyin214/wls_filter/blob/master/wls/wls_filter.py
import numpy as np
from scipy.sparse import spdiags, linalg
from torch import nn as nn
import torch

EPS = 1e-4
DIM_X = 1
DIM_Y = 0

def ref_wls_filter(luma, lambda_=1, alpha=1.2):
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
        array=diff_log_luma_y, pad_width=((0, 1), (0, 0)), # (top, bottom)(left, right)
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
    log_luma = torch.log(luma + EPS)

    # affinities between adjacent pixels based on gradients of luma
    # dy
    diff_log_luma_y = diff(log_luma, n=1, axis=DIM_Y)
    diff_log_luma_y = - lambda_ / (torch.abs(diff_log_luma_y) ** alpha + EPS)
    diff_log_luma_y = nn.functional.pad(
        diff_log_luma_y, pad=(0, 0, 0, 1),
        mode="constant"
    )
    diff_log_luma_y = diff_log_luma_y.view(-1)

    # dx
    diff_log_luma_x = diff(log_luma, n=1, axis=DIM_X)
    diff_log_luma_x = - lambda_ / (np.abs(diff_log_luma_x) ** alpha + EPS)
    diff_log_luma_x = nn.functional.pad(
        diff_log_luma_x, pad=(0, 1, 0, 0),
        mode="constant"
    )
    diff_log_luma_x = diff_log_luma_x.view(-1)

    # construct a five-point spatially inhomogeneous Laplacian matrix
    diff_log_luma = torch.stack((diff_log_luma_y, diff_log_luma_x), dim=0)
    smooth_weights = spdiags(data=diff_log_luma, diags=[-width, -1], m=size, n=size)

    w = nn.functional.pad(diff_log_luma_y, pad=(width, 0), mode="constant")
    w = w[: -width]
    n = nn.functional.pad(diff_log_luma_x, pad=(1, 0), mode="constant")
    n = n[: -1]

    diag_data = 1 - (diff_log_luma_x + w + diff_log_luma_y + n)
    smooth_weights = smooth_weights + smooth_weights.transpose() + \
        spdiags(data=diag_data, diags=0, m=size, n=size)

    cg = CG(smooth_weights)
    out = cg(luma)
    out = out.reshape((height, width))
    # out = np.clip(a=out, a_min=0, a_max=100)

    return out

import torch
import time

def diff(input, n=1, axis=0):
    if axis == 0:
        return input[n:,:,:] - input[:-n,:,:]
    if axis == 1:
        return input[:, n:,:] - input[:, :-n,:]
    if axis == 2:
        return input[:, :, n:] - input[:, :, :-n]

    # if axis == 0:
    #     return input[n:,:,:,:] - input[:-n,:,:,:]
    # if axis == 1:
    #     return input[:, n:, :, :] - input[:, :-n, :, :]
    # if axis == 2:
    #     return input[:,:,n:,:] - input[:,:,:-n,:]
    # if axis == 3:
    #     return input[n:,:,:,n:] - input[:,:,:,:-n]

# source: https://github.com/sbarratt/torch_cg/blob/master/torch_cg/cg_batch.py
def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):

    def __init__(self, A_bmm, M_bmm=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X, _ = cg_batch(self.A_bmm, B, M_bmm=self.M_bmm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    def backward(self, dX):
        dB, _ = cg_batch(self.A_bmm, dX, M_bmm=self.M_bmm, X0=X0, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB

if __name__ == "__main__":
    in_tensor = torch.randn(3, 3, 3)
    mine = wls_filter(in_tensor)


    ref = ref_wls_filter(in_tensor.numpy())
    print(ref)
"""
HMWB baseline for hyperspectral image fusion.

This module contains the implementation used to reproduce the
HMWB method as a baseline for comparison with the proposed
Fused Gromov-Wasserstein (EFGW) method.
"""

import time

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.signal import convolve2d
from skimage.transform import resize


def interpolation_spectral(M, m, reg):
    """
    Reconstruct hyperspectral bands from the multispectral image.

    The reconstruction is performed using Tikhonov-regularized
    least squares.

    Parameters
    ----------
    M : np.ndarray, shape (B_m, B_h)
        Spectral degradation matrix from HSI bands to MSI bands.
    m : np.ndarray, shape (H, W, B_m)
        Multispectral image.
    reg : float
        Regularization parameter.

    Returns
    -------
    np.ndarray, shape (H, W, B_h)
        Spectrally interpolated hyperspectral image.
    """

    H, W, B_m = m.shape

    if B_m != M.shape[0]:
        raise ValueError(
            f"The number of bands in m ({B_m}) does not match "
            f"the number of rows of M ({M.shape[0]})."
        )

    # Vectorize the MSI.
    m_vec = m.reshape(-1, B_m).T

    # Tikhonov-regularized pseudo-inverse.
    MtM = M.T @ M
    MtM_reg = MtM + reg * np.eye(MtM.shape[0])

    M_tilde = np.linalg.inv(MtM_reg) @ M.T

    # Apply the spectral interpolation.
    m_tilde_vec = M_tilde @ m_vec

    B_h = M_tilde.shape[0]

    # Restore the spatial cube representation.
    m_tilde = m_tilde_vec.T.reshape(
        H,
        W,
        B_h
    )

    return m_tilde


def interpolation_spatial(
    h,
    target_shape,
    kernel_size,
    sigma
):
    """
    Upsample and spatially smooth a low-resolution HSI.

    Parameters
    ----------
    h : np.ndarray, shape (H_h, W_h, B)
        Low-spatial-resolution hyperspectral image.
    target_shape : tuple of int
        Target shape (H_target, W_target, B).
    kernel_size : int
        Size of the Gaussian smoothing kernel.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray, shape target_shape
        Spatially interpolated hyperspectral image.
    """

    if target_shape is None:
        target_shape = h.shape

    # Build the Gaussian kernel.
    ax = np.arange(
        -(kernel_size // 2),
        kernel_size // 2 + 1
    )

    xx, yy = np.meshgrid(
        ax,
        ax
    )

    kernel = np.exp(
        -(xx**2 + yy**2)
        / (2 * sigma**2)
    )

    kernel /= kernel.sum()

    # Spatial upsampling.
    h_tilde = resize(
        h,
        target_shape,
        order=3,
        mode="reflect",
        anti_aliasing=False
    )

    # Band-wise spatial smoothing.
    smoothed = np.zeros_like(h_tilde)

    for band in range(h_tilde.shape[2]):
        smoothed[:, :, band] = convolve2d(
            h_tilde[:, :, band],
            kernel,
            mode="same",
            boundary="wrap"
        )

    return smoothed


def kernel_matrix(
    nl,
    nc,
    nb,
    alpha,
    gamma,
    mode="econ"
):
    """
    Construct the separable kernel used by HMWB.

    Parameters
    ----------
    nl : int
        Number of spatial rows.
    nc : int
        Number of spatial columns.
    nb : int
        Number of spectral bands.
    alpha : float
        Spectral kernel weight.
    gamma : float
        Kernel regularization parameter.
    mode : {"econ", "full"}, optional
        If "econ", return the FFT-compatible tensor representation.
        Otherwise, return the full kernel matrix.

    Returns
    -------
    np.ndarray
        Kernel representation. In "econ" mode, the shape is
        (2*nl-1, 2*nc-1, 2*nb-1).
    """

    k00 = np.fft.ifftshift(
        np.exp(
            -np.arange(-nl + 1, nl) ** 2
            / gamma
        )
    )

    k0c = np.fft.ifftshift(
        np.exp(
            -np.arange(-nc + 1, nc) ** 2
            / gamma
        )
    )

    ki = np.fft.ifftshift(
        np.exp(
            -alpha
            * np.arange(-nb + 1, nb) ** 2
            / gamma
        )
    )

    if mode == "econ":
        return np.einsum(
            "i,j,k->ijk",
            k00,
            k0c,
            ki
        )

    # Full matrix representation.
    l, c, b = np.meshgrid(
        np.arange(nl),
        np.arange(nc),
        np.arange(nb)
    )

    l = l.flatten()
    c = c.flatten()
    b = b.flatten()

    distance = (
        (l[..., None] - l) ** 2
        + (c[..., None] - c) ** 2
        + alpha * (b[..., None] - b) ** 2
    )

    return np.exp(
        -distance / gamma
    )


def Tprodn(Tvals, y):
    """
    Multiply a generalized Toeplitz operator by an input matrix
    using FFT operations.

    Parameters
    ----------
    Tvals : np.ndarray
        Tensor containing the diagonal values of the generalized
        Toeplitz matrix.
    y : np.ndarray, shape (N, R)
        Input vector or matrix.

    Returns
    -------
    np.ndarray
        Result of the Toeplitz matrix multiplication.
    """

    d = Tvals.ndim
    n = Tvals.shape

    if n[-1] == 1:
        d -= 1

    n = n[:d]

    r = y.shape[1]

    ndiag = tuple(
        (np.asarray(n) + 1) // 2
    )

    # Reshape input according to the spatial-spectral dimensions.
    z = np.reshape(
        y,
        ndiag + (r,)
    )

    # FFT-based Toeplitz multiplication.
    Tz = np.real(
        ifftn(
            fftn(
                Tvals,
                axes=tuple(range(d))
            )[..., None]
            *
            fftn(
                z,
                s=n,
                axes=tuple(range(d))
            ),
            axes=tuple(range(d))
        )
    )

    Tz = np.squeeze(Tz)

    # Crop to the original dimensions.
    crop = tuple(
        slice(0, ndiag[i])
        for i in range(d)
    )

    Tz = Tz[crop]

    return np.reshape(
        Tz,
        (np.prod(ndiag), r)
    )


def hmwb(
    m_tilde,
    h_tilde,
    xi_m,
    xi_h,
    lam=0.1,
    niter=10,
    eps=1e-10
):
    """
    Run the HMWB fusion algorithm.

    This implementation reproduces the HMWB procedure used as
    a baseline for comparison with EFGW.

    Parameters
    ----------
    m_tilde : np.ndarray, shape (H, W, B)
        Spectrally interpolated multispectral image.
    h_tilde : np.ndarray, shape (H, W, B)
        Spatially interpolated hyperspectral image.
    xi_m : np.ndarray
        Kernel tensor associated with the MSI.
    xi_h : np.ndarray
        Kernel tensor associated with the HSI.
    lam : float, optional
        Weight used in the barycentric update.
    niter : int, optional
        Number of HMWB iterations.
    eps : float, optional
        Small numerical constant used to avoid division by zero.

    Returns
    -------
    np.ndarray, shape (H, W, B)
        Fused hyperspectral image produced by HMWB.
    """

    # Dimensions.
    nl, nc, nb = m_tilde.shape

    # Flatten spatial dimensions.
    m_flat = m_tilde.reshape(
        nl * nc,
        nb
    ).T

    h_flat = h_tilde.reshape(
        nl * nc,
        nb
    ).T

    # Initialize scaling variables.
    v_M = np.ones_like(m_flat)
    v_H = np.ones_like(h_flat)

    u_M = np.ones_like(m_flat)
    u_H = np.ones_like(h_flat)

    # Main HMWB iterations.
    for iteration in range(niter):

        # Sinkhorn scaling updates.
        Km = Tprodn(
            xi_m,
            u_M.reshape(-1, 1)
        ).reshape(
            nb,
            nl * nc
        )

        Kh = Tprodn(
            xi_h,
            u_H.reshape(-1, 1)
        ).reshape(
            nb,
            nl * nc
        )

        v_M = m_flat / (Km + eps)
        v_H = h_flat / (Kh + eps)

        # Compute the barycenter.
        Km_v = Tprodn(
            xi_m,
            v_M.reshape(-1, 1)
        ).reshape(
            nb,
            nl * nc
        )

        Kh_v = Tprodn(
            xi_h,
            v_H.reshape(-1, 1)
        ).reshape(
            nb,
            nl * nc
        )

        Km_v = np.where(
            Km_v < 0,
            eps,
            Km_v
        )

        f = np.exp(
            lam * np.log(
                u_M * Km_v + eps
            )
            +
            (1 - lam)
            * np.log(
                u_H * Kh_v + eps
            )
        )

        # Normalization.
        f = f / np.sum(f)

        # Update scaling variables.
        u_M = f / (Km_v + eps)
        u_H = f / (Kh_v + eps)

        if iteration % 5 == 0:
            print(
                f"[HMWB iteration {iteration}] "
                f"sum(f) = {f.sum():.4e}"
            )

    # Restore cube representation.
    f_cube = f.T.reshape(
        nl,
        nc,
        nb
    )

    return f_cube


def run_hmwb(
    reference,
    msi,
    hsi,
    spectral_response,
    spectral_reg=1e-6,
    alpha_m=1e-1,
    gamma_m=1e-6,
    alpha_h=300,
    gamma_h=1e-4,
    lam=0.5,
    niter=100,
):
    """
    Run the complete HMWB baseline pipeline.

    The pipeline consists of spectral interpolation of the MSI,
    spatial interpolation of the HSI, kernel construction, and
    HMWB fusion.

    Parameters
    ----------
    reference : np.ndarray, shape (H, W, B)
        Reference hyperspectral image. Its spatial and spectral
        dimensions define the target dimensions.
    msi : np.ndarray, shape (H, W, B_m)
        Multispectral image.
    hsi : np.ndarray, shape (H_h, W_h, B)
        Low-spatial-resolution hyperspectral image.
    spectral_response : np.ndarray, shape (B_m, B)
        Spectral response matrix used for MSI generation.
    spectral_reg : float, optional
        Regularization parameter for spectral interpolation.
    alpha_m : float, optional
        Kernel spectral parameter for the MSI.
    gamma_m : float, optional
        Kernel regularization parameter for the MSI.
    alpha_h : float, optional
        Kernel spectral parameter for the HSI.
    gamma_h : float, optional
        Kernel regularization parameter for the HSI.
    lam : float, optional
        HMWB barycenter parameter.
    niter : int, optional
        Number of HMWB iterations.

    Returns
    -------
    f_hmwb : np.ndarray, shape (H, W, B)
        Fused hyperspectral image produced by HMWB.
    elapsed_time : float
        HMWB execution time in seconds.
    """

    # Spectral interpolation of the MSI.
    m_tilde = interpolation_spectral(
        spectral_response,
        msi,
        reg=spectral_reg
    )

    # Spatial interpolation of the HSI.
    h_tilde = interpolation_spatial(
        hsi,
        target_shape=reference.shape,
        kernel_size=5,
        sigma=2
    )

    # Normalize the interpolated observations.
    eps = 1e-8

    m_tilde = m_tilde / (
        m_tilde.sum() + eps
    )

    h_tilde = h_tilde / (
        h_tilde.sum() + eps
    )

    # Target dimensions.
    nl, nc, nb = reference.shape

    # Construct HMWB kernels.
    xi_m = kernel_matrix(
        nl,
        nc,
        nb,
        alpha=alpha_m,
        gamma=gamma_m
    )

    xi_h = kernel_matrix(
        nl,
        nc,
        nb,
        alpha=alpha_h,
        gamma=gamma_h
    )

    # Measure execution time of the HMWB optimization.
    start_time = time.time()

    f_hmwb = hmwb(
        m_tilde,
        h_tilde,
        xi_m,
        xi_h,
        lam=lam,
        niter=niter
    )

    elapsed_time = time.time() - start_time

    return f_hmwb, elapsed_time
"""
Data loading, patch extraction, and synthetic observation generation.

This module provides the functions required to:

1. Load the reference hyperspectral datasets.
2. Extract spatial patches.
3. Generate synthetic low-spatial-resolution HSI observations.
4. Generate synthetic high-spatial-resolution MSI observations.
"""

import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d


def load_pavia(path="data/PaviaU.mat"):
    """
    Load the Pavia University hyperspectral image.

    Parameters
    ----------
    path : str, optional
        Path to the MATLAB file containing the Pavia University
        hyperspectral image.

    Returns
    -------
    cube : np.ndarray
        Hyperspectral image with shape

            (H, W, B_h)

        where:
        - H is the image height,
        - W is the image width,
        - B_h is the number of hyperspectral bands.
    """
    data = sio.loadmat(path)
    return data["paviaU"]


def load_indian_pines(path="data/indianpinearray.npy"):
    """
    Load the Indian Pines hyperspectral image.

    Parameters
    ----------
    path : str, optional
        Path to the NumPy file containing the Indian Pines
        hyperspectral cube.

    Returns
    -------
    cube : np.ndarray
        Hyperspectral image with shape

            (H, W, B_h).

    Notes
    -----
    The expected file is ``indianpinearray.npy``.
    """
    return np.load(path)


def extract_patch(cube, row, col, height, width):
    """
    Extract a spatial patch from a hyperspectral image.

    Parameters
    ----------
    cube : np.ndarray
        Input hyperspectral cube with shape

            (H, W, B).

    row : int
        Starting row of the patch.

    col : int
        Starting column of the patch.

    height : int
        Requested patch height.

    width : int
        Requested patch width.

    Returns
    -------
    patch : np.ndarray
        Extracted hyperspectral patch with shape

            (H_p, W_p, B),

        where H_p <= height and W_p <= width if the requested
        patch reaches an image boundary.
    """
    H, W, _ = cube.shape

    row_end = min(row + height, H)
    col_end = min(col + width, W)

    return cube[row:row_end, col:col_end, :]


def gaussian_kernel(size, sigma):
    """
    Construct a normalized two-dimensional Gaussian kernel.

    Parameters
    ----------
    size : int
        Kernel size. For example, ``size=5`` produces a
        5 x 5 kernel.

    sigma : float
        Standard deviation of the Gaussian function.

    Returns
    -------
    kernel : np.ndarray
        Normalized Gaussian kernel with shape

            (size, size).

        The kernel coefficients sum to one.
    """
    ax = np.arange(-(size // 2), size // 2 + 1)

    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    return kernel / kernel.sum()


def generate_hsi(
    image,
    kernel_size=5,
    sigma=2,
    downsample=4,
    noise_std=0
):
    """
    Generate a synthetic low-spatial-resolution HSI.

    The reference image is first spatially blurred and then
    spatially downsampled.

    Parameters
    ----------
    image : np.ndarray
        Reference hyperspectral image with shape

            (H, W, B_h).

    kernel_size : int, optional
        Size of the spatial Gaussian blur kernel.

    sigma : float, optional
        Standard deviation of the Gaussian blur kernel.

    downsample : int, optional
        Spatial downsampling factor.

    noise_std : float, optional
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    h : np.ndarray
        Synthetic hyperspectral observation with shape

            (H_h, W_h, B_h),

        where H_h and W_h are reduced spatial dimensions.

    Notes
    -----
    The spectral dimension is preserved. Only the spatial
    resolution is degraded.
    """
    B = image.shape[2]

    kernel = gaussian_kernel(
        size=kernel_size,
        sigma=sigma
    )

    blurred = np.zeros_like(image)

    for band in range(B):
        blurred[:, :, band] = convolve2d(
            image[:, :, band],
            kernel,
            mode="same",
            boundary="wrap"
        )

    h = blurred[::downsample, ::downsample, :]

    noise = np.random.normal(
        0,
        noise_std,
        h.shape
    )

    return h + noise


def generate_msi(
    image,
    n_bands=6,
    noise_std=0,
    sigma=5
):
    """
    Generate a synthetic multispectral image.

    The MSI is obtained by applying Gaussian spectral response
    functions to the hyperspectral bands.

    Parameters
    ----------
    image : np.ndarray
        Reference hyperspectral image with shape

            (H, W, B_h).

    n_bands : int, optional
        Number of multispectral bands to generate, denoted B_m.

    noise_std : float, optional
        Standard deviation of additive Gaussian noise.

    sigma : float, optional
        Standard deviation of the Gaussian spectral response
        functions.

    Returns
    -------
    m : np.ndarray
        Synthetic multispectral image with shape

            (H, W, B_m).

    M : np.ndarray
        Spectral response matrix with shape

            (B_m, B_h).

        Each row describes the spectral response of one MSI band.

    centers : np.ndarray
        Spectral centers of the Gaussian response functions.
        Shape:

            (B_m,).

    Notes
    -----
    The spatial resolution is preserved while the spectral
    resolution is reduced.
    """
    H, W, B = image.shape

    lambda_h = np.arange(1, B + 1)

    centers = np.linspace(
        10,
        B - 10,
        n_bands
    ).astype(int)

    sigmas = [sigma] * n_bands

    M = np.zeros(
        (n_bands, B)
    )

    for i in range(n_bands):
        M[i, :] = np.exp(
            -0.5
            * (
                (lambda_h - centers[i])
                / sigmas[i]
            ) ** 2
        )

    image_2d = image.reshape(-1, B).T

    M_image = M @ image_2d

    m = M_image.T.reshape(
        H,
        W,
        n_bands
    )

    noise = np.random.normal(
        0,
        noise_std,
        m.shape
    )

    return m + noise, M, centers



"""
Fused Gromov-Wasserstein cost fusion and optimization.

This module constructs the different quantities required by the
Fused Gromov-Wasserstein problem:

- spectral cost;
- structural distance matrices;
- pixel-band cost matrix;
- optimal transport masses;
- entropic FGW transport plan.
"""

import numpy as np
import ot
from scipy.spatial.distance import cdist


def sam_cost(A, B):
    """
    Compute the Spectral Angle Mapper cost matrix.

    Parameters
    ----------
    A : np.ndarray
        First set of spectra with shape

            (N_a, B).

    B : np.ndarray
        Second set of spectra with shape

            (N_b, B).

        Both inputs must have the same spectral dimension B.

    Returns
    -------
    C : np.ndarray
        Pairwise SAM cost matrix with shape

            (N_a, N_b).

        Each entry C[i, j] is the spectral angle between
        spectrum A[i] and spectrum B[j].
    """
    A_norm = A / (
        np.linalg.norm(
            A,
            axis=1,
            keepdims=True
        )
        + 1e-12
    )

    B_norm = B / (
        np.linalg.norm(
            B,
            axis=1,
            keepdims=True
        )
        + 1e-12
    )

    similarity = np.clip(
        A_norm @ B_norm.T,
        -1.0,
        1.0
    )

    return np.arccos(similarity)


def distance_matrix(cube, alpha=1.0):
    """
    Construct a structural distance matrix for a hyperspectral cube.

    Each voxel is represented by a three-dimensional coordinate
    composed of:

    - normalized spatial coordinate x;
    - normalized spatial coordinate y;
    - normalized spectral coordinate z.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral or multispectral cube with shape

            (H, W, B).

    alpha : float, optional
        Weight applied to the spectral coordinate.

        The spectral coordinate is multiplied by sqrt(alpha).

    Returns
    -------
    C : np.ndarray
        Pairwise Euclidean distance matrix with shape

            (N, N),

        where

            N = H * W * B.

    Notes
    -----
    This matrix is constructed explicitly and can therefore
    become very large for high-resolution cubes.
    """
    H, W, C = cube.shape

    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    z = np.linspace(0, 1, C)

    X, Y, Z = np.meshgrid(
        x,
        y,
        z,
        indexing="xy"
    )

    coords = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])

    coords[:, 2] *= np.sqrt(alpha)

    return cdist(
        coords,
        coords,
        metric="euclidean"
    )


def gaussian_response(
    B_h,
    B_m,
    centers,
    sigma=5.0
):
    """
    Construct Gaussian spectral response functions.

    Parameters
    ----------
    B_h : int
        Number of hyperspectral bands.

    B_m : int
        Number of multispectral bands.

    centers : np.ndarray
        Spectral centers of the MSI response functions.
        Shape:

            (B_m,).

    sigma : float, optional
        Standard deviation of the Gaussian response functions.

    Returns
    -------
    R : np.ndarray
        Spectral response matrix with shape

            (B_m, B_h).

        Each row corresponds to one MSI spectral response.
    """
    lambda_h = np.arange(B_h)

    R_mat = np.zeros(
        (B_m, B_h),
        dtype=np.float64
    )

    for i in range(B_m):
        R_mat[i] = np.exp(
            -0.5
            * (
                (lambda_h - centers[i])
                / sigma
            ) ** 2
        )

    R_mat /= (
        R_mat.sum(
            axis=1,
            keepdims=True
        )
        + 1e-12
    )

    return R_mat


def build_spectral_cost(
    m,
    h,
    centers
):
    """
    Construct the spectral cost between MSI and HSI pixels.

    The HSI spectra are first projected into the MSI spectral
    domain using Gaussian spectral response functions. The
    spectral discrepancy is then measured using SAM.

    Parameters
    ----------
    m : np.ndarray
        Multispectral image with shape

            (H_m, W_m, B_m).

    h : np.ndarray
        Hyperspectral image with shape

            (H_h, W_h, B_h).

    centers : np.ndarray
        Spectral centers used to construct the MSI response.
        Shape:

            (B_m,).

    Returns
    -------
    C_spectral : np.ndarray
        Spectral cost matrix with shape

            (N_m, N_h),

        where

            N_m = H_m * W_m
            N_h = H_h * W_h.

        Each entry compares one MSI pixel with one HSI pixel.
    """
    C_m = m.shape[2]
    C_h = h.shape[2]

    A = m.reshape(
        -1,
        C_m
    )

    B = h.reshape(
        -1,
        C_h
    )

    R = gaussian_response(
        C_h,
        C_m,
        centers
    )

    B_proj = B @ R.T

    return sam_cost(
        A,
        B_proj
    )


def build_pixel_band_cost(
    M_cost_pix,
    C_m,
    C_h
):
    """
    Extend a pixel-level cost to the pixel-band domain.

    Parameters
    ----------
    M_cost_pix : np.ndarray
        Pixel-level spectral cost matrix with shape

            (N_m, N_h).

    C_m : int
        Number of MSI bands.

    C_h : int
        Number of HSI bands.

    Returns
    -------
    M_cost : np.ndarray
        Pixel-band cost matrix with shape

            (N_m * C_m, N_h * C_h).

    Notes
    -----
    The FGW transport problem is defined between MSI
    pixel-band elements and HSI pixel-band elements.
    """
    M_cost = np.repeat(
        np.repeat(
            M_cost_pix,
            C_h,
            axis=1
        ),
        C_m,
        axis=0
    )

    return M_cost / (
        M_cost.max()
        + 1e-12
    )


def solve_efgw(
    M_cost,
    C1,
    C2,
    p,
    q,
    epsilon=1e-1
):
    """
    Solve the entropic Fused Gromov-Wasserstein problem.

    Parameters
    ----------
    M_cost : np.ndarray
        Feature cost matrix between the two pixel-band domains.

        Shape:

            (N_m * C_m, N_h * C_h).

    C1 : np.ndarray
        Structural distance matrix of the MSI pixel-band domain.

        Shape:

            (N_m * C_m, N_m * C_m).

    C2 : np.ndarray
        Structural distance matrix of the HSI pixel-band domain.

        Shape:

            (N_h * C_h, N_h * C_h).

    p : np.ndarray
        Probability distribution over the MSI pixel-band domain.

        Shape:

            (N_m * C_m,).

        Must be non-negative and sum to one.

    q : np.ndarray
        Probability distribution over the HSI pixel-band domain.

        Shape:

            (N_h * C_h,).

        Must be non-negative and sum to one.

    epsilon : float, optional
        Entropic regularization parameter.

    Returns
    -------
    T : np.ndarray
        Optimal transport plan with shape

            (N_m * C_m, N_h * C_h).

    log : dict
        Optimization information returned by POT.

    Notes
    -----
    The structural matrices are normalized internally before
    solving the FGW problem.
    """
    C1_n = C1 / (
        C1.max()
        + 1e-12
    )

    C2_n = C2 / (
        C2.max()
        + 1e-12
    )

    T, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M=M_cost,
        C1=C1_n,
        C2=C2_n,
        p=p,
        q=q,
        loss_fun="square_loss",
        epsilon=epsilon,
        log=True,
        verbose=True
    )

    return T, log

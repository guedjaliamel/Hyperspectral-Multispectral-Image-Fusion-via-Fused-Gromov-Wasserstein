"""
Evaluation metrics for hyperspectral image fusion.
"""

import numpy as np


# ============================================================
# 1. SAM
# ============================================================

def spectral_angle_mapper(I, F):
    """
    Compute the mean Spectral Angle Mapper (SAM).

    Parameters
    ----------
    I : np.ndarray
        Reference hyperspectral image, shape (H, W, B).

    F : np.ndarray
        Estimated hyperspectral image, shape (H, W, B).

    Returns
    -------
    float
        Mean SAM in radians.
    """

    nl, nc, nb = I.shape

    I_flat = I.reshape(-1, nb)
    F_flat = F.reshape(-1, nb)

    dot = np.sum(
        I_flat * F_flat,
        axis=1
    )

    norm_I = np.linalg.norm(
        I_flat,
        axis=1
    )

    norm_F = np.linalg.norm(
        F_flat,
        axis=1
    )

    cos_theta = dot / (
        norm_I * norm_F + 1e-12
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0
    )

    theta = np.arccos(
        cos_theta
    )

    return np.mean(theta)


# ============================================================
# 2. CC
# ============================================================

def correlation_coefficient(I, F):
    """
    Compute the global correlation coefficient.

    Parameters
    ----------
    I : np.ndarray
        Reference hyperspectral image, shape (H, W, B).

    F : np.ndarray
        Estimated hyperspectral image, shape (H, W, B).

    Returns
    -------
    float
        Global correlation coefficient.
    """

    I_flat = I.reshape(-1)
    F_flat = F.reshape(-1)

    return np.corrcoef(
        I_flat,
        F_flat
    )[0, 1]


# ============================================================
# 3. R-SNR
# ============================================================

def relative_snr(I, F):
    """
    Compute the Relative Signal-to-Noise Ratio.

    Parameters
    ----------
    I : np.ndarray
        Reference hyperspectral image, shape (H, W, B).

    F : np.ndarray
        Estimated hyperspectral image, shape (H, W, B).

    Returns
    -------
    float
        R-SNR in dB.
    """

    num = 0.0
    denum = 0.0

    for k in range(I.shape[2]):

        I_band = I[:, :, k].reshape(-1)

        F_band = F[:, :, k].reshape(-1)

        num += (
            np.linalg.norm(I_band) ** 2
        )

        denum += (
            np.linalg.norm(
                F_band - I_band
            ) ** 2
        )

    return 10 * np.log10(
        num / denum
    )


# ============================================================
# 4. ERGAS
# ============================================================

def ergas(I, F, d1, d2):
    """
    Compute ERGAS.

    Parameters
    ----------
    I : np.ndarray
        Reference hyperspectral image, shape (H, W, B).

    F : np.ndarray
        Estimated hyperspectral image, shape (H, W, B).

    d1 : float
        First spatial undersampling factor.

    d2 : float
        Second spatial undersampling factor.

    Returns
    -------
    float
        ERGAS value.
    """

    ergas_sum = 0.0

    for k in range(I.shape[2]):

        I_band = I[:, :, k]
        F_band = F[:, :, k]

        mu = np.mean(I_band)

        error = F_band - I_band

        ergas_sum += (
            np.linalg.norm(
                error.reshape(-1)
            ) ** 2
            / (mu ** 2)
        )

    return (
        100
        * (1 / np.sqrt(d1 * d2))
        * np.sqrt(
            ergas_sum / I.size
        )
    )


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate(
    I_patch,
    F_patch,
    d1=1,
    d2=1
):
    """
    Evaluate a fused hyperspectral image.

    The two cubes are first normalized globally
    by their sum, as in the original evaluation code.

    Parameters
    ----------
    I_patch : np.ndarray
        Reference hyperspectral image, shape (H, W, B).

    F_patch : np.ndarray
        Estimated hyperspectral image, shape (H, W, B).

    d1 : float, optional
        First spatial undersampling factor.
        Default is 1.

    d2 : float, optional
        Second spatial undersampling factor.
        Default is 1.

    Returns
    -------
    dict
        Dictionary containing:

        - SAM
        - CC
        - R-SNR
        - ERGAS
    """

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if I_patch.shape != F_patch.shape:

        raise ValueError(
            f"Dimensions différentes : "
            f"I={I_patch.shape}, "
            f"F={F_patch.shape}"
        )

    # --------------------------------------------------------
    # Global normalization
    # --------------------------------------------------------

    I_norm = (
        I_patch
        / np.sum(I_patch)
    )

    F_norm = (
        F_patch
        / np.sum(F_patch)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    sam_val = spectral_angle_mapper(
        I_norm,
        F_norm
    )

    cc_val = correlation_coefficient(
        I_norm,
        F_norm
    )

    snr_val = relative_snr(
        I_norm,
        F_norm
    )

    ergas_val = ergas(
        I_norm,
        F_norm,
        d1,
        d2
    )

    return {
        "SAM": sam_val,
        "CC": cc_val,
        "R-SNR": snr_val,
        "ERGAS": ergas_val
    }
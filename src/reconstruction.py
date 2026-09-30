"""
Reconstruction of the fused hyperspectral image from the transport plan.
"""


def reconstruct_from_transport(
    T,
    m_shape,
    h_shape
):
    """
    Reconstruct a fused hyperspectral image from an FGW transport plan.

    Parameters
    ----------
    T : np.ndarray
        Optimal transport plan defined on the pixel-band domains.

        Shape:

            (N_m * C_m, N_h * C_h)

        where:
            N_m = H_m * W_m
            C_m = number of MSI bands
            N_h = H_h * W_h
            C_h = number of HSI bands.

    m_shape : tuple of int
        Shape of the MSI:

            (H_m, W_m, C_m).

    h_shape : tuple of int
        Shape of the HSI:

            (H_h, W_h, C_h).

    Returns
    -------
    fused : np.ndarray
        Reconstructed fused hyperspectral image with shape

            (H_m, W_m, C_h).

        The fused image has the spatial resolution of the MSI
        and the spectral resolution of the HSI.

    Notes
    -----
    The transport plan is first reshaped as

        T4[i, k, j, l]

    where:

        i = MSI spatial pixel
        k = MSI spectral band
        j = HSI spatial pixel
        l = HSI spectral band.

    The reconstruction is obtained by

        F[i, l] = sum_k sum_j T4[i, k, j, l].

    Thus, the MSI spectral dimension and HSI spatial dimension
    are marginalized.
    """
    H_m, W_m, C_m = m_shape
    H_h, W_h, C_h = h_shape

    N_m = H_m * W_m
    N_h = H_h * W_h

    T4 = T.reshape(
        N_m,
        C_m,
        N_h,
        C_h
    )

    # Marginalization over the MSI spectral dimension.
    W = T4.sum(axis=1)

    # Marginalization over the HSI spatial dimension.
    F = W.sum(axis=1)

    return F.reshape(
        H_m,
        W_m,
        C_h
    )

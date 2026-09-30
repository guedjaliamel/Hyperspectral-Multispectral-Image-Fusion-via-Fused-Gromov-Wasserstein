import numpy as np
import os
import matplotlib.pyplot as plt


# ============================================================
# 1. COMPARAISON DES BANDES
# ============================================================

def plot_all_bands_comparison(
    I,
    m,
    h,
    f_efgw,
    f_hmwb,
    bands_m,
    bands_h,
    save_dir,
    dataset,
    cmap="gray"
):
    """
    Generate one figure per spectral-band pair.

    Each figure contains:

        Reference | MSI | HSI | HMWB | EFGW

    Parameters
    ----------
    I : np.ndarray
        Reference hyperspectral image, shape (H, W, B_h).

    m : np.ndarray
        Multispectral image, shape (H_m, W_m, B_m).

    h : np.ndarray
        Hyperspectral low-spatial-resolution image,
        shape (H_h, W_h, B_h).

    f_efgw : np.ndarray
        Fused hyperspectral image reconstructed with EFGW.

    f_hmwb : np.ndarray
        Fused hyperspectral image reconstructed with HMWB.

    bands_m : list of int
        MSI bands to display.

    bands_h : list of int
        Corresponding hyperspectral bands to display.

    save_dir : str
        Output directory.

    dataset : str
        Dataset name used in output filenames.

    cmap : str
        Matplotlib colormap.
    """

    os.makedirs(save_dir, exist_ok=True)

    if len(bands_m) != len(bands_h):
        raise ValueError(
            "bands_m and bands_h must have the same length."
        )

    # Check MSI band indices
    for band_m in bands_m:
        if band_m < 0 or band_m >= m.shape[2]:
            raise IndexError(
                f"Invalid MSI band {band_m}. "
                f"Number of MSI bands = {m.shape[2]}"
            )

    # Check HSI band indices
    for band_h in bands_h:
        if band_h < 0 or band_h >= h.shape[2]:
            raise IndexError(
                f"Invalid HSI band {band_h}. "
                f"Number of HSI bands = {h.shape[2]}"
            )

    # Check fused images
    for band_h in bands_h:

        if band_h >= f_efgw.shape[2]:
            raise IndexError(
                f"Invalid EFGW band {band_h}. "
                f"Number of EFGW bands = {f_efgw.shape[2]}"
            )

        if band_h >= f_hmwb.shape[2]:
            raise IndexError(
                f"Invalid HMWB band {band_h}. "
                f"Number of HMWB bands = {f_hmwb.shape[2]}"
            )

    # ========================================================
    # Generate figures
    # ========================================================

    for band_m, band_h in zip(bands_m, bands_h):

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 5)
        )

        panels = [
            I[:, :, band_h],
            m[:, :, band_m],
            h[:, :, band_h],
            f_hmwb[:, :, band_h],
            f_efgw[:, :, band_h]
        ]

        titles = [
            "Reference",
            "MSI",
            "HSI",
            "HMWB",
            "EFGW"
        ]

        for ax, img, title in zip(
            axes,
            panels,
            titles
        ):

            ax.imshow(
                img,
                cmap=cmap,
                interpolation="nearest"
            )

            ax.set_title(
                title,
                fontsize=13,
                fontweight="bold"
            )

            ax.axis("off")

        plt.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=0.88,
            wspace=0.02,
            hspace=0
        )

        filename = (
            f"{dataset}_EFGW_"
            f"m{band_m:02d}_"
            f"h{band_h:02d}.png"
        )

        filepath = os.path.join(
            save_dir,
            filename
        )

        plt.savefig(
            filepath,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300
        )

        plt.close(fig)

        print(
            f"Figure saved: {filepath}"
        )


# ============================================================
# 2. NORMALISATION
# ============================================================

def minmax_scale(cube):
    """
    Apply global min-max normalization to a cube.
    """

    cube = cube.astype(np.float64)

    return (
        (cube - cube.min())
        / (cube.max() - cube.min() + 1e-12)
    )


# ============================================================
# 3. SIGNATURES SPECTRALES
# ============================================================

def plot_spectral_signatures(
    cubes,
    pixels,
    wl_min,
    wl_max,
    labels,
    save_path
):
    """
    Plot spectral signatures for several hyperspectral cubes.

    Parameters
    ----------
    cubes : list of np.ndarray
        Hyperspectral cubes to compare.

    pixels : list of tuple
        Pixel coordinates defined in the reference image.

    wl_min : float
        Minimum wavelength in nm.

    wl_max : float
        Maximum wavelength in nm.

    labels : list of str
        Labels corresponding to the cubes.

    save_path : str
        Output PDF path.
    """

    if len(cubes) != len(labels):
        raise ValueError(
            "cubes and labels must have the same length."
        )

    # --------------------------------------------------------
    # Normalize cubes
    # --------------------------------------------------------

    cubes_s = [
        minmax_scale(cube)
        for cube in cubes
    ]

    # --------------------------------------------------------
    # Reference dimensions
    # --------------------------------------------------------

    H_ref, W_ref, _ = cubes_s[0].shape

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(cubes_s),
        figsize=(8, 2.8)
    )

    if len(cubes_s) == 1:
        axes = [axes]

    # ========================================================
    # Plot each cube
    # ========================================================

    for ax, cube, label in zip(
        axes,
        cubes_s,
        labels
    ):

        H_c, W_c, C_c = cube.shape

        wavelengths = np.linspace(
            wl_min,
            wl_max,
            C_c
        )

        for i, j in pixels:

            ii = min(
                int(i * H_c / H_ref),
                H_c - 1
            )

            jj = min(
                int(j * W_c / W_ref),
                W_c - 1
            )

            ax.plot(
                wavelengths,
                cube[ii, jj, :],
                linewidth=2.0,
                label=f"({i},{j})"
            )

        ax.set_title(
            label,
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Wavelength (nm)",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_ylabel(
            "Reflectance",
            fontsize=12,
            fontweight="bold"
        )

        ax.tick_params(
            axis="both",
            labelsize=10,
            width=1.2
        )

        ax.set_ylim(
            -0.05,
            1.05
        )

        ax.grid(
            True,
            alpha=0.3,
            linewidth=0.8
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.legend(
            fontsize=8,
            ncol=1,
            frameon=True,
            handlelength=2.0,
            labelspacing=0.3,
            borderpad=0.5,
            loc="upper right"
        )

    plt.tight_layout()

    save_directory = os.path.dirname(save_path)

    if save_directory:
        os.makedirs(
            save_directory,
            exist_ok=True
        )

    plt.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Spectral signatures saved: {save_path}"
    )
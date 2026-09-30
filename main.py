"""
Main entry point for hyperspectral-multispectral image fusion.

The user only needs to change DATASET to select the experiment.
All dataset-specific parameters are automatically selected.
"""

import os
import time

import numpy as np

from src.data import (
    load_pavia,
    load_indian_pines,
    extract_patch,
    generate_hsi,
    generate_msi,
)

from src.fgw import (
    build_spectral_cost,
    build_pixel_band_cost,
    distance_matrix,
    solve_efgw,
)

from src.reconstruction import (
    reconstruct_from_transport,
)

from src.hmwb import (
    run_hmwb,
)

from src.evaluation import (
    evaluate,
)

from src.visualization import (
    plot_all_bands_comparison,
    plot_spectral_signatures,
)


# =========================================================================
# DATASET SELECTION
# =========================================================================

DATASET = "indian_pines"

# Available options:
#
# DATASET = "pavia"
# DATASET = "indian_pines"


# =========================================================================
# GENERAL PARAMETERS
# =========================================================================

N_MSI_BANDS = 6
MSI_SIGMA = 5

HSI_KERNEL_SIZE = 5
HSI_SIGMA = 2

EPSILON = 1e-1


# =========================================================================
# DATASET-SPECIFIC PARAMETERS
# =========================================================================

DATASETS = {

    "pavia": {

        "loader": load_pavia,

        "patch": {
            "row": 150,
            "col": 100,
            "height": 50,
            "width": 50,
        },

        "downsample": 4,

        "eta_m": 3.8893,
        "eta_h": 4.7697,

        "wavelengths": (430, 860),

        "bands_m": [
            0, 1, 2, 3, 4, 5
        ],

        "bands_h": [
            9, 25, 42, 58, 75, 92
        ],

        "signature_pixels": [
            (39, 10),
            (39, 17),
            (10, 39),
            (15, 30),
        ],
    },


    "indian_pines": {

        "loader": load_indian_pines,

        "patch": {
            "row": 40,
            "col": 80,
            "height": 40,
            "width": 40,
        },

        "downsample": 4,

        "eta_m": 1.5789,
        "eta_h": 3.3002,

        "wavelengths": (400, 2500),

        "bands_m": [
            0, 1, 2, 3, 4, 5
        ],

        "bands_h": [
            10, 46, 82, 118, 154, 190
        ],

        "signature_pixels": [
            (38, 28),
            (2, 30),
            (20, 20),
            (30, 30),
        ],
    },
}


def main():
    """
    Run the complete hyperspectral-multispectral fusion experiment.

    The experiment performs:

    1. Dataset loading.
    2. Reference patch extraction.
    3. Synthetic MSI and HSI generation.
    4. EFGW cost construction.
    5. EFGW optimization.
    6. Fused HSI reconstruction.
    7. HMWB baseline computation.
    8. Quantitative evaluation.
    9. Fused image saving.
    10. Metric saving.
    11. Band comparison visualization.
    12. Spectral signature visualization.
    """

    # =========================================================================
    # CHECK DATASET
    # =========================================================================

    if DATASET not in DATASETS:

        raise ValueError(
            f"Unknown dataset: {DATASET}. "
            f"Available datasets: {list(DATASETS.keys())}"
        )

    config = DATASETS[DATASET]


    # =========================================================================
    # OUTPUT DIRECTORIES
    # =========================================================================

    results_dir = os.path.join("results", DATASET)
    figures_dir = os.path.join(results_dir, "figures")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)


    # =========================================================================
    # HEADER
    # =========================================================================

    print("=" * 70)
    print("Hyperspectral-Multispectral Image Fusion")
    print("EFGW vs HMWB")
    print("=" * 70)

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Results directory: {results_dir}"
    )


    # =========================================================================
    # 1. LOAD REFERENCE IMAGE
    # =========================================================================

    print("\n" + "=" * 70)
    print("Loading reference image")
    print("=" * 70)

    reference_image = config["loader"]()

    print(
        f"Original image shape: "
        f"{reference_image.shape}"
    )

    patch = config["patch"]

    reference = extract_patch(
        reference_image,
        row=patch["row"],
        col=patch["col"],
        height=patch["height"],
        width=patch["width"],
    )

    reference = reference.astype(
        np.float64
    )

    print(
        f"Reference patch shape: "
        f"{reference.shape}"
    )


    # =========================================================================
    # 2. GENERATE SYNTHETIC OBSERVATIONS
    # =========================================================================

    print("\n" + "=" * 70)
    print("Generating synthetic observations")
    print("=" * 70)

    hsi = generate_hsi(
        reference,
        kernel_size=HSI_KERNEL_SIZE,
        sigma=HSI_SIGMA,
        downsample=config["downsample"],
        noise_std=0,
    )

    msi, spectral_response, centers = generate_msi(
        reference,
        n_bands=N_MSI_BANDS,
        noise_std=0,
        sigma=MSI_SIGMA,
    )

    hsi = hsi.astype(
        np.float64
    )

    msi = msi.astype(
        np.float64
    )

    # Same normalization used in the original experiments.
    normalization = (
        msi.max()
        + 1e-12
    )

    msi /= normalization
    hsi /= normalization

    print(
        f"MSI shape: "
        f"{msi.shape}"
    )

    print(
        f"HSI shape: "
        f"{hsi.shape}"
    )


    # =========================================================================
    # 3. BUILD EFGW COSTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("Building EFGW costs")
    print("=" * 70)

    start_efgw = time.perf_counter()

    # -------------------------------------------------------------------------
    # Spectral cost
    # -------------------------------------------------------------------------

    spectral_cost = build_spectral_cost(
        msi,
        hsi,
        centers
    )

    print(
        f"Pixel spectral cost shape: "
        f"{spectral_cost.shape}"
    )

    # -------------------------------------------------------------------------
    # Pixel-band cost
    # -------------------------------------------------------------------------

    pixel_band_cost = build_pixel_band_cost(
        spectral_cost,
        msi.shape[2],
        hsi.shape[2]
    )

    print(
        f"Pixel-band cost shape: "
        f"{pixel_band_cost.shape}"
    )

    # -------------------------------------------------------------------------
    # Structural matrices
    # -------------------------------------------------------------------------

    C1 = distance_matrix(
        msi,
        alpha=config["eta_m"]
    )

    C2 = distance_matrix(
        hsi,
        alpha=config["eta_h"]
    )

    print(
        f"MSI structural matrix shape: "
        f"{C1.shape}"
    )

    print(
        f"HSI structural matrix shape: "
        f"{C2.shape}"
    )


    # =========================================================================
    # 4. DEFINE TRANSPORT MASSES
    # =========================================================================

    print("\n" + "=" * 70)
    print("Defining transport masses")
    print("=" * 70)

    p = msi.reshape(
        -1
    ).astype(
        np.float64
    )

    p /= (
        p.sum()
        + 1e-12
    )

    q = hsi.reshape(
        -1
    ).astype(
        np.float64
    )

    q /= (
        q.sum()
        + 1e-12
    )

    print(
        f"MSI mass vector shape: "
        f"{p.shape}"
    )

    print(
        f"HSI mass vector shape: "
        f"{q.shape}"
    )


    # =========================================================================
    # 5. SOLVE EFGW
    # =========================================================================

    print("\n" + "=" * 70)
    print("Solving EFGW")
    print("=" * 70)

    T, log = solve_efgw(
        pixel_band_cost,
        C1,
        C2,
        p,
        q,
        epsilon=EPSILON
    )


    # =========================================================================
    # 6. RECONSTRUCT FUSED HSI
    # =========================================================================

    fused_efgw = reconstruct_from_transport(
        T,
        m_shape=msi.shape,
        h_shape=hsi.shape
    )

    time_efgw = (
        time.perf_counter()
        - start_efgw
    )

    print(
        f"Transport plan shape: "
        f"{T.shape}"
    )

    print(
        f"EFGW output shape: "
        f"{fused_efgw.shape}"
    )

    print(
        f"EFGW total time: "
        f"{time_efgw:.2f} s"
    )


    # =========================================================================
    # 7. RUN HMWB BASELINE
    # =========================================================================

    print("\n" + "=" * 70)
    print("Running HMWB")
    print("=" * 70)

    start_hmwb = time.perf_counter()

    fused_hmwb, hmwb_core_time = run_hmwb(
        reference,
        msi,
        hsi,
        spectral_response
    )

    time_hmwb = (
        time.perf_counter()
        - start_hmwb
    )

    print(
        f"HMWB output shape: "
        f"{fused_hmwb.shape}"
    )

    print(
        f"HMWB total time: "
        f"{time_hmwb:.2f} s"
    )

    print(
        f"HMWB optimization time: "
        f"{hmwb_core_time:.2f} s"
    )


    # =========================================================================
    # 8. EVALUATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)

    metrics_efgw = evaluate(
        reference,
        fused_efgw,
        d1=4,
        d2=4
    )

    metrics_hmwb = evaluate(
        reference,
        fused_hmwb,
        d1=4,
        d2=4
    )

    print("\nEFGW:")

    for name, value in metrics_efgw.items():

        print(
            f"  {name}: {value:.6f}"
        )

    print("\nHMWB:")

    for name, value in metrics_hmwb.items():

        print(
            f"  {name}: {value:.6f}"
        )


    # =========================================================================
    # 9. SAVE FUSED IMAGES
    # =========================================================================

    efgw_path = os.path.join(
        results_dir,
        f"{DATASET}_efgw.npy"
    )

    hmwb_path = os.path.join(
        results_dir,
        f"{DATASET}_hmwb.npy"
    )

    np.save(
        efgw_path,
        fused_efgw
    )

    np.save(
        hmwb_path,
        fused_hmwb
    )

    print(
        f"\nEFGW result saved: {efgw_path}"
    )

    print(
        f"HMWB result saved: {hmwb_path}"
    )


    # =========================================================================
    # 10. SAVE METRICS
    # =========================================================================

    metrics_path = os.path.join(
        results_dir,
        "metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 70 + "\n")
        file.write(
            f"Dataset: {DATASET}\n"
        )
        file.write("=" * 70 + "\n\n")

        # ---------------------------------------------------------------------
        # EFGW
        # ---------------------------------------------------------------------

        file.write("EFGW\n")
        file.write("-" * 30 + "\n")

        for name, value in metrics_efgw.items():

            file.write(
                f"{name}: {value:.6f}\n"
            )

        file.write(
            f"Time: {time_efgw:.2f} s\n"
        )

        file.write("\n")

        # ---------------------------------------------------------------------
        # HMWB
        # ---------------------------------------------------------------------

        file.write("HMWB\n")
        file.write("-" * 30 + "\n")

        for name, value in metrics_hmwb.items():

            file.write(
                f"{name}: {value:.6f}\n"
            )

        file.write(
            f"Time: {time_hmwb:.2f} s\n"
        )

        file.write(
            f"HMWB optimization time: "
            f"{hmwb_core_time:.2f} s\n"
        )

        file.write("\n")

    print(
        f"Metrics saved: {metrics_path}"
    )


    # =========================================================================
    # 11. VISUAL COMPARISON
    # =========================================================================

    print("\n" + "=" * 70)
    print("Generating band comparison figures")
    print("=" * 70)

    plot_all_bands_comparison(
        reference,
        msi,
        hsi,
        fused_efgw,
        fused_hmwb,
        bands_m=config["bands_m"],
        bands_h=config["bands_h"],
        save_dir=figures_dir,
        dataset=DATASET
    )


    # =========================================================================
    # 12. SPECTRAL SIGNATURES
    # =========================================================================

    print("\n" + "=" * 70)
    print("Generating spectral signatures")
    print("=" * 70)

    wl_min, wl_max = config["wavelengths"]

    signatures_path = os.path.join(
        figures_dir,
        f"{DATASET}_signatures.pdf"
    )

    plot_spectral_signatures(
        cubes=[
            reference,
            fused_hmwb,
            fused_efgw
        ],
        pixels=config["signature_pixels"],
        wl_min=wl_min,
        wl_max=wl_max,
        labels=[
            "Reference",
            "HMWB",
            "EFGW"
        ],
        save_path=signatures_path
    )


    # =========================================================================
    # 13. END
    # =========================================================================

    print("\n" + "=" * 70)
    print("Experiment completed.")
    print("=" * 70)

    print(
        f"All results saved in: {results_dir}"
    )


if __name__ == "__main__":
    main()
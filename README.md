<h1 align="center">Hyperspectral–Multispectral Image Fusion<br>via Fused Gromov-Wasserstein</h1>

<p align="center">
  Official implementation of <b>"Fused Gromov-Wasserstein for Hyperspectral-Multispectral Image Fusion"</b><br>
  A. Guedjali, EH. Djermoune, P. Catala, S. Delchini — <i>submitted to ICASSP 2027</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/POT-Optimal%20Transport-orange" alt="POT">
  <img src="https://img.shields.io/badge/status-research%20code-lightgrey" alt="Status">
</p>

---

## Table of contents

- [Overview](#overview)
- [Method](#method)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Evaluation metrics](#evaluation-metrics)
- [Citation](#citation)
- [References](#references)
- [Funding](#funding)
- [Authors](#authors)

---

## Overview

Hyperspectral (HSI) and multispectral (MSI) images are complementary:

| | Spatial resolution | Spectral resolution |
|---|:---:|:---:|
| **HSI** | low | high (100+ bands) |
| **MSI** | high | low (a few bands) |

**Goal:** reconstruct a fused hyperspectral image that has the **spatial grid of the MSI** and the **spectral content of the HSI**.

**Idea:** an entropic **Fused Gromov-Wasserstein (FGW)** transport plan matches the voxels of the two images, *without requiring a common embedding space and without interpolation*. The plan is then used to transfer the HSI spectra onto the MSI spatial grid.

## Method

Let $n_1 = l_m c_m b_m$ and $n_2 = l_h c_h b_h$ be the numbers of voxels of M and H. The optimal transport plan $P \in \mathbb{R}_+^{n_1 \times n_2}$ coupling the marginals $\mu$ and $\nu$ minimizes:

```math
\mathrm{FGW}_{\alpha,\varepsilon}(\mu,\nu) = \min_{P \in \Pi(\mu,\nu)} (1-\alpha)\langle C, P \rangle + \alpha \sum_{q,q',r,r'} |A_{qq'} - B_{rr'}|^2 P_{qr} P_{q'r'} - \varepsilon H(P)
```

where $\alpha \in [0,1]$ balances the spectral term against the structural term, $(q,q')$ index voxels of M and $(r,r')$ voxels of H.

| Term | Role |
|---|---|
| **C** ∈ ℝ<sup>n₁×n₂</sup> — spectral cost | Spectral Angle Mapper (SAM) between MSI spectra and HSI spectra projected onto the MSI bands |
| **A** ∈ ℝ<sup>n₁×n₁</sup>, **B** ∈ ℝ<sup>n₂×n₂</sup> — structure matrices | Intra-image spatial-spectral distances (spatial coordinates + η-weighted spectral coordinate) |
| **ε** | Entropic regularization (solved with Sinkhorn iterations, via [POT](https://pythonot.github.io/)) |

**Spectral cost.** With $s_i$ the spectrum of MSI pixel $i$ and $\tilde{s}_j = D h_j$ the spectrum of HSI pixel $j$ projected onto the MSI bands by the spectral response $D$:

```math
C_{ij} = \arccos\left(\frac{\langle s_i, \tilde{s}_j \rangle}{\lVert s_i \rVert \lVert \tilde{s}_j \rVert}\right)
```

**Structure matrices.** For two pixels $i$ and $i'$ with spatial coordinates $(u,v)$ and spectral coordinate $z$:

```math
d(i,i') = (u_i - u_{i'})^2 + (v_i - v_{i'})^2 + \eta\,(z_i - z_{i'})^2
```

where $\eta > 0$ balances the spectral contribution relative to the spatial distance (this is the parameter η<sub>M</sub> / η<sub>H</sub> in the [Configuration](#configuration) table).

> **Third-party code.** The entropic FGW solver is **not our own implementation**: it relies on the [POT (Python Optimal Transport)](https://github.com/PythonOT/POT) library ([documentation](https://pythonot.github.io/)), specifically [`ot.gromov.entropic_fused_gromov_wasserstein`](https://pythonot.github.io/_modules/ot/gromov/_bregman.html#entropic_fused_gromov_wasserstein). Our contribution is the construction of the cost and structure matrices (C, A, B) for hyperspectral/multispectral images and the reconstruction from the transport plan.

**Reconstruction.** The optimal plan $P^\star \in \mathbb{R}_+^{(n_m b_m)\times(n_h b_h)}$ is reshaped into a tensor $T \in \mathbb{R}_+^{n_m \times b_m \times n_h \times b_h}$, where $T_{ikjl}$ is the mass transported between MSI voxel $(k,i)$ and HSI voxel $(l,j)$. The fused image is obtained by marginalizing over the MSI spectral dimension and the HSI spatial dimension:

```math
F_{il} = \sum_{k=1}^{b_m} \sum_{j=1}^{n_h} T_{ikjl}, \quad i = 1,\dots,n_m, \; l = 1,\dots,b_h
```

The matrix $F \in \mathbb{R}_+^{n_m \times b_h}$ is reshaped to the MSI spatial grid ($n_m = l_m c_m$), giving the fused cube of size $l_m \times c_m \times b_h$: the spatial resolution comes from M and the spectral dimension from H, without interpolation.

## Quick start

```bash
git clone https://github.com/guedjaliamel/Hyperspectral-Multispectral-Image-Fusion-via-Fused-Gromov-Wasserstein.git
cd Hyperspectral-Multispectral-Image-Fusion-via-Fused-Gromov-Wasserstein
pip install -r requirements.txt
```

1. Select the dataset at the top of `main.py` (the data are already provided in `data/`):

   ```python
   DATASET = "pavia"          # or "indian_pines"
   ```

2. Run:

   ```bash
   python main.py
   ```

Dataset-specific parameters are selected automatically; **no modification is needed to reproduce the reported experiments.** Outputs (reconstructions, metrics, figures) are written to `results/<dataset>/`.

**Requirements:** NumPy, SciPy, Matplotlib, scikit-image, POT.

## Datasets

Both datasets are already included in the `data/` folder. Original sources are given for reference.

| Dataset | Bands | Extract used | File | Original source |
|---|:---:|:---:|---|---|
| Pavia University | 103 | 50 × 50 | `data/PaviaU.mat` | [Kaggle](https://www.kaggle.com/code/ardaorcun/hyperspectral-paviau?select=PaviaU.mat) |
| Indian Pines | 200 | 40 × 40 | `data/indianpinearray.npy` | [Kaggle](https://www.kaggle.com/datasets/abhijeetgo/indian-pines-hyperspectral-dataset/data?select=indianpinearray.npy) |

**Simulation protocol.** The extract is the reference image. The HSI is obtained by Gaussian low-pass filtering, spatial subsampling and additive Gaussian noise. The MSI (6 bands) is obtained by Gaussian spectral response and additive noise. Both images are normalized to sum to 1.

## Configuration

Shared by both datasets: **α = 0.5**, **ε = 1e-1**.

| Parameter | Pavia University | Indian Pines |
|---|---:|---:|
| MSI bands | 6 | 6 |
| MSI spectral response σ | 5 | 5 |
| HSI Gaussian kernel size | 5 | 5 |
| HSI Gaussian σ | 2 | 2 |
| Spatial downsampling | 4 | 4 |
| FGW ε | 1e-1 | 1e-1 |
| MSI structural parameter η<sub>M</sub> | 3.8893 | 1.5789 |
| HSI structural parameter η<sub>H</sub> | 4.7697 | 3.3002 |

All parameters can be changed in `main.py` for custom experiments.

## Results

FGW is compared with **HMWB** and **B-SCOTT**. **Bold** = best. FGW obtains the best CC and ERGAS on Pavia University, and the best value on all four metrics on Indian Pines.

### Pavia University

| Method | SAM (rad) ↓ | CC ↑ | R-SNR (dB) ↑ | ERGAS ↓ |
|---|---:|---:|---:|---:|
| **FGW** (ours) | 0.16 | **0.97** | 17.17 | **4.01** |
| B-SCOTT | **0.10** | 0.95 | **17.82** | 4.02 |
| HMWB | 0.25 | 0.90 | 12.61 | 6.52 |

### Indian Pines

| Method | SAM (rad) ↓ | CC ↑ | R-SNR (dB) ↑ | ERGAS ↓ |
|---|---:|---:|---:|---:|
| **FGW** (ours) | **0.07** | **0.99** | **22.96** | **2.06** |
| B-SCOTT | 0.12 | 0.59 | 4.26 | 13.02 |
| HMWB | 0.61 | 0.62 | 2.79 | 18.28 |

### Visual results

Each figure shows, from left to right: reference, MSI, HSI, HMWB, FGW.

#### Pavia University

![Pavia University band 25](results/pavia/figures/pavia_EFGW_m01_h25.png)

*Spectral band 25 (MSI band 1)*

![Pavia University band 75](results/pavia/figures/pavia_EFGW_m04_h75.png)

*Spectral band 75 (MSI band 4)*

#### Indian Pines

![Indian Pines band 46](results/indian_pines/figures/indian_pines_EFGW_m01_h46.png)

*Spectral band 46 (MSI band 1)*

![Indian Pines band 118](results/indian_pines/figures/indian_pines_EFGW_m03_h118.png)

*Spectral band 118 (MSI band 3)*

### Spectral signatures

Spectra of selected pixels (reference vs. HMWB vs. FGW).

#### Pavia University

![Pavia University spectral signatures](results/pavia/figures/pavia_signatures.png)

#### Indian Pines

![Indian Pines spectral signatures](results/indian_pines/figures/indian_pines_signatures.png)

## Repository structure

```text
.
├── main.py                 # Entry point: dataset selection, parameters, full pipeline
├── requirements.txt
├── src/
│   ├── data.py             # Loading, simulation of MSI/HSI, normalization
│   ├── fgw.py              # Cost/structure matrices and entropic FGW solver (POT)
│   ├── reconstruction.py   # Transport plan → fused image (marginalization)
│   ├── hmwb.py             # HMWB baseline
│   ├── evaluation.py       # SAM, CC, R-SNR, ERGAS
│   └── visualization.py    # Figures and spectral signatures
├── data/                   # PaviaU.mat, indianpinearray.npy
└── results/
    ├── pavia/              # pavia_efgw.npy, pavia_hmwb.npy, metrics.txt, figures/
    └── indian_pines/       # indian_pines_efgw.npy, indian_pines_hmwb.npy, metrics.txt, figures/
```

The method is called **FGW** in the paper and this README; some output filenames keep the internal `efgw` (entropic FGW) name.
B-SCOTT results come from its original implementation (see [References](#references)) and are not recomputed by `main.py`.

## Evaluation metrics

| Metric | Meaning | Better |
|---|---|:---:|
| **SAM** | Spectral Angle Mapper (rad) | lower |
| **CC** | Cross-correlation | higher |
| **R-SNR** | Reconstruction signal-to-noise ratio (dB) | higher |
| **ERGAS** | Relative dimensionless global synthesis error | lower |

## Citation

The associated paper is currently under review. If you use this code in the meantime, please refer to this repository and to the paper title:

> A. Guedjali, EH. Djermoune, P. Catala and S. Delchini, *Fused Gromov-Wasserstein for Hyperspectral-Multispectral Image Fusion*, submitted to ICASSP 2027.

A BibTeX entry will be added here once the paper is accepted.

## References

- R. Flamary et al., *POT: Python Optimal Transport*, Journal of Machine Learning Research, 2021. Code: [PythonOT/POT](https://github.com/PythonOT/POT)
- T. Vayer et al., *Fused Gromov-Wasserstein distance for structured objects: theoretical foundations and mathematical properties*. [arXiv:1811.02834](https://arxiv.org/abs/1811.02834)
- M. Mifdal et al., *Hyperspectral image fusion using Wasserstein barycenters* (HMWB). [HAL 01620601v1](https://hal.science/hal-01620601)
- C. Prévost, K. Usevich, P. Comon and D. Brie, *Hyperspectral Super-Resolution with Coupled Tucker Approximation: Recoverability and SVD-based Algorithms* (B-SCOTT), IEEE Transactions on Signal Processing, 2020. [HAL hal-01911969](https://hal.science/hal-01911969) — implementation: [cprevost4/HSR_Software](https://github.com/cprevost4/HSR_Software)

## Funding

Supported by the **ANR France 2030 – PEPR Sous-sol, InnovTech project**, grant **ANR-22-EXSS-0006**.

## Authors

| | Affiliation |
|---|---|
| **Amel Guedjali** | Université de Lorraine, CNRS, CRAN, France |
| **El-Hadi Djermoune** | Université de Lorraine, CNRS, CRAN, France |
| **Paul Catala** | Université de Lorraine, CNRS, CRAN, France |
| **Sylvain Delchini** | BRGM – French Geological Survey, Orléans, France |

---

<sub>Copyright © 2026 Amel Guedjali, El-Hadi Djermoune, Paul Catala, Sylvain Delchini.</sub>
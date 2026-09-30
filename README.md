# Hyperspectral-Multispectral Image Fusion via Fused Gromov-Wasserstein

Copyright (c) 2026 Amel Guedjali, El-Hadi Djermoune, Paul Catala, Sylvain Delchini

This repository contains the implementation of **hyperspectral-multispectral image fusion using Fused Gromov-Wasserstein (FGW) optimal transport**.

The proposed approach exploits both **spectral information** and **spatial-spectral structure** to reconstruct a high-spatial-resolution hyperspectral image from a multispectral image (MSI) and a low-spatial-resolution hyperspectral image (HSI).

The work is associated with the following manuscript, submitted to **ICASSP 2027**:

> A. Guedjali, E.-H. Djermoune, P. Catala, and S. Delchini,
> *Fused Gromov-Wasserstein for Hyperspectral-Multispectral Image Fusion*,
> submitted to ICASSP 2027.

---

## Content

```text
Hyperspectral-Multispectral-Image-Fusion-via-Fused-Gromov-Wasserstein/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── fgw.py
│   ├── reconstruction.py
│   ├── hmwb.py
│   ├── evaluation.py
│   └── visualization.py
│
├── data/
│   ├── PaviaU.mat
│   └── indianpinearray.npy
│
└── results/
    ├── pavia/
    │   ├── pavia_efgw.npy
    │   ├── pavia_hmwb.npy
    │   ├── metrics.txt
    │   └── figures/
    │
    └── indian_pines/
        ├── indian_pines_efgw.npy
        ├── indian_pines_hmwb.npy
        ├── metrics.txt
        └── figures/
```

### Main modules

* `main.py`: runs the complete experimental pipeline.
* `src/data.py`: data loading, patch extraction, and synthetic observation generation.
* `src/fgw.py`: spectral and structural cost construction and FGW optimization.
* `src/reconstruction.py`: reconstruction of the fused hyperspectral image from the transport plan.
* `src/hmwb.py`: implementation of the HMWB baseline.
* `src/evaluation.py`: computation of reconstruction metrics.
* `src/visualization.py`: generation of image comparisons and spectral signatures.

---

## Requirements

The code is written in Python 3 and requires:

* NumPy
* SciPy
* POT (Python Optimal Transport)
* Matplotlib
* scikit-image

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Data

Two hyperspectral datasets are used in the experiments:

* **Pavia University**
* **Indian Pines**

### Pavia University

The original Pavia University hyperspectral image has dimensions:

```text
610 × 340 × 103
```

where:

* `610 × 340` is the spatial resolution;
* `103` is the number of spectral bands.

The dataset is available from Kaggle:

[Pavia University dataset on Kaggle](https://www.kaggle.com/code/ardaorcun/hyperspectral-paviau?select=PaviaU.mat&utm_source=chatgpt.com)

Place the downloaded file in:

```text
data/PaviaU.mat
```

The MATLAB variable containing the hyperspectral image is:

```text
paviaU
```

For the experiment, a spatial patch is extracted from the original image:

```text
50 × 50 × 103
```

### Indian Pines

The original Indian Pines hyperspectral image has dimensions:

```text
145 × 145 × 200
```

where:

* `145 × 145` is the spatial resolution;
* `200` is the number of spectral bands.

The dataset is available from Kaggle:

[Indian Pines dataset on Kaggle](https://www.kaggle.com/datasets/abhijeetgo/indian-pines-hyperspectral-dataset/data?select=indianpinearray.npy&utm_source=chatgpt.com)

Place the downloaded file in:

```text
data/indianpinearray.npy
```

For the experiment, a spatial patch is extracted from the original image:

```text
40 × 40 × 200
```

---

## Experimental Setup

The original hyperspectral patch is considered as the **reference hyperspectral image**.

Two observations are generated from the reference image:

* an **MSI** with high spatial resolution and a reduced number of spectral bands;
* an **HSI** with lower spatial resolution and the full spectral information.

The synthetic HSI is generated using spatial Gaussian blurring followed by spatial downsampling.

The synthetic MSI is generated using Gaussian spectral response functions.

The fusion problem can be summarized as:

```text
                    Reference HSI
                    H × W × B_H
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
             MSI                   HSI
          H × W × B_M          H' × W' × B_H
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                  Fused Gromov-
                   Wasserstein
                         │
                         ▼
                    Fused HSI
                    H × W × B_H
```

The target fused image has the spatial resolution of the MSI and the spectral resolution of the HSI.

---

## Run the Experiments

The complete pipeline is implemented in `main.py`.

Select the dataset at the beginning of the script.

For Pavia University:

```python
DATASET = "pavia"
```

For Indian Pines:

```python
DATASET = "indian_pines"
```

Then run:

```bash
python main.py
```

The script automatically:

1. loads the selected dataset;
2. extracts the experimental patch;
3. generates the synthetic MSI and HSI;
4. constructs the spectral cost;
5. constructs the structural distance matrices;
6. computes the entropic FGW transport plan;
7. reconstructs the fused hyperspectral image;
8. runs HMWB;
9. evaluates the reconstructed images;
10. generates the visualization results;
11. saves the reconstructed images and metrics.

---

## Fused Gromov-Wasserstein

The proposed fusion method is based on **Fused Gromov-Wasserstein (FGW)** optimal transport.

The implementation is available in:

```text
src/fgw.py
```

FGW combines:

* a **linear feature cost**, describing the similarity between observations;
* a **quadratic Gromov-Wasserstein cost**, describing the internal structure of each domain;
* an **entropic regularization** term.

This formulation simultaneously exploits spectral information and spatial-spectral relationships.

### Spectral Cost

The spectral cost is computed between MSI and HSI pixel signatures.

The HSI spectra are projected into the MSI spectral domain using Gaussian spectral response functions.

The spectral similarity is measured using the **Spectral Angle Mapper (SAM)**:

$$
C_{ij}
=
\arccos
\left(
\frac{
\langle m_i,\tilde{h}_j\rangle
}{
\|m_i\|_2\|\tilde{h}_j\|_2
}
\right).
$$

where \(m_i\) is an MSI spectral signature and \(\tilde{h}_j\) is the corresponding projected HSI spectral signature.

### Structural Cost

The spatial-spectral structure is represented using a distance between voxels:

$$
d(i,i')
=
(u_i-u_{i'})^2
+
(v_i-v_{i'})^2
+
\eta(z_i-z_{i'})^2.
$$

The three coordinates represent the spatial and spectral dimensions.

The structural distance matrices are computed independently for the MSI and HSI domains.

### Transport Plan

The FGW optimization produces a transport plan between the MSI and HSI pixel-band domains.

This transport plan is subsequently used to reconstruct the fused hyperspectral image.

---

## Fused Image Reconstruction

The reconstruction is implemented in:

```text
src/reconstruction.py
```

The optimal transport matrix is reshaped as:

$$
T
\in
\mathbb{R}_+^{N_M\times B_M\times N_H\times B_H}.
$$

The fused image is reconstructed by marginalizing the transport plan:

$$
F_{i\ell}
=
\sum_k \sum_j T_{ikj\ell}.
$$

The resulting matrix is reshaped onto the high-resolution MSI spatial grid.

The reconstructed image has dimensions:

```text
H_M × W_M × B_H
```

Thus, the fused image combines:

* the **spatial resolution of the MSI**;
* the **spectral resolution of the HSI**.

---

## B-SCOTT

**B-SCOTT (Blind-SCOTT)** is included as an additional reference method in the experimental comparison.

The B-SCOTT implementation is not part of this repository. The original MATLAB implementation is publicly available in:

[HSR_Software repository](https://github.com/cprevost4/HSR_Software?utm_source=chatgpt.com)

The repository provides the code associated with the hyperspectral super-resolution work of Prévost et al., including SCOTT and B-SCOTT.

B-SCOTT results reported below were obtained from the corresponding experiments and are included for comparison with the proposed FGW approach.

---

## HMWB

**HMWB** is used as a baseline method for comparison.

The implementation is available in:

```text
src/hmwb.py
```

HMWB is run using the same reference image, MSI, and HSI as the proposed FGW approach.

The resulting fused image has the same target dimensions:

```text
H_M × W_M × B_H
```

Reference:

[HMWB reference — HAL](https://hal.science/hal-01620601v1?utm_source=chatgpt.com)

---

## Evaluation Metrics

The reconstructed images are compared with the reference hyperspectral image using four metrics:

* **SAM (rad)** — Spectral Angle Mapper;
* **CC** — Correlation Coefficient;
* **R-SNR (dB)** — Reconstruction Signal-to-Noise Ratio;
* **ERGAS** — Erreur Relative Globale Adimensionnelle de Synthèse.

For SAM and ERGAS, lower values indicate smaller reconstruction errors. For CC and R-SNR, higher values indicate stronger agreement with the reference image.

The metrics are implemented in:

```text
src/evaluation.py
```

---

## Results

### Pavia University

| Method  | SAM (rad) ↓ |     CC ↑ | R-SNR (dB) ↑ |  ERGAS ↓ |
| :------ | ----------: | -------: | -----------: | -------: |
| FGW     |        0.16 | **0.97** |        17.17 | **4.01** |
| B-SCOTT |    **0.10** |     0.95 |    **17.82** |     4.02 |
| HMWB    |        0.25 |     0.90 |        12.61 |     6.52 |

### Indian Pines

| Method  | SAM (rad) ↓ |     CC ↑ | R-SNR (dB) ↑ |  ERGAS ↓ |
| :------ | ----------: | -------: | -----------: | -------: |
| FGW     |    **0.07** | **0.99** |    **22.96** | **2.06** |
| B-SCOTT |        0.12 |     0.59 |         4.26 |    13.02 |
| HMWB    |        0.61 |     0.62 |         2.79 |    18.28 |

---

## Visualization

The visualization functions are implemented in:

```text
src/visualization.py
```

The generated comparisons contain:

```text
Reference | MSI | HSI | HMWB | FGW
```

for selected spectral bands.

The figures are stored in:

```text
results/pavia/figures/
```

and:

```text
results/indian_pines/figures/
```

The code can also generate spectral signatures for selected pixels.

---

## Output Files

For Pavia University, the reconstructed images are saved as:

```text
results/pavia/pavia_efgw.npy
results/pavia/pavia_hmwb.npy
```

The evaluation metrics are saved in:

```text
results/pavia/metrics.txt
```

For Indian Pines:

```text
results/indian_pines/indian_pines_efgw.npy
results/indian_pines/indian_pines_hmwb.npy
```

with the metrics stored in:

```text
results/indian_pines/metrics.txt
```

Figures are stored in the corresponding `figures/` directories.

> **Note:** The current output filenames still contain `efgw` for compatibility with the existing implementation. The proposed method is referred to as **FGW** throughout the manuscript and documentation.

---

## Reproduce the Experiments

### Pavia University

Set:

```python
DATASET = "pavia"
```

and run:

```bash
python main.py
```

### Indian Pines

Set:

```python
DATASET = "indian_pines"
```

and run:

```bash
python main.py
```

The complete FGW and HMWB pipeline is executed automatically.

B-SCOTT can be reproduced using the external implementation:

[HSR_Software — B-SCOTT implementation](https://github.com/cprevost4/HSR_Software?utm_source=chatgpt.com)

---

## References

### Fused Gromov-Wasserstein

The FGW formulation used in this work is based on:

> T. Vayer, L. Chapel, R. Flamary, R. Tavenard, and N. Courty,
> *Fused Gromov-Wasserstein distance for structured objects: theoretical foundations and mathematical properties.*

[arXiv:1811.02834](https://arxiv.org/abs/1811.02834?utm_source=chatgpt.com)

### B-SCOTT

The B-SCOTT implementation used for comparison is available at:

[HSR_Software — Clémence Prévost et al.](https://github.com/cprevost4/HSR_Software?utm_source=chatgpt.com)

### HMWB

The HMWB baseline is based on the Wasserstein barycenter approach for hyperspectral-multispectral image fusion:

[HAL: hal-01620601v1](https://hal.science/hal-01620601v1?utm_source=chatgpt.com)

### This Work

> A. Guedjali, E.-H. Djermoune, P. Catala, and S. Delchini,
> *Fused Gromov-Wasserstein for Hyperspectral-Multispectral Image Fusion*,
> submitted to ICASSP 2027.

---

## Funding

This work is supported by the **ANR France 2030 – PEPR Sous-sol**, through the **InnovTech** project (ANR-22-EXSS-0006).

---

## Authors

**Amel Guedjali**
Université de Lorraine, CNRS, CRAN, France

**El-Hadi Djermoune**
Université de Lorraine, CNRS, CRAN, France

**Paul Catala**
Université de Lorraine, CNRS, CRAN, France

**Sylvain Delchini**
BRGM, Orléans, France

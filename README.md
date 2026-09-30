# Hyperspectral-Multispectral Image Fusion via Fused Gromov-Wasserstein

Copyright (c) 2026 Amel Guedjali, El-Hadi Djermoune, Paul Catala, Sylvain Delchini

This repository contains the Python implementation of **hyperspectral-multispectral image fusion using Fused Gromov-Wasserstein (FGW) optimal transport**.

The objective is to reconstruct a high-spatial-resolution hyperspectral image from a high-spatial-resolution multispectral image (MSI) and a low-spatial-resolution hyperspectral image (HSI).

This work is associated with the manuscript:

> A. Guedjali, E.-H. Djermoune, P. Catala, and S. Delchini,
> *Fused Gromov-Wasserstein for Hyperspectral-Multispectral Image Fusion*,
> submitted to ICASSP 2027.

## Repository structure

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

## Requirements

Python 3.9+ is recommended.

Install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets

### Pavia University

**Original size:** 610 × 340 × 103

**Experimental patch:** 50 × 50 × 103

Dataset: <a href="https://www.kaggle.com/code/ardaorcun/hyperspectral-paviau?select=PaviaU.mat" target="_blank">Pavia University — Kaggle</a>

Place the dataset at:

```text
data/PaviaU.mat
```

The MATLAB variable is `paviaU`.

### Indian Pines

**Original size:** 145 × 145 × 200

**Experimental patch:** 40 × 40 × 200

Dataset: <a href="https://www.kaggle.com/datasets/abhijeetgo/indian-pines-hyperspectral-dataset/data?select=indianpinearray.npy" target="_blank">Indian Pines — Kaggle</a>

Place the dataset at:

```text
data/indianpinearray.npy
```

## Method

The proposed approach formulates hyperspectral-multispectral fusion as an **entropic Fused Gromov-Wasserstein (FGW)** transport problem.

FGW jointly exploits:

* spectral similarity between MSI and HSI observations;
* spatial-spectral structural relationships within each modality.

The optimal transport plan is used to reconstruct the fused hyperspectral image on the spatial grid of the MSI.

The FGW implementation is available in:

```text
src/fgw.py
```

The reconstruction from the transport plan is implemented in:

```text
src/reconstruction.py
```

For comparison, HMWB is implemented in:

```text
src/hmwb.py
```

B-SCOTT is used as an external comparison method and is not implemented in this repository.

## Run

Select the dataset in `main.py`:

```python
DATASET = "pavia"
```

or:

```python
DATASET = "indian_pines"
```

Then run:

```bash
python main.py
```

The reconstructed images, metrics and figures are saved in:

```text
results/
```

## Results

The following results correspond to the experiments reported in the ICASSP 2027 manuscript.

### Pavia University

| Method  | SAM (rad) |       CC | R-SNR (dB) |    ERGAS |
| ------- | --------: | -------: | ---------: | -------: |
| FGW     |      0.16 | **0.97** |      17.17 | **4.01** |
| B-SCOTT |  **0.10** |     0.95 |  **17.82** |     4.02 |
| HMWB    |      0.25 |     0.90 |      12.61 |     6.52 |

### Indian Pines

| Method  | SAM (rad) |       CC | R-SNR (dB) |    ERGAS |
| ------- | --------: | -------: | ---------: | -------: |
| FGW     |  **0.07** | **0.99** |  **22.96** | **2.06** |
| B-SCOTT |      0.12 |     0.59 |       4.26 |    13.02 |
| HMWB    |      0.61 |     0.62 |       2.79 |    18.28 |

## Visual Results

The following figures are the same visual comparisons presented in the paper.

The image order in each figure is:

**Reference → MSI → HSI → HMWB → FGW**

### Pavia University

**Band 25**

![Pavia University - Band 25](results/pavia/figures/pavia_EFGW_m01_h25.png)

**Band 75**

![Pavia University - Band 75](results/pavia/figures/pavia_EFGW_m04_h75.png)

### Indian Pines

**Band 46**

![Indian Pines - Band 46](results/indian_pines/figures/indian_pines_EFGW_m01_h46.png)

**Band 118**

![Indian Pines - Band 118](results/indian_pines/figures/indian_pines_EFGW_m03_h118.png)

## Spectral Signatures

The paper reports spectral signatures for **4 selected pixels** for each dataset.

The order of the curves is:

**Reference → HMWB → FGW**

### Pavia University

<a href="results/pavia/figures/pavia_signatures.pdf" target="_blank">View Pavia University spectral signatures</a>

### Indian Pines

<a href="results/indian_pines/figures/indian_pines_signatures.pdf" target="_blank">View Indian Pines spectral signatures</a>

## Evaluation Metrics

The fusion quality is evaluated using:

* **SAM** — Spectral Angle Mapper
* **CC** — Cross-Correlation
* **R-SNR** — Reconstruction Signal-to-Noise Ratio
* **ERGAS** — Erreur Relative Globale Adimensionnelle de Synthèse

## Output

Results are stored separately for each dataset:

```text
results/
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

The output filenames retain `efgw` for compatibility with the current implementation. The proposed method is referred to as **FGW** throughout the manuscript and README.

## References

### Fused Gromov-Wasserstein

T. Vayer, L. Chapel, R. Flamary, R. Tavenard, and N. Courty,
*Fused Gromov-Wasserstein distance for structured objects: theoretical foundations and mathematical properties.*

<a href="https://arxiv.org/abs/1811.02834" target="_blank">FGW — arXiv</a>

### HMWB

M. Mifdal et al.,
*Hyperspectral image fusion using Wasserstein barycenters.*

<a href="https://hal.science/hal-01620601v1" target="_blank">HMWB — HAL</a>

### B-SCOTT

B-SCOTT implementation:

<a href="https://github.com/cprevost4/HSR_Software" target="_blank">B-SCOTT — HSR_Software</a>

## Funding

This work was supported by the French Research Agency (ANR) under France 2030 – PEPR Sous-sol program for the InnovTech project (ANR-22-EXSS-0006).

## Authors

**Amel Guedjali** — Université de Lorraine, CNRS, CRAN, France

**El-Hadi Djermoune** — Université de Lorraine, CNRS, CRAN, France

**Paul Catala** — Université de Lorraine, CNRS, CRAN, France

**Sylvain Delchini** — BRGM, Orléans, France

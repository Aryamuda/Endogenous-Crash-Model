# Stochastic Precursors of Critical Transitions in Endogenously Synchronized Markets

> **Revision in progress for resubmission to Chaos, Solitons & Fractals (Elsevier)**  
> Preprint: [SSRN DOI: 10.2139/ssrn.6070347](https://ssrn.com/abstract=6070347)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)

This repository contains the full simulation and empirical validation code for the thesis **"Stochastic Precursors of Critical Transitions in Endogenously Synchronized Markets"**.

## Overview
This project models financial market crashes as **endogenous synchronization events** using a coupled system of stochastic oscillators (Kuramoto Model). It demonstrates that **Variance Inflation** and **Autocorrelation Deepening** in the hidden synchronization manifold propagate to observable market risk prior to a collapse.

## Repository Structure
```
├── src/                # Core model, diagnostics, and visualization logic
├── empirical/          # S&P 500 correlation analysis (Empirical Validation)
├── notebooks/          # Exploratory prototypes
├── run_simulation.py   # Main simulation script (Figures 1-5)
├── run_alpha_robustness.py  # alpha-sensitivity robustness script
├── requirements.txt    # Python dependencies
└── README.md
```

## Quick Start

### 1. Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/Aryamuda/Endogenous-Crash-Model.git
cd Endogenous-Crash-Model
pip install -r requirements.txt
```

### 2. Theoretical Simulation (Figures 1-5)
Execute the main simulation script to generate the primary figures:
```bash
python run_simulation.py
```
This generates `Figure1.pdf` through `Figure5.pdf`.

### 3. Empirical Validation (Figure 6)
To replicate the empirical S&P 500 analysis (mean pairwise correlation proxy), run the scripts in the `empirical/` directory:
```bash
python empirical/run_multistock_zscore.py
```
*Note: This requires an internet connection to fetch historical data via `yfinance`.*

### 4. Alpha Robustness Study
To verify the invariance of relative precursors to the projection parameter $\alpha$:
```bash
python run_alpha_robustness.py
```

## Key Findings
- **Layer Orthogonality**: Market risk reflects hidden synchronization without direct feedback.
- **CSD Propagation**: Precursors generated at the meso-scale remain detectable at the macroscopic hazard level.
- **Robustness**: Signal detectability is invariant to the coupling sensitivity $\alpha$ and robust across population sizes.

---
**Author:** Arya Muda Siregar  
**Institution:** Institut Teknologi Sumatera (ITERA)

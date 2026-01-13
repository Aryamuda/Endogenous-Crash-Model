# Stochastic Precursors of Critical Transitions in Endogenously Synchronized Markets

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)

> **Dragon King events are not random outliers; they are mechanically predictable phase transitions.**

This repository contains the simulation code for the paper **"Stochastic Precursors of Critical Transitions in Endogenously Synchronized Markets"**

## Overview
This project models financial market crashes as **endogenous synchronization events** using a coupled system of stochastic oscillators (Kuramoto Model). Unlike standard equilibrium models that rely on external shocks, this framework demonstrates how internal alignment drives volatility clustering.

**Key Mechanism:**
1. **Micro-Layer:** $N$ agents modeled as stochastic oscillators.
2. **Phase Transition:** As coupling strength $K$ approaches $K_c$, agents synchronize.
3. **Macro-Layer:** A Poisson market observable responds to the hidden order parameter $r(t)$.

## Quick Start

### 1. Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/Aryamuda/Endogenous-Crash-Model.git
cd Endogenous-Crash-Model
pip install -r requirements.txt
```

### 2. Run the Simulation
Execute the main script to generate all figures:
```bash
python run_simulation.py
```

This will automatically:
- Run all 5 simulation blocks
- Generate color-blind accessible figures
- Save outputs as: `Figure1.pdf`, `Figure2.pdf`, `Figure3.pdf`, `Figure4.pdf`, `Figure5.pdf`

## Results
The simulation demonstrates that **Variance Inflation** and **Autocorrelation** in the hidden order parameter appear *before* the market crash.

**Generated Figures:**
- **Figure1.pdf**: Phase transition and time evolution across regimes
- **Figure2.pdf**: Stochastic precursor detection (variance & autocorrelation)
- **Figure3.pdf**: Market projection mechanism (3-layer crash visualization)
- **Figure4.pdf**: Falsifiability tests (counterfactual scenarios)
- **Figure5.pdf**: Robustness validation (parameter sensitivity)

## Repository Structure
```
├── src/
│   ├── model.py           # Core Kuramoto model classes
│   ├── diagnostics.py     # Precursor analysis functions
│   └── visualization.py   # Color-blind safe plotting utilities
├── notebooks/
│   └── Thesisfinal.ipynb  # Interactive prototype and exploration
├── run_simulation.py      # Main executable (generates all figures)
├── requirements.txt       # Python dependencies
└── README.md
```

## Features
- **Reproducible Results**: Fixed seed (42) ensures identical outputs
- **Color-Blind Accessible**: Figures use Okabe-Ito palette with multiple visual cues
- **Publication Ready**: All outputs in vector PDF format
- **Modular Design**: Clean separation of model, diagnostics, and visualization



---
**Author:** Arya Muda Siregar  
**Institution:** Institut Teknologi Sumatera (ITERA)

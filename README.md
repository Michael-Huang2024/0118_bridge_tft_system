# Physics-Guided TFT Platform for Bridge Deterioration Forecasting

This repository implements a physics-guided time-series forecasting platform for bridge deck deterioration analysis.  
The system is designed as a **preparatory and validation framework for real DOT/NBI bridge inspection data**, using physically consistent simulations to verify data integration, missing-data handling, and multi-horizon prediction with Temporal Fusion Transformers (TFT).

---

## 1. Project Motivation

Bridge inspection data collected by DOT and NBI are characterized by:

- Sparse and irregular inspection intervals  
- Variable sequence lengths across bridges  
- Missing observations due to inspection gaps  
- Strong physical constraints governing deterioration processes  

This project addresses these challenges by combining:

- **Physics-based simulation** (Fick’s second law of chloride diffusion)
- **Probabilistic data augmentation** (diffusion-based variability)
- **Deep sequence modeling** (Temporal Fusion Transformer)

The current implementation uses simulated but physically grounded data to validate the full modeling pipeline before integrating real-world inspection records.

---

## 2. System Overview

The workflow consists of the following stages:

1. **Physics-based sequence generation**  
   - Chloride diffusion is simulated using Fick’s second law
   - Generates physically consistent baseline trajectories (e.g., `C_rebar`, `C_surf`, `D_eff`, `J_cl`)

2. **Probabilistic augmentation**  
   - Controlled stochastic variability is introduced to emulate environmental and material uncertainty
   - Long-term diffusion-dominated trends are preserved

3. **Missing data injection**  
   - Missing values are intentionally introduced at the input level
   - Emulates realistic inspection and monitoring gaps
   - No deterministic imputation is applied prior to modeling

4. **Dataset construction for TFT**  
   - Multi-variable, multi-horizon sequences are assembled
   - Variable-length histories are supported naturally

5. **TFT training and forecasting**  
   - Temporal Fusion Transformer is trained for multi-step forecasting
   - Outputs include future chloride-related states and cumulative damage indicators

---

## 3. Data Characteristics

### 3.1 Input Variables

Each bridge time series may include:

- `C_rebar` – Chloride concentration at reinforcement depth  
- `C_surf` – Surface chloride concentration  
- `D_eff` – Effective diffusion coefficient  
- `J_cl` – Chloride flux  
- `Damage` – Cumulative deterioration indicator (0–1 scale)

### 3.2 Missing Data

- Missing values are **explicitly included** in the observed inputs  
- Missingness represents inspection gaps rather than sensor failure  
- TFT learns directly from incomplete sequences, improving robustness to real-world data conditions

### 3.3 Variable-Length Sequences

- Bridges have different inspection histories
- No artificial truncation or forced equal-length padding is applied
- TFT encoder–decoder architecture naturally accommodates varying sequence lengths

---

## 4. Repository Structure

```text
0118_bridge_tft_system/
│
├── scripts/
│   ├── 01_generate_sim_data.py      # Physics-based simulation (Fick’s law)
│   ├── 02_inject_missing.py          # Missing data generation
│   ├── 03_build_dataset.py           # TFT dataset construction
│   ├── 04_train_tft.py               # TFT model training
│   ├── 05_predict_and_eval.py        # Prediction and evaluation
│
├── outputs/
│   ├── figures/                      # RMSE and prediction plots
│   └── results/                      # Saved predictions and metrics
│
├── README.md

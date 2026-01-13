# Cyclodextrin-PFAS Host-Guest Analysis

First phase of SYROP project aiming to predict the binding energy between cyclodextrins and relevant organic compounds, including PFAS, in a host-guest complex.

## Table of Contents

- [Project Organization](#project-organization)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [CLI Commands Reference](#cli-commands-reference)
  - [1. Data Download](#1-data-download)
  - [2. Feature & Embedding Generation](#2-feature--embedding-generation)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation & Prediction](#4-model-evaluation--prediction)
  - [5. Leave-One-Out Cross-Validation](#5-leave-one-out-cross-validation-loocv)
  - [6. Visualization & Figure Generation](#6-visualization--figure-generation)
- [Typical Workflow Example](#typical-workflow-example)
- [Adding Dependencies](#adding-dependencies)
- [Formatting and Checking](#formatting-and-checking)
- [Documentation](#documentation)
- [Versions](#versions)
- [License](#license)

## Project Organization

**IMPORTANT** These folders are not available in GitHub due to size constraints, but are available at [4TU Repository](https://doi.org/10.4121/a5051137-a93d-433e-9cfe-50980247930c). These folders can be downloaded and inserted into the downloaded Github repository for seamless use. Notably, when downloading the complete repository from 4TU, all analyses are complete, including representation and figure generation, reducing the need to perform many of the pipeline steps described below.

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external*       <- Data from third party sources.
│   ├── interim*        <- Intermediate data that has been transformed.
│   ├── processed*      <- The final, canonical data sets for modeling.
│   └── raw*            <- The original, immutable data dump.
│
├── docs                       <- A default mkdocs project; see www.mkdocs.org for details
│
├── models*                    <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks                  <- Jupyter notebooks adressing feature generation and minor analyses.
│
├── pyproject.toml             <- Project configuration file with package metadata for
│                                 cd_host_guest and configuration for tools like black
│
├── references                 <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports*                     <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures*                 <- Generated graphics and figures to be used in reporting
│   └── val_to_train_neighbors*  <- Neighbor analysis between training and test molecules
│
├── requirements.txt           <- The requirements file for reproducing the analysis environment without uv, e.g.
│                                 generated with `uv freeze > requirements.txt`
│
├── uv.lock                    <- Contains the uv dependencies for the current venv
│
├── setup.cfg                  <- Configuration file for flake8
│
├── mkdocs.yml                 <- Configuration file with mkdocs settings
│
├── .python-version            <- File containing the venv Python version
│
├── .pre-commit-config.yaml    <- File containing the venv Python version
│
├── .github                    <- Contains files defining the workflows to be executed to run tests and generate
│                                 deploy docs
│
└── cd_host_guest   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cd_host_guest a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Installation


1. Install [uv](https://docs.astral.sh/uv/):

    - Linux and MacOS

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    - Windows

        ```bash
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

2. Install the dependencies, including the dev dependencies

    ```bash
    uv sync
    ```
    or install only the runtime dependencies

    ```bash
    uv sync --no-dev
    ```

--------

## Data Preparation

### 1. Download OpenCycloDB

The primary training dataset (OpenCycloDB) can be downloaded automatically:

```bash
uv run python -m cd_host_guest.dataset download OpenCycloDB
```

This downloads the dataset from [Zenodo](https://zenodo.org/records/7575539) and extracts it to `data/raw/OpenCycloDB/`.

### 2. External Validation Datasets (Manual)

Two external validation datasets must be downloaded manually:

#### PFAS Validation Dataset
- **Source:** [https://doi.org/10.1021/acs.jpcb.7b05901](https://doi.org/10.1021/acs.jpcb.7b05901)
- **Download Location:** Place CSV file in `data/external/validation/pfas_val/`
- **Required Columns:** `Host`, `Guest`, `Binding_Energy` (in kJ/mol)

#### Cyclodextrin Validation Dataset
- **Source:** [https://doi.org/10.1016/j.hazadv.2025.100904](https://doi.org/10.1016/j.hazadv.2025.100904)
- **Download Location:** Place CSV file in `data/external/validation/cd_val/`
- **Required Columns:** `Host`, `Guest`, `Binding_Energy` (in kJ/mol)

After downloading both datasets, canonicalize SMILES strings:

```bash
# Canonicalize CD validation dataset
uv run python -m cd_host_guest.features canonicalize-smiles \
  data/external/validation/cd_val/cd_val.csv \
  --output-csv data/external/validation/cd_val/cd_val_canonical.csv

# Canonicalize PFAS validation dataset
uv run python -m cd_host_guest.features canonicalize-smiles \
  data/external/validation/pfas_val/pfas_val.csv \
  --output-csv data/external/validation/pfas_val/pfas_val_canonical.csv
```

For detailed information on data structure and feature dimensions, see [data/README.md](data/README.md).

--------

## CLI Commands Reference

This project provides comprehensive CLI tools for the entire machine learning pipeline, from data download to model evaluation and visualization. All commands use the `uv run python -m cd_host_guest.<module>` pattern.

### 1. Data Download

#### Download OpenCycloDB Dataset

```bash
# Download OpenCycloDB dataset (automated)
uv run python -m cd_host_guest.dataset download OpenCycloDB

# Force re-download even if files exist
uv run python -m cd_host_guest.dataset download OpenCycloDB --force
```

**Note:** Two external validation datasets (CD validation and PFAS validation) must be downloaded manually and placed in `data/external/validation/`.

**Representations used in manuscript:** Original Enriched, ECFP, ECFP+, UniMol, GROVER, and ChemBERTa (string and string_finetuned).

**Note on naming conventions:**
- **String representations** refer to ChemBERTa models trained on SMILES
- **end2end** terminology in old code refers to ChemBERTa models
- Current naming: `string` (base ChemBERTa) and `string_finetuned` (fine-tuned ChemBERTa)

---

### 2. Feature & Embedding Generation


#### Canonicalize SMILES

```bash
# Convert SMILES to canonical isomeric form
uv run python -m cd_host_guest.features canonicalize-smiles INPUT_FILE.csv --output-csv OUTPUT_FILE.csv
```

#### Generate All Basic Features

```bash
# Generate ECFP fingerprints, SMILES representations, graphs, and enriched features
uv run python -m cd_host_guest.features main
```

This command processes the raw OpenCycloDB data and generates:
- ECFP fingerprints (Guest and Host)
- String representations (used for ChemBERTa models)
- Original enriched features

#### ECFP+, GROVER, UniMol2, ChemBERTa

The ECFP+, GROVER, UniMol2, ChemBERTa representations were generated separately. The following notebooks outline the generation process:

- ECFP+: notebooks\ECFPplus_gen.ipynb
- GROVER: notebooks\GROVER_gen.ipynb
- UniMol2: notebooks\UniMol_gen.ipynb
- ChemBERTa: notebooks\ChemBERTa_gen.ipynb

**Note:** ChemBERTa models (end2end and end2end_finetuned) are trained directly on string representations (SMILES), not pretrained embeddings. Two versions:
- **string**: Base ChemBERTa embeddings
- **string_finetuned**: Fine-tuned ChemBERTa embeddings on external validation data


#### Compare PFAS in Guests

```bash
# Check for PFAS compounds in guest molecules
uv run python -m cd_host_guest.features compare_PFAS_in_guests
```

### 3. Model Training

#### Hyperparameter Tuning (Sweep)

```bash
# Train and tune LightGBM model
uv run python -m cd_host_guest.modeling.train train \
  --model-type lgbm \
  --data-source OpenCycloDB \
  --representation ecfp

# Train and tune FNN model
uv run python -m cd_host_guest.modeling.train train \
  --model-type fnn \
  --data-source OpenCycloDB \
  --representation ecfp

# Train ChemBERTa FNN on end-to-end model with string representations (with hyperparameter sweep)
uv run python -m cd_host_guest.modeling.train train \
  --model-type end2end \
  --data-source OpenCycloDB \
  --representation string
```

**Supported representations:** `original_enriched`, `ecfp`, `ecfp_plus`, `unimol`, `grover`, `string` (for ChemBERTa)

#### Train Final Models with Best Parameters

```bash
# Train final model for a specific representation
uv run python -m cd_host_guest.modeling.train train-final-model \
  --model-type lgbm \
  --data-source OpenCycloDB \
  --representation ecfp

# Train all final models for all representations
uv run python -m cd_host_guest.modeling.train train-all-final-models \
  --model-types ["lgbm", "fnn", "end2end"] \
  --data-source OpenCycloDB
```

#### Fine-tune End2End Model

The ChemBERTa model can be fine-tuned to the OpenCycloDB data during FNN training:

```bash
# Fine-tune end2end ChemBERTa model
uv run python -m cd_host_guest.modeling.train fine-tune-end2end \
  --pretrained-model-name models/OpenCycloDB/string/end2end_string \
  --representation "string" \
  --data-source "OpenCycloDB"

# Hyperparameter sweep for fine-tuning
uv run python -m cd_host_guest.modeling.train sweep-finetune-end2end \
  --pretrained-model-name models/OpenCycloDB/string/end2end_string \
  --representation "string" \
  --data-source "OpenCycloDB"

# Train final fine-tuned model
uv run python -m cd_host_guest.modeling.train train-final-finetuned \
  --pretrained-model-name models/OpenCycloDB/string/end2end_string \
  --source-type project
```

#### List Transformer Layers

```bash
# List all available transformer layers for unfreezing
uv run python -m cd_host_guest.modeling.train list-transformer-layers
```

---

### 4. Model Evaluation & Prediction

#### Generate OpenCycloDB Predictions

```bash
# Predict with trained model
uv run python -m cd_host_guest.modeling.predict predict \
  --model-type lgbm \
  --representation ecfp

# Predict with fine-tuned end2end model
uv run python -m cd_host_guest.modeling.predict predict-finetuned

# Predict with all models
uv run python -m cd_host_guest.modeling.predict predict-all-models
```

#### Generate External Validation Predictions

```bash
# Predict on external validation datasets
uv run python -m cd_host_guest.modeling.predict predict-external-validation \
  --model-type lgbm \
  --representation ecfp \
  --val-set cd_val

# Evaluate all external validation predictions
uv run python -m cd_host_guest.modeling.predict evaluate-external-validation \
  --val-set cd_val

# Evaluate all models on all external datasets
uv run python -m cd_host_guest.modeling.predict evaluate-all-external-validation
```

#### Model Comparison & Metrics

```bash
# Compare all models
uv run python -m cd_host_guest.modeling.predict compare-models \
  --data-source OpenCycloDB \
  --use-finetuned \
  --save-results

# Get comprehensive metrics (train/val/test)
uv run python -m cd_host_guest.modeling.predict comprehensive-metrics \
  --data-source OpenCycloDB \
  --save-results
```

#### Retrieve WandB Configs

```bash
# Get best hyperparameters from WandB for a specific model
uv run python -m cd_host_guest.modeling.predict get-wandb-model \
  --model-type lgbm \
  --representation ecfp

# Get all WandB configs for all projects
uv run python -m cd_host_guest.modeling.predict get-all-wandb-configs
```

---

### 5. Leave-One-Out Cross-Validation (LOOCV)

LOOCV is performed on external validation data (cd_val and pfas_val combined) to assess compare with the OpenCycloDB models to assess generalization performance.

#### Download Best Parameters from WandB

If you've already completed hyperparameter sweeps and just need to retrieve the best parameters:

```bash
# Download best parameters for all representations
uv run python -m cd_host_guest.modeling.train loocv-download-params

# Download for a specific representation
uv run python -m cd_host_guest.modeling.train loocv-download-params --representation ecfp

# Customize project naming if needed
uv run python -m cd_host_guest.modeling.train loocv-download-params \
  --project-prefix "LOOCV-LightGBM" \
  --project-suffix "external-validation"
```

**Supported representations:** `unimol`, `grover`, `ecfp`, `string` (ChemBERTa embeddings), `string_finetuned` (fine-tuned ChemBERTa)

**Note:** The command maps WandB project names to folder names:
- `smiles_tokenizer` → saves to `string/`
- `finetuned` → saves to `string_finetuned/`

#### LOOCV Hyperparameter Tuning

```bash
# Tune hyperparameters for LOOCV using WandB sweeps
uv run python -m cd_host_guest.modeling.train loocv-tune \
  --representation ecfp \
  --model-type lgbm \
  --sweep-count 50

# Tune all representations
uv run python -m cd_host_guest.modeling.train loocv-tune-all \
  --model-type lgbm \
  --sweep-count 50
```

**What happens during tuning:**
1. Runs WandB hyperparameter sweep
2. Performs LOOCV for each hyperparameter combination
3. Automatically retrieves best parameters from WandB after sweep completes
4. Saves parameters to `models/external_validation/{representation}/loocv_best_params_{representation}.json`

#### Run LOOCV with Best Parameters

```bash
# Run final LOOCV with tuned parameters (saves model + result files)
uv run python -m cd_host_guest.modeling.train loocv-final \
  --representation ecfp \
  --model-type lgbm

# Save only the model file (skip CSV/JSON result files)
uv run python -m cd_host_guest.modeling.train loocv-final \
  --representation ecfp \
  --model-type lgbm \
  --save-results false

# Run LOOCV for all representations
uv run python -m cd_host_guest.modeling.train loocv-final-all \
  --model-type lgbm

# Save only model files for all representations
uv run python -m cd_host_guest.modeling.train loocv-final-all \
  --model-type lgbm \
  --save-results false
```

**Output files** (when `save-results=true`):
- `loocv_lgbm_{representation}_final.pkl` - Trained model
- `loocv_results_{representation}_final.csv` - Predictions vs actual values
- `loocv_metrics_{representation}_final.json` - RMSE, MAE, R², MSE
- `loocv_params_{representation}_final.json` - Model hyperparameters
- `loocv_final_summary.csv` - Summary across all representations (from loocv-final-all)

#### Standard LOOCV (without tuning)

```bash
# Run LOOCV with default parameters
uv run python -m cd_host_guest.modeling.train loocv \
  --representation ecfp \
  --model-type lgbm

# Run LOOCV for all representations
uv run python -m cd_host_guest.modeling.train loocv-all \
  --model-type lgbm
```

---

### 6. Visualization & Figure Generation

#### Basic Plots

```bash

# Create parity plot (true vs predicted)
uv run python -m cd_host_guest.plots create-parity-plot \
  --labels_path PREDICTIONS_PATH \
  --output-path OUTPUT.png

```

#### PCA Variance Analysis

```bash
# Perform PCA and plot variance explained
uv run python -m cd_host_guest.plots perform-pca-and-plot-variance INPUT.csv
```

#### Feature Importance Visualization

```bash
# SHAP feature importance
uv run python -m cd_host_guest.plots shap-feature-importance \
  --data-source OpenCycloDB \
  --representation ecfp \
  --model-type lgbm \
  --max-features 25 \
  --save-plot

# LightGBM feature importance
uv run python -m cd_host_guest.plots lgbm-feature-importance \
  --data-source OpenCycloDB \
  --representation ecfp \
  --model-type lgbm \
  --top-features 25 \
  --save-plot

# LightGBM cumulative importance
uv run python -m cd_host_guest.plots lgbm-total-importance \
  --data-source OpenCycloDB \
  --representation ecfp \
  --model-type lgbm \
  --save-plot
```

#### Dataset Overview

```bash
# Plot comprehensive dataset overview
uv run python -m cd_host_guest.plots plot-dataset-overview \
  --data-source OpenCycloDB
```

#### UMAP/PCA Visualization

```bash
# Generate UMAP embeddings
uv run python -m cd_host_guest.plots generate-umap-embeddings \
  --data-source OpenCycloDB \
  --representation ecfp

# Plot train/val representation UMAP grid
uv run python -m cd_host_guest.plots plot-train-val-representation-umap-grid \
  --data-source OpenCycloDB

```

---

## Typical Workflow Example

Here's a complete end-to-end workflow:

```bash
# 1. Download data
uv run python -m cd_host_guest.dataset download OpenCycloDB

# 2. Generate features/embeddings
uv run python -m cd_host_guest.features main
# Generate remaining features using notebook files

# 3. Generate external validation embeddings
uv run python -m cd_host_guest.features generate-external-validation-all-embeddings

# 4. Train models (hyperparameter sweep) on all representations
uv run python -m cd_host_guest.modeling.train train --model-type lgbm --representation original_enriched
uv run python -m cd_host_guest.modeling.train train --model-type lgbm --representation ecfp
uv run python -m cd_host_guest.modeling.train train --model-type lgbm --representation ecfp_plus
uv run python -m cd_host_guest.modeling.train train --model-type lgbm --representation unimol
uv run python -m cd_host_guest.modeling.train train --model-type lgbm --representation grover
uv run python -m cd_host_guest.modeling.train train --model-type fnn --representation original_enriched
uv run python -m cd_host_guest.modeling.train train --model-type fnn --representation ecfp
uv run python -m cd_host_guest.modeling.train train --model-type fnn --representation ecfp_plus
uv run python -m cd_host_guest.modeling.train train --model-type fnn --representation unimol
uv run python -m cd_host_guest.modeling.train train --model-type fnn --representation grover
uv run python -m cd_host_guest.modeling.train train --model-type end2end --representation string


# 5. Train final models with best parameters
uv run python -m cd_host_guest.modeling.train train-all-final-models --data-source OpenCycloDB

# 6. Fine-tune ChemBERTa
uv run python -m cd_host_guest.modeling.train sweep-finetune-end2end --pretrained-model-name models/OpenCycloDB/string/end2end_string
uv run python -m cd_host_guest.modeling.train train-final-finetuned --pretrained-model-name models/OpenCycloDB/string/end2end_string --source-type project

# 7. Generate predictions
uv run python -m cd_host_guest.modeling.predict predict-all-models

# 8. Evaluate on external validation
uv run python -m cd_host_guest.modeling.predict evaluate-all-external-validation

# 9. Compare all models
uv run python -m cd_host_guest.modeling.predict compare-models --save-results
uv run python -m cd_host_guest.modeling.predict comprehensive-metrics --save-results

# 10. Generate figures
uv run python -m cd_host_guest.plots plot-dataset-overview
uv run python -m cd_host_guest.plots create-parity-plot --labels-path LABELS_PATH.csv --predictions-path PRED_PATH.csv --output-path OUTPUT_PATH.png
uv run python -m cd_host_guest.plots generate-umap-embeddings --embedding-type "guest" --force-regenerate
uv run python -m cd_host_guest.plots plot-train-val-representation-umap-grid --embedding-type "guest"
```

---

## License

Distributed under the terms of the [MIT License](LICENSE).

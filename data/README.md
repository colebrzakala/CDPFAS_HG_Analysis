# Data Directory Structure

This directory contains all datasets used in the cyclodextrin host-guest binding prediction project.

## Directory Organization

```
data/
├── raw/              # Original, immutable data
├── interim/          # Intermediate transformed data
├── processed/        # Final datasets ready for modeling
└── external/         # Third-party data and validation sets
```

## Detailed Structure

### raw/
Contains the original downloaded datasets without modifications.

- **OpenCycloDB/**: Host-guest binding affinity data for cyclodextrins
  - Source: [Zenodo (DOI: 10.5281/zenodo.7575539)](https://zenodo.org/records/7575539)
  - Downloaded via: `uv run python -m cd_host_guest.dataset download OpenCycloDB`
  - Files: `Data.zip` (extracted to OpenCycloDB/)
  - Format: CSV with columns `Host`, `Guest`, `Binding_Energy` (kJ/mol)

### interim/
Intermediate processing outputs including generated embeddings and features.

**Subdirectories by representation type:**
- `ecfp/` - Extended Connectivity Fingerprints
- `ecfp_plus/` - ECFP with additional molecular descriptors
- `grover/` - GROVER transformer embeddings
- `original_enriched/` - OpenCycloDB features
- `string/` - String-based molecular representations (used for ChemBERTa models)
- `unimol/` - UniMol2 transformer embeddings

### processed/
Final feature matrices split into train/validation/test sets, ready for model training.

**Structure per representation:**
```
processed/
└── OpenCycloDB/
    ├── ecfp/
    │   ├── train_features.csv
    │   ├── train_labels.csv
    │   ├── val_features.csv
    │   ├── val_labels.csv
    │   ├── test_features.csv
    │   └── test_labels.csv
    ├── ecfp_plus/
    ├── grover/
    ├── original_enriched/
    ├── string/
    └── unimol/
```

### external/
External data sources and validation datasets.

- **validation/**: External validation datasets for model testing
  - `cd_val/`: Cyclodextrin validation dataset (manually downloaded)
  - `pfas_val/`: PFAS validation dataset (manually downloaded)
  - `base_cdpfas_val/`: Combined base validation set

## Feature Dimensions

| Representation | Dimension | Description |
|----------------|-----------|-------------|
| Original Enriched | Variable | Original features with chemical enrichment |
| ECFP | 1024 | Extended Connectivity Fingerprints (Morgan, radius=2) |
| ECFP+ | 1024 + descriptors | ECFP with additional molecular properties |
| UniMol2 | 512 | UniMol2 transformer embeddings (pretrained) |
| GROVER | 2048 | GROVER transformer embeddings (pretrained) |
| string/ChemBERTa (end2end) | 768 | ChemBERTa model trained end-to-end on string representations |
| string/ChemBERTa (finetuned) | 768 | ChemBERTa model finetuned on external validation data |

**Note:** String representations are used as input for training the ChemBERTa models (end2end and end2end_finetuned), which then produce 768-dimensional embeddings.
## Data Splits

The OpenCycloDB dataset is split into three sets:

- **Training Set**: 80% of data
  - Used for model training and hyperparameter tuning

- **Validation Set**: 10% of data
  - Used for early stopping and model selection during training

- **Test Set**: 10% of data
  - Held-out set for final model evaluation

**Split Strategy**: Random stratified (by CD type) split with fixed random seed for reproducibility.

## External Validation Datasets

### External PFAS Validation (pfas_val)
- **Purpose**: Test generalization to PFAS compounds
- **Size**: 21 host-guest pairs
- **Source**: [https://pubs.acs.org/doi/10.1021/acs.jpcb.7b05901](https://pubs.acs.org/doi/10.1021/acs.jpcb.7b05901)
- **Required columns**: `Host`, `Guest`, `Binding Energy` (kJ/mol)

### External Beta-Cyclodextrin Validation (cd_val)
- **Purpose**: Test generalization to different cyclodextrin complexes
- **Size**: 42 host-guest pairs
- **Source**: [https://www.sciencedirect.com/science/article/pii/S2772416625003158#sec0007](https://www.sciencedirect.com/science/article/pii/S2772416625003158#sec0007)
- **Required columns**: `Host`, `Guest`, `Binding Energy` (kJ/mol)

## Data Format Requirements

All CSV files must contain at minimum:
- `Host`: SMILES string of host molecule (cyclodextrin)
- `Guest`: SMILES string of guest molecule
- `Binding Energy`: Binding free energy in kJ/mol (for labeled data)

## Generating Features

See the main [README.md](../README.md#2-feature--embedding-generation) for commands to generate features from raw data.

## Notes

- All feature matrices are stored as CSV files with samples as rows and features as columns
- Labels are stored separately from features
- SMILES strings are canonicalized using RDKit before feature generation
- Missing values are handled according to the specific feature type (typically imputed with 0 for fingerprints)

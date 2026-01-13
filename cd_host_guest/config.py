from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# LightGBM hyperparameter sweep configuration
lightgbm_sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_rmse",
        "goal": "minimize",
    },
    "parameters": {
        "num_leaves": {"distribution": "int_uniform", "min": 10, "max": 300},
        "max_depth": {"distribution": "int_uniform", "min": 10, "max": 250},
        "learning_rate": {"distribution": "uniform", "min": 0.0001, "max": 0.2},
        "n_estimators": {"distribution": "int_uniform", "min": 25, "max": 2500},
        "reg_alpha": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "reg_lambda": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "bagging_fraction": {"distribution": "uniform", "min": 0.2, "max": 1.0},
        "feature_fraction": {"distribution": "uniform", "min": 0.2, "max": 1.0},
    },
}

# Number of LightGBM hyperparameter sweep iterations
lgbm_sweep_count = 100


# FNN hyperparameter sweep configuration
fnn_sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_rmse",
        "goal": "minimize",
    },
    "parameters": {
        "epochs": {"values": [1000]},
        "hidden_dim": {"distribution": "int_uniform", "min": 100, "max": 1500},
        "num_layers": {"distribution": "int_uniform", "min": 2, "max": 20},
        "activation": {"values": ["relu"]},
        "dropout": {"distribution": "uniform", "min": 0, "max": 0.4},
        "learning_rate": {"distribution": "uniform", "min": 1.0e-4, "max": 1.0e-2},
        "regularization": {"distribution": "uniform", "min": 0.0, "max": 1.0e-1},
    },
}

# Number of FNN hyperparameter sweep iterations
fnn_sweep_count = 100


# End-to-End (SMILES) hyperparameter sweep configuration
end2end_sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_rmse",
        "goal": "minimize",
    },
    "parameters": {
        # SMILES tokenizer/transformer hyperparameters
        # "unfreeze_layers": {"values": [[
        # "encoder.layer.5.attention.output.dense.weight",
        # "encoder.layer.5.attention.output.dense.bias",
        # "encoder.layer.5.attention.output.LayerNorm.weight",
        # "encoder.layer.5.attention.output.LayerNorm.bias",
        # "encoder.layer.5.intermediate.dense.weight",
        # "encoder.layer.5.intermediate.dense.bias",
        # "encoder.layer.5.output.dense.weight",
        # "encoder.layer.5.output.dense.bias",
        # "encoder.layer.5.output.LayerNorm.weight",
        # "encoder.layer.5.output.LayerNorm.bias",
        #     "encoder.layer.4.attention.output.dense.weight",
        #     "encoder.layer.4.attention.output.dense.bias",
        #     "encoder.layer.4.attention.output.LayerNorm.weight",
        #     "encoder.layer.4.attention.output.LayerNorm.bias",
        #     "encoder.layer.4.intermediate.dense.weight",
        #     "encoder.layer.4.intermediate.dense.bias",
        #     "encoder.layer.4.output.dense.weight",
        #     "encoder.layer.4.output.dense.bias",
        #     "encoder.layer.4.output.LayerNorm.weight",
        #     "encoder.layer.4.output.LayerNorm.bias",
        #     # "encoder.layer.3.attention.output.dense.weight",
        #     # "encoder.layer.3.attention.output.dense.bias",
        #     # "encoder.layer.3.attention.output.LayerNorm.weight",
        #     # "encoder.layer.3.attention.output.LayerNorm.bias",
        #     # "encoder.layer.3.intermediate.dense.weight",
        #     # "encoder.layer.3.intermediate.dense.bias",
        #     # "encoder.layer.3.output.dense.weight",
        #     # "encoder.layer.3.output.dense.bias",
        #     # "encoder.layer.3.output.LayerNorm.weight",
        #     # "encoder.layer.3.output.LayerNorm.bias",
        #     # "encoder.layer.2.attention.output.dense.weight",
        #     # "encoder.layer.2.attention.output.dense.bias",
        #     # "encoder.layer.2.attention.output.LayerNorm.weight",
        #     # "encoder.layer.2.attention.output.LayerNorm.bias",
        #     # "encoder.layer.2.intermediate.dense.weight",
        #     # "encoder.layer.2.intermediate.dense.bias",
        #     # "encoder.layer.2.output.dense.weight",
        #     # "encoder.layer.2.output.dense.bias",
        #     # "encoder.layer.2.output.LayerNorm.weight",
        #     # "encoder.layer.2.output.LayerNorm.bias",
        # ]]},
        # "unfreeze_layers": {"values": [["encoder.layer.5"],["encoder.layer.4", "encoder.layer.5"],["encoder.layer.3", "encoder.layer.4", "encoder.layer.5"]]},
        "unfreeze_layers": {"values": [[]]},
        # "transformer_learning_rate": {
        #     "distribution": "uniform",
        #     "min": 1e-6,
        #     "max": 1e-4,
        # },
        "transformer_learning_rate": {"values": [0]},
        # FNN hyperparameters - Use a distribution for searching
        # "epochs": {"distribution": "int_uniform", "min": 999, "max": 1000},
        "epochs": {"values": [500]},
        "hidden_dim": {"distribution": "int_uniform", "min": 200, "max": 1200},
        "num_layers": {"distribution": "int_uniform", "min": 2, "max": 10},
        "activation": {"values": ["relu"]},
        "dropout": {"distribution": "uniform", "min": 0, "max": 0.7},
        "learning_rate": {"distribution": "uniform", "min": 1.0e-5, "max": 1.0e-3},
        # "learning_rate": {"values": [1.0e-4]},
        "regularization": {"distribution": "uniform", "min": 0.0, "max": 2.0e-1},
        # # FNN hyperparameters
        # "hidden_dim": {"values": [924]},
        # "num_layers": {"values": [4]},
        # "activation": {"values": ["relu"]},
        # "dropout": {"values": [0.05323]},
        # "learning_rate": {"values": [0.0006760719358722428]},
        # "learning_rate": {"values": [0]},
        # "regularization": {"values": [0.00064673]},
        # "epochs": {"values": [1000]},
    },
}

# Number of end2end hyperparameter sweep iterations
end2end_sweep_count = 50

# Fine-tuning hyperparameter sweep configuration for end2end models
end2end_finetune_sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_rmse",
        "goal": "minimize",
    },
    "parameters": {
        # Transformer layers to unfreeze - direct layer name lists
        "unfreeze_layers": {
            "values": [
                # Just the last layer (layer 11)
                [
                    "encoder.layer.5.attention.output.dense.weight",
                    "encoder.layer.5.attention.output.dense.bias",
                    "encoder.layer.5.attention.output.LayerNorm.weight",
                    "encoder.layer.5.attention.output.LayerNorm.bias",
                    "encoder.layer.5.intermediate.dense.weight",
                    "encoder.layer.5.intermediate.dense.bias",
                    "encoder.layer.5.output.dense.weight",
                    "encoder.layer.5.output.dense.bias",
                    "encoder.layer.5.output.LayerNorm.weight",
                    "encoder.layer.5.output.LayerNorm.bias",
                ],
                # Last 2 layers (layers 10, 11)
                [
                    "encoder.layer.5.attention.output.dense.weight",
                    "encoder.layer.5.attention.output.dense.bias",
                    "encoder.layer.5.attention.output.LayerNorm.weight",
                    "encoder.layer.5.attention.output.LayerNorm.bias",
                    "encoder.layer.5.intermediate.dense.weight",
                    "encoder.layer.5.intermediate.dense.bias",
                    "encoder.layer.5.output.dense.weight",
                    "encoder.layer.5.output.dense.bias",
                    "encoder.layer.5.output.LayerNorm.weight",
                    "encoder.layer.5.output.LayerNorm.bias",
                    "encoder.layer.4.attention.output.dense.weight",
                    "encoder.layer.4.attention.output.dense.bias",
                    "encoder.layer.4.attention.output.LayerNorm.weight",
                    "encoder.layer.4.attention.output.LayerNorm.bias",
                    "encoder.layer.4.intermediate.dense.weight",
                    "encoder.layer.4.intermediate.dense.bias",
                    "encoder.layer.4.output.dense.weight",
                    "encoder.layer.4.output.dense.bias",
                    "encoder.layer.4.output.LayerNorm.weight",
                    "encoder.layer.4.output.LayerNorm.bias",
                ],
                # Last 3 layers (layers 9, 10, 11)
                [
                    "encoder.layer.5.attention.output.dense.weight",
                    "encoder.layer.5.attention.output.dense.bias",
                    "encoder.layer.5.attention.output.LayerNorm.weight",
                    "encoder.layer.5.attention.output.LayerNorm.bias",
                    "encoder.layer.5.intermediate.dense.weight",
                    "encoder.layer.5.intermediate.dense.bias",
                    "encoder.layer.5.output.dense.weight",
                    "encoder.layer.5.output.dense.bias",
                    "encoder.layer.5.output.LayerNorm.weight",
                    "encoder.layer.5.output.LayerNorm.bias",
                    "encoder.layer.4.attention.output.dense.weight",
                    "encoder.layer.4.attention.output.dense.bias",
                    "encoder.layer.4.attention.output.LayerNorm.weight",
                    "encoder.layer.4.attention.output.LayerNorm.bias",
                    "encoder.layer.4.intermediate.dense.weight",
                    "encoder.layer.4.intermediate.dense.bias",
                    "encoder.layer.4.output.dense.weight",
                    "encoder.layer.4.output.dense.bias",
                    "encoder.layer.4.output.LayerNorm.weight",
                    "encoder.layer.4.output.LayerNorm.bias",
                    "encoder.layer.3.attention.output.dense.weight",
                    "encoder.layer.3.attention.output.dense.bias",
                    "encoder.layer.3.attention.output.LayerNorm.weight",
                    "encoder.layer.3.attention.output.LayerNorm.bias",
                    "encoder.layer.3.intermediate.dense.weight",
                    "encoder.layer.3.intermediate.dense.bias",
                    "encoder.layer.3.output.dense.weight",
                    "encoder.layer.3.output.dense.bias",
                    "encoder.layer.3.output.LayerNorm.weight",
                    "encoder.layer.3.output.LayerNorm.bias",
                ],
            ]
        },
        # Learning rates
        "transformer_learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-5,
        },
        "fnn_learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-8,
            "max": 1e-3,
        },
        # Training parameters
        "epochs": {
            "distribution": "int_uniform",
            "min": 50,
            "max": 300,
        },
        "batch_size": {"values": [16, 32, 64]},
        # Learning rate scheduling
        "lr_scheduler_patience": {
            "distribution": "int_uniform",
            "min": 25,
            "max": 100,
        },
        "lr_scheduler_factor": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.8,
        },
    },
}

# Number of fine-tuning hyperparameter sweep iterations
end2end_finetune_sweep_count = 30

# Fine-tuning configuration (for direct fine-tuning without sweep)
end2end_finetune_config = {
    "unfreeze_layers": [
        "encoder.layer.5.attention.output.dense.weight",
        "encoder.layer.5.attention.output.dense.bias",
        "encoder.layer.5.attention.output.LayerNorm.weight",
        "encoder.layer.5.attention.output.LayerNorm.bias",
        "encoder.layer.5.intermediate.dense.weight",
        "encoder.layer.5.intermediate.dense.bias",
        "encoder.layer.5.output.dense.weight",
        "encoder.layer.5.output.dense.bias",
        "encoder.layer.5.output.LayerNorm.weight",
        "encoder.layer.5.output.LayerNorm.bias",
        "encoder.layer.4.attention.output.dense.weight",
        "encoder.layer.4.attention.output.dense.bias",
        "encoder.layer.4.attention.output.LayerNorm.weight",
        "encoder.layer.4.attention.output.LayerNorm.bias",
        "encoder.layer.4.intermediate.dense.weight",
        "encoder.layer.4.intermediate.dense.bias",
        "encoder.layer.4.output.dense.weight",
        "encoder.layer.4.output.dense.bias",
        "encoder.layer.4.output.LayerNorm.weight",
        "encoder.layer.4.output.LayerNorm.bias",
    ],  # Transformer layers to unfreeze
    "transformer_learning_rate": 1e-5,  # Learning rate for transformer parameters
    "fnn_learning_rate": 1e-6,  # Learning rate for FNN parameters (set to reasonable value for adaptation)
    "epochs": 200,  # Number of fine-tuning epochs (set to reasonable value)
    "batch_size": 256,  # Batch size for fine-tuning
}

# Early stopping configuration
early_stopping_patience = 75  # Number of epochs to wait for improvement
early_stopping_min_delta = 0.001  # Minimum change to qualify as an improvement
early_stopping_restore_best_weights = True  # Whether to restore weights from best epoch

# Learning rate scheduler configuration
warmup_steps = 0  # Number of steps for warmup
lr_scheduler_mode = (
    "min"  # 'min' for reducing on loss plateau, 'max' for accuracy plateau
)
lr_scheduler_factor = 0.5  # Factor by which the learning rate will be reduced
lr_scheduler_patience = (
    early_stopping_patience // 3
)  # Number of epochs with no improvement after which learning rate will be reduced
lr_scheduler_verbose = True  # Whether to print a message when learning rate is reduced
lr_scheduler_min_lr = 1e-7  # Lower bound on the learning rate

# UMAP hyperparameter configuration (global for all plots)
umap_hyperparameters = {
    "n_neighbors": 25,
    "min_dist": 0.3,
    "random_state": 42,
    "metric": "euclidean",
}

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
import wandb
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from cd_host_guest.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    early_stopping_min_delta,
    early_stopping_patience,
    early_stopping_restore_best_weights,
    end2end_finetune_config,
    end2end_finetune_sweep_config,
    end2end_finetune_sweep_count,
    end2end_sweep_config,
    end2end_sweep_count,
    fnn_sweep_config,
    fnn_sweep_count,
    lgbm_sweep_count,
    lightgbm_sweep_config,
    lr_scheduler_factor,
    lr_scheduler_min_lr,
    lr_scheduler_mode,
    lr_scheduler_patience,
    lr_scheduler_verbose,
)

app = typer.Typer()


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        Check if early stopping criteria is met.

        Args:
            val_loss (float): Current validation loss
            model: PyTorch model to potentially save weights from

        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                # Save current best weights
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            # No improvement
            self.counter += 1

        return self.counter >= self.patience

    def restore_weights(self, model):
        """Restore the best weights to the model."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class RMSELoss(torch.nn.Module):
    """Standardized RMSE loss that all models will use"""

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, predictions, targets):
        return torch.sqrt(self.mse(predictions, targets))


class StandardizedMetrics:
    """Standardized metrics calculation for all models"""

    @staticmethod
    def calculate_standard_metrics(predictions, targets):
        """Calculate standardized metrics for all models"""
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        # MSE and RMSE (primary metrics)
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(predictions - targets))

        # R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Pearson correlation
        correlation = (
            np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
        )

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "correlation": float(correlation),
            "predictions_mean": float(np.mean(predictions)),
            "predictions_std": float(np.std(predictions)),
            "targets_mean": float(np.mean(targets)),
            "targets_std": float(np.std(targets)),
        }

    @staticmethod
    def calculate_rmse(predictions, targets):
        """Calculate RMSE using the same method as neural networks"""
        mse = np.mean((predictions - targets) ** 2)
        return np.sqrt(mse)


class Trainer:
    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        model_path: Path,
        random_seed: int = 42,  # Add a random seed parameter
    ):
        self.features_path = features_path
        self.labels_path = labels_path
        self.model_path = model_path
        self.random_seed = random_seed  # Store the random seed
        self.check_directories()
        self.features, self.labels = self.load_data()
        self.best_params = None
        np.random.seed(self.random_seed)  # Set the random seed

    def check_directories(self):
        """
        Check if the directories for features and labels paths exist.
        Create the directory for the model path if it does not exist.
        """
        if not self.features_path.parent.exists():
            msg = (
                f"Directory for features at {self.features_path.parent} does not exist."
            )
            raise FileNotFoundError(msg)

        if not self.labels_path.parent.exists():
            msg = f"Directory for labels at {self.labels_path.parent} does not exist."
            raise FileNotFoundError(msg)

        if not self.model_path.parent.exists():
            logger.info(f"Creating directory for model at {self.model_path.parent}")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self, file_type: str = "csv") -> tuple[pd.DataFrame, pd.Series]:
        """
        Load features and labels from a file. Supports CSV and .pt files.
        """
        if file_type == "csv":
            if not self.features_path.exists():
                msg = "Features file not found."
                raise FileNotFoundError(msg)

            features = pd.read_csv(self.features_path)

            if not self.labels_path.exists():
                msg = "Labels file not found."
                raise FileNotFoundError(msg)

            labels = pd.read_csv(self.labels_path)

        elif file_type == "pt":
            if not self.features_path.exists():
                msg = "Features file not found."
                raise FileNotFoundError(msg)

            features = torch.load(self.features_path)

            if not self.labels_path.exists():
                msg = "Labels file not found."
                raise FileNotFoundError(msg)

            labels = torch.load(self.labels_path)

        else:
            msg = f"Unsupported file type: {file_type}"
            raise ValueError(msg)

        return features, labels

    def split_data(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        features_file_type: str = "csv",
        labels_file_type: str = "csv",
        split_type: str = "stratified",
    ):
        """
        Split data into training, validation, and test sets and save them.
        Supports different file types for features and labels.

        Args:
            test_size: Fraction of data to use for test set
            val_size: Fraction of data to use for validation set
            features_file_type: File type for features ("csv" or "pt")
            labels_file_type: File type for labels ("csv" or "pt")
            split_type: Type of split to perform ("random" or "stratified")
        """
        # Check if 'Host' column is present for stratified split
        features = self.features
        stratify = None

        if split_type == "stratified":
            # If 'Host' column is not in features, try to get it from raw dataset
            if "Host" not in features.columns:
                logger.info(
                    "'Host' column not found in features. Attempting to append from raw dataset."
                )
                try:
                    # Load raw dataset that contains Host information
                    raw_data_path = Path(
                        RAW_DATA_DIR / "OpenCycloDB" / "Data" / "CDEnrichedData.csv"
                    )

                    # Ensure the path exists
                    if raw_data_path.exists():
                        raw_data = pd.read_csv(raw_data_path)

                        # The raw data contains DeltaG and Host columns
                        if "Host" in raw_data.columns:
                            # Rather than mapping by values, just append the Host column directly
                            # This assumes that the order of entries is preserved between raw_data and features
                            if len(raw_data) == len(features):
                                features["Host"] = raw_data["Host"].values
                                logger.info(
                                    "Successfully added 'Host' column from raw dataset."
                                )
                            else:
                                logger.warning(
                                    f"Data length mismatch: raw_data has {len(raw_data)} rows while features has {len(features)} rows"
                                )
                                logger.info("Falling back to random split.")
                                split_type = "random"
                        else:
                            logger.warning("Required columns not found in raw dataset.")
                            logger.info("Falling back to random split.")
                            split_type = "random"
                    else:
                        logger.warning(f"Raw data file not found at {raw_data_path}")
                        logger.info("Falling back to random split.")
                        split_type = "random"
                except Exception as e:
                    logger.error(f"Failed to append 'Host' column: {e}")
                    logger.info("Falling back to random split.")
                    split_type = "random"

            # Extract host type from the 'Host' column for better stratification
            if split_type == "stratified" and "Host" in features.columns:
                # Create Host_Type column based on the Host column value
                features["Host_Type"] = features["Host"].apply(
                    lambda x: "alpha"
                    if "alpha" in str(x).lower()
                    else (
                        "beta"
                        if "beta" in str(x).lower()
                        else "gamma"
                        if "gamma" in str(x).lower()
                        else "unknown"
                    )
                )
                # Use Host_Type for stratification
                stratify = features["Host_Type"]
                logger.info(
                    f"Using stratified split based on 'Host_Type' with {len(features['Host_Type'].unique())} unique host types."
                )

        # Split features and labels into temp and test sets
        if split_type == "stratified" and stratify is not None:
            # First split: separate out the test set
            temp_features, test_features, temp_labels, test_labels, temp_stratify, _ = (
                train_test_split(
                    features,
                    self.labels,
                    stratify,
                    test_size=test_size,
                    random_state=self.random_seed,
                )
            )

            # Second split: divide temp set into training and validation sets
            train_features, val_features, train_labels, val_labels = train_test_split(
                temp_features,
                temp_labels,
                test_size=val_size / (1 - test_size),
                random_state=self.random_seed,
                stratify=temp_stratify,
            )

            # Remove Host_Type column from each subset after splitting
            train_features.drop(columns=["Host_Type"], inplace=True)
            val_features.drop(columns=["Host_Type"], inplace=True)
            test_features.drop(columns=["Host_Type"], inplace=True)

            # Also remove Host column if it was added for stratification
            train_features.drop(columns=["Host"], inplace=True)
            val_features.drop(columns=["Host"], inplace=True)
            test_features.drop(columns=["Host"], inplace=True)

        elif split_type == "random":
            # Random split if stratified is not possible or not requested
            logger.info("Performing random split.")
            # First split: separate out the test set
            temp_features, test_features, temp_labels, test_labels = train_test_split(
                features,
                self.labels,
                test_size=test_size,
                random_state=self.random_seed,
            )

            # Second split: divide temp set into training and validation sets
            train_features, val_features, train_labels, val_labels = train_test_split(
                temp_features,
                temp_labels,
                test_size=val_size / (1 - test_size),
                random_state=self.random_seed,
            )
        else:
            msg = f"Unsupported split type: {split_type}"
            raise ValueError(msg)

        # Get the relative path of the features and labels
        relative_features_path = self.features_path.resolve().relative_to(
            INTERIM_DATA_DIR.resolve()
        )

        # Create corresponding directories in PROCESSED_DATA_DIR while preserving structure
        features_processed_dir = PROCESSED_DATA_DIR / relative_features_path.parent
        labels_processed_dir = PROCESSED_DATA_DIR / relative_features_path.parent

        # Create directories if they don't exist
        features_processed_dir.mkdir(parents=True, exist_ok=True)
        labels_processed_dir.mkdir(parents=True, exist_ok=True)

        # Save features based on specified file type
        if features_file_type == "csv":
            train_features.to_csv(
                features_processed_dir / "train_features.csv", index=False
            )
            val_features.to_csv(
                features_processed_dir / "val_features.csv", index=False
            )
            test_features.to_csv(
                features_processed_dir / "test_features.csv", index=False
            )
        elif features_file_type == "pt":
            torch.save(train_features, features_processed_dir / "train_features.pt")
            torch.save(val_features, features_processed_dir / "val_features.pt")
            torch.save(test_features, features_processed_dir / "test_features.pt")
        else:
            msg = f"Unsupported features file type: {features_file_type}"
            raise ValueError(msg)

        # Save labels based on specified file type
        if labels_file_type == "csv":
            train_labels.to_csv(labels_processed_dir / "train_labels.csv", index=False)
            val_labels.to_csv(labels_processed_dir / "val_labels.csv", index=False)
            test_labels.to_csv(labels_processed_dir / "test_labels.csv", index=False)
        elif labels_file_type == "pt":
            torch.save(train_labels, labels_processed_dir / "train_labels.pt")
            torch.save(val_labels, labels_processed_dir / "val_labels.pt")
            torch.save(test_labels, labels_processed_dir / "test_labels.pt")
        else:
            msg = f"Unsupported labels file type: {labels_file_type}"
            raise ValueError(msg)

        return (
            train_features,
            val_features,
            test_features,
            train_labels,
            val_labels,
            test_labels,
        )

    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning.
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)

    def train_final_model(self):
        """
        Train the final model with the best hyperparameters.
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)


class LightGBMTrainer(Trainer):
    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        model_path: Path,
        random_seed: int = 42,
    ):
        super().__init__(features_path, labels_path, model_path, random_seed)

    def hyperparameter_tuning(
        self,
        sweep_config: dict,
        train_features,
        train_labels,
        val_features,
        val_labels,
        data_type: str,
    ):
        """
        Run a wandb sweep using the parameters defined in the sweep_config dictionary.
        GPU acceleration is used if available.
        """

        def train():
            # Initialize a new wandb run
            wandb.init()
            config = wandb.config

            # Update parameters with the current sweep configuration
            params = {
                "seed": self.random_seed,
                # "device": "cuda" if torch.cuda.is_available() else "cpu",
                "device": "cpu",
                "objective": "regression",
                "metric": ["rmse", "mae"],
                "boosting_type": "gbdt",
                "verbosity": -1,
                "num_leaves": config.num_leaves,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "reg_alpha": config.reg_alpha,
                "reg_lambda": config.reg_lambda,
                "bagging_fraction": config.bagging_fraction,
                "feature_fraction": config.feature_fraction,
            }

            # Train the model with early stopping
            model = lgb.train(
                params,
                lgb.Dataset(train_features, label=train_labels),
                valid_sets=[lgb.Dataset(val_features, label=val_labels)],
                num_boost_round=config.n_estimators,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_patience),
                    lgb.log_evaluation(period=0),  # Suppress verbose output
                ],
            )
            # Calculate and log standardized metrics
            val_predictions = model.predict(
                val_features, num_iteration=model.best_iteration
            )
            train_predictions = model.predict(
                train_features, num_iteration=model.best_iteration
            )

            # Use standardized metrics calculation
            val_metrics = StandardizedMetrics.calculate_standard_metrics(
                val_predictions, val_labels
            )
            train_metrics = StandardizedMetrics.calculate_standard_metrics(
                train_predictions, train_labels
            )

            # Log comprehensive metrics
            wandb.log(
                {
                    "val_rmse": val_metrics["rmse"],
                    "train_rmse": train_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "train_mae": train_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                    "train_r2": train_metrics["r2"],
                    "val_correlation": val_metrics["correlation"],
                    "train_correlation": train_metrics["correlation"],
                    "predictions_mean": val_metrics["predictions_mean"],
                    "predictions_std": val_metrics["predictions_std"],
                }
            )

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=f"lgbm_{data_type}")

        # Run the sweep
        wandb.agent(sweep_id, function=train, count=lgbm_sweep_count)

        api = wandb.Api()
        sweep = api.sweep(f"lgbm_{data_type}/{sweep_id}")
        self.best_params = sweep.best_run().config
        logger.info(f"Best hyperparameters: {self.best_params}")
        wandb.finish()

    def train_final_model(self, train_features, train_labels):
        """
        Train the final LightGBM model with the best hyperparameters.
        """
        params = dict(self.best_params)
        final_model = lgb.train(
            params,
            lgb.Dataset(train_features, label=train_labels),
            valid_sets=[lgb.Dataset(train_features, label=train_labels)],
        )

        return final_model


class FNNTrainer(Trainer):
    def __init__(self, features_path: Path, labels_path: Path, model_path: Path):
        super().__init__(features_path, labels_path, model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ):
        """
        Build a simple feedforward neural network model with the specified number of layers.
        """
        layers = [torch.nn.Linear(input_dim, hidden_dim)]
        layers.append(
            torch.nn.BatchNorm1d(hidden_dim)
        )  # Add BatchNorm after first layer
        activation_fn = torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh()
        layers.append(activation_fn)
        layers.append(torch.nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(
                torch.nn.BatchNorm1d(hidden_dim)
            )  # Add BatchNorm after each hidden layer
            layers.append(activation_fn)
            layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*layers).to(self.device)

    def hyperparameter_tuning(
        self,
        sweep_config: dict,
        train_features,
        train_labels,
        val_features,
        val_labels,
        data_type: str,
    ):
        """
        Run a wandb sweep using the parameters defined in the sweep_config dictionary.
        """

        def train():
            # Initialize a new wandb run
            wandb.init()
            config = wandb.config

            torch.manual_seed(
                self.random_seed
            )  # Set the random seed for reproducibility

            # Build the model with the current sweep configuration
            self.build_model(
                train_features.shape[1],
                config.hidden_dim,
                1,
                config.num_layers,
                config.dropout,
                config.activation,
            )
            criterion = RMSELoss()
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.regularization,
            )

            # Convert data to tensors
            train_features_tensor = torch.tensor(
                train_features, dtype=torch.float32
            ).to(self.device)
            train_labels_tensor = torch.tensor(
                np.squeeze(
                    train_labels.values
                    if hasattr(train_labels, "values")
                    else train_labels
                ),
                dtype=torch.float32,
            ).to(self.device)
            val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(
                self.device
            )
            # Handle both pandas Series and numpy arrays for val_labels
            val_labels_tensor = torch.tensor(
                np.squeeze(
                    val_labels.values if hasattr(val_labels, "values") else val_labels
                ),
                dtype=torch.float32,
            ).to(self.device)

            # Initialize early stopping
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
            )

            # Initialize learning rate scheduler
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_mode,
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                verbose=lr_scheduler_verbose,
                min_lr=lr_scheduler_min_lr,
            )

            # Training loop
            for epoch in range(config.epochs):
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(train_features_tensor).squeeze()
                loss = criterion(outputs, train_labels_tensor)
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_features_tensor).squeeze()

                    # Use standardized metrics calculation
                    train_metrics = StandardizedMetrics.calculate_standard_metrics(
                        outputs.detach().cpu().numpy(),
                        train_labels_tensor.detach().cpu().numpy(),
                    )
                    val_metrics = StandardizedMetrics.calculate_standard_metrics(
                        val_outputs.detach().cpu().numpy(),
                        val_labels_tensor.detach().cpu().numpy(),
                    )

                    # Log standardized metrics
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_rmse": train_metrics[
                                "rmse"
                            ],  # Use standardized RMSE
                            "val_rmse": val_metrics["rmse"],  # Use standardized RMSE
                            "train_mae": train_metrics["mae"],
                            "val_mae": val_metrics["mae"],
                            "train_r2": train_metrics["r2"],
                            "val_r2": val_metrics["r2"],
                            "train_correlation": train_metrics["correlation"],
                            "val_correlation": val_metrics["correlation"],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "predictions_mean": val_metrics["predictions_mean"],
                            "predictions_std": val_metrics["predictions_std"],
                        }
                    )

                # Update learning rate scheduler using standardized RMSE
                scheduler.step(val_metrics["rmse"])

                # Check early stopping using standardized RMSE
                if early_stopping(val_metrics["rmse"], self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=f"fnn_{data_type}")

        # Run the sweep
        wandb.agent(sweep_id, function=train, count=fnn_sweep_count)

        api = wandb.Api()
        sweep = api.sweep(f"fnn_{data_type}/{sweep_id}")
        self.best_params = sweep.best_run().config
        logger.info(f"Best hyperparameters: {self.best_params}")
        wandb.finish()

    def train_final_model(
        self, train_features, train_labels, val_features=None, val_labels=None
    ):
        """
        Train the final feedforward neural network model with the best hyperparameters.
        """
        self.build_model(
            train_features.shape[1],
            self.best_params["hidden_dim"],
            1,
            self.best_params["num_layers"],
            self.best_params["dropout"],
            self.best_params["activation"],
        )
        criterion = RMSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.best_params["learning_rate"],
            weight_decay=self.best_params["regularization"],
        )

        # Convert data to tensors
        train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(
            self.device
        )
        train_labels_tensor = torch.tensor(
            train_labels.squeeze(), dtype=torch.float32
        ).to(self.device)

        # Initialize early stopping if validation data is provided
        early_stopping = None
        val_features_tensor = None
        val_labels_tensor = None

        if val_features is not None and val_labels is not None:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
            )
            val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(
                self.device
            )
            val_labels_tensor = torch.tensor(
                val_labels.squeeze(), dtype=torch.float32
            ).to(self.device)

        # Initialize learning rate scheduler if validation data is provided
        scheduler = None
        if val_features is not None and val_labels is not None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_mode,
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                verbose=lr_scheduler_verbose,
                min_lr=lr_scheduler_min_lr,
            )

        # Training loop
        for epoch in range(self.best_params["epochs"]):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(train_features_tensor).squeeze()
            loss = criterion(outputs, train_labels_tensor)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Check early stopping if validation data is provided
            if early_stopping is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_features_tensor).squeeze()

                    # Use standardized metrics calculation
                    val_metrics = StandardizedMetrics.calculate_standard_metrics(
                        val_outputs.detach().cpu().numpy(),
                        val_labels_tensor.detach().cpu().numpy(),
                    )
                    val_rmse = val_metrics["rmse"]

                # Update learning rate scheduler using standardized RMSE
                if scheduler is not None:
                    scheduler.step(val_rmse)

                if early_stopping(val_rmse, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        return self.model


class EndToEndTrainer(Trainer):
    def __init__(
        self,
        features_path: Path,
        labels_path: Path,
        model_path: Path,
        verbose: bool = False,
    ):
        super().__init__(features_path, labels_path, model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.verbose = verbose
        self.tokenizer = None
        self.transformer = None

    def load_pretrained_model(
        self, pretrained_model_path: Path, config_path: Path = None
    ):
        """
        Load a pre-trained end2end model for fine-tuning.

        Args:
            pretrained_model_path: Path to the pre-trained model state dict
            config_path: Optional path to the config file. If None, will try to infer from model path
        """
        if not pretrained_model_path.exists():
            msg = f"Pre-trained model not found at {pretrained_model_path}"
            raise FileNotFoundError(msg)

        # Load config if provided, otherwise try to infer
        if config_path is None:
            config_path = (
                pretrained_model_path.parent
                / f"{pretrained_model_path.stem}_config.json"
            )

        if not config_path.exists():
            msg = f"Config file not found at {config_path}"
            raise FileNotFoundError(msg)

        # Load hyperparameters from config
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Store the loaded config as best_params
        self.best_params = config

        # Build the model with the loaded hyperparameters
        self.build_model(
            fnn_hidden_dim=config["hidden_dim"],
            fnn_output_dim=1,
            fnn_num_layers=config["num_layers"],
            fnn_dropout=config["dropout"],
            fnn_activation=config["activation"],
            unfreeze_layers=config.get("unfreeze_layers", []),
            transformer_learning_rate=config.get("transformer_learning_rate", 1e-5),
        )

        # Load the pre-trained weights
        state_dict = torch.load(pretrained_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        logger.info(
            f"Successfully loaded pre-trained model from {pretrained_model_path}"
        )
        logger.info(f"Model configuration: {config}")

        return self.model

    def fine_tune_model(
        self,
        train_host_smiles: list[str],
        train_guest_smiles: list[str],
        train_labels: np.ndarray,
        val_host_smiles: list[str] = None,
        val_guest_smiles: list[str] = None,
        val_labels: np.ndarray = None,
        unfreeze_layers: list[str] = None,
        transformer_learning_rate: float = 1e-6,
        fnn_learning_rate: float = None,
        epochs: int = None,
        batch_size: int = 128,
        num_workers: int = 0,
        save_suffix: str = "finetuned",
        wandb_log: bool = True,
    ):
        """
        Fine-tune a pre-trained end2end model with unfrozen transformer layers.

        Args:
            train_host_smiles: List of host SMILES strings for training
            train_guest_smiles: List of guest SMILES strings for training
            train_labels: Training labels
            val_host_smiles: List of host SMILES strings for validation (optional)
            val_guest_smiles: List of guest SMILES strings for validation (optional)
            val_labels: Validation labels (optional)
            unfreeze_layers: List of transformer layer names to unfreeze for fine-tuning
            transformer_learning_rate: Learning rate for transformer parameters
            fnn_learning_rate: Learning rate for FNN parameters (if None, uses original rate)
            epochs: Number of fine-tuning epochs (if None, uses fewer epochs than original)
            batch_size: Batch size for training
            num_workers: Number of workers for DataLoader
            save_suffix: Suffix to add to the saved fine-tuned model filename
            wandb_log: Whether to log metrics to wandb

        Returns:
            The fine-tuned model
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_pretrained_model first.")

        if self.best_params is None:
            raise ValueError(
                "No configuration loaded. Call load_pretrained_model first."
            )

        # Use provided parameters or defaults from original training
        if fnn_learning_rate is None:
            fnn_learning_rate = self.best_params.get("learning_rate", 1e-3)

        if epochs is None:
            # Use fewer epochs for fine-tuning (typically 25-50% of original)
            original_epochs = self.best_params.get("epochs", 100)
            epochs = max(10, original_epochs // 3)

        if unfreeze_layers is None:
            unfreeze_layers = []

        logger.info(f"Starting fine-tuning with {epochs} epochs...")
        logger.info(f"Unfreezing transformer layers: {unfreeze_layers}")
        logger.info(
            f"Transformer LR: {transformer_learning_rate}, FNN LR: {fnn_learning_rate}"
        )

        # Initialize wandb for fine-tuning tracking
        if wandb_log:
            # Create a unique project name for fine-tuning
            original_model_name = Path(self.model_path).stem
            project_name = f"end2end_string_{save_suffix}"

            wandb.init(
                project=project_name,
                config={
                    "original_model": original_model_name,
                    "unfreeze_layers": unfreeze_layers,
                    "transformer_learning_rate": transformer_learning_rate,
                    "fnn_learning_rate": fnn_learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "save_suffix": save_suffix,
                    "fine_tuning": True,
                    # Include original hyperparameters for reference
                    **{f"original_{k}": v for k, v in self.best_params.items()},
                },
            )
            logger.info(f"Initialized wandb project: {project_name}")

        # Unfreeze specified transformer layers
        for name, param in self.transformer.named_parameters():
            param.requires_grad = False  # Re-freeze everything first
            for layer_name in unfreeze_layers:
                if layer_name in name:
                    param.requires_grad = True
                    if self.verbose:
                        logger.info(f"Unfrozen parameter: {name}")

        # Ensure FNN parameters are trainable during fine-tuning
        for name, param in self.model.fnn.named_parameters():
            param.requires_grad = True
            if self.verbose:
                logger.info(f"FNN parameter set to trainable: {name}")

        # Also ensure BatchNorm parameters are trainable
        for name, param in self.model.batchnorm.named_parameters():
            param.requires_grad = True
            if self.verbose:
                logger.info(f"BatchNorm parameter set to trainable: {name}")

        # Setup loss function
        criterion = RMSELoss()

        # Separate parameter groups for optimizer
        transformer_params = [
            p for n, p in self.transformer.named_parameters() if p.requires_grad
        ]
        fnn_params = [
            p for n, p in self.model.fnn.named_parameters() if p.requires_grad
        ]
        batchnorm_params = [
            p for n, p in self.model.batchnorm.named_parameters() if p.requires_grad
        ]

        # Setup optimizer with different learning rates
        optimizer_params = []
        if transformer_params:
            optimizer_params.append(
                {
                    "params": transformer_params,
                    "lr": transformer_learning_rate,
                }
            )
            logger.info(
                f"Added {len(transformer_params)} transformer parameters to optimizer"
            )

        # Combine FNN and BatchNorm parameters with the same learning rate
        fnn_and_bn_params = fnn_params + batchnorm_params
        if fnn_and_bn_params:
            optimizer_params.append(
                {
                    "params": fnn_and_bn_params,
                    "lr": fnn_learning_rate,
                    "weight_decay": self.best_params.get("regularization", 0.01),
                }
            )
            logger.info(
                f"Added {len(fnn_params)} FNN parameters and {len(batchnorm_params)} BatchNorm parameters to optimizer"
            )

        if not optimizer_params:
            raise ValueError(
                "No parameters to optimize. Check unfreeze_layers configuration and ensure FNN is trainable."
            )

        optimizer = torch.optim.Adam(optimizer_params)

        # Define Dataset and DataLoader classes (reuse from train_final_model)
        class HostGuestDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                host_smiles: list[str],
                guest_smiles: list[str],
                labels: np.ndarray,
            ):
                if not (len(host_smiles) == len(guest_smiles) == len(labels)):
                    raise ValueError("Input lists must have the same length.")
                self.host_smiles = host_smiles
                self.guest_smiles = guest_smiles
                self.labels = labels

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int):
                label = self.labels[idx]
                if isinstance(label, np.ndarray):
                    label = label.item()
                return self.host_smiles[idx], self.guest_smiles[idx], float(label)

        def collate_fn(batch):
            host, guest, label = zip(*batch, strict=False)
            return list(host), list(guest), torch.tensor(label, dtype=torch.float32)

        # Create training DataLoader
        train_dataset = HostGuestDataset(
            train_host_smiles, train_guest_smiles, train_labels
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        # Create validation DataLoader if validation data is provided
        val_loader = None
        if (
            val_host_smiles is not None
            and val_guest_smiles is not None
            and val_labels is not None
        ):
            val_dataset = HostGuestDataset(
                val_host_smiles, val_guest_smiles, val_labels
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        # Run initial validation before training to verify model loading
        if val_loader is not None:
            logger.info("🔍 Running initial validation before fine-tuning...")
            self.model.eval()
            initial_val_losses = []
            initial_val_outputs_all = []
            initial_val_labels_all = []

            with torch.no_grad():
                for val_host, val_guest, val_labels_batch in val_loader:
                    val_labels_batch = val_labels_batch.to(self.device)
                    val_outputs = self.model(val_host, val_guest)
                    if val_outputs.shape != val_labels_batch.shape:
                        val_outputs = val_outputs.view_as(val_labels_batch)
                    val_loss_batch = criterion(val_outputs, val_labels_batch)
                    initial_val_losses.append(val_loss_batch.item())
                    initial_val_outputs_all.append(val_outputs.cpu().numpy())
                    initial_val_labels_all.append(val_labels_batch.cpu().numpy())

            initial_val_outputs_all_np = np.concatenate(initial_val_outputs_all)
            initial_val_labels_all_np = np.concatenate(initial_val_labels_all)

            # Use standardized metrics calculation
            initial_val_metrics = StandardizedMetrics.calculate_standard_metrics(
                initial_val_outputs_all_np, initial_val_labels_all_np
            )

            initial_val_loss = initial_val_metrics["rmse"]
            initial_val_mae = initial_val_metrics["mae"]

            logger.info("📊 Initial validation performance (before fine-tuning):")
            logger.info(f"  Initial Val RMSE: {initial_val_loss:.4f}")
            logger.info(f"  Initial Val MAE: {initial_val_mae:.4f}")
            logger.info(f"  Initial Val R²: {initial_val_metrics['r2']:.4f}")
            logger.info(
                f"  Initial Val Correlation: {initial_val_metrics['correlation']:.4f}"
            )
            logger.info(
                f"  Val predictions range: [{initial_val_outputs_all_np.min():.4f}, {initial_val_outputs_all_np.max():.4f}]"
            )
            logger.info(
                f"  Val predictions mean: {initial_val_outputs_all_np.mean():.4f}"
            )
            logger.info(
                f"  Val labels range: [{initial_val_labels_all_np.min():.4f}, {initial_val_labels_all_np.max():.4f}]"
            )
            logger.info(f"  Val labels mean: {initial_val_labels_all_np.mean():.4f}")

            # Sample a few predictions vs labels for inspection
            sample_indices = np.random.choice(
                len(initial_val_outputs_all_np),
                min(5, len(initial_val_outputs_all_np)),
                replace=False,
            )
            logger.info("  Sample initial predictions vs labels:")
            for i in sample_indices:
                logger.info(
                    f"    Pred: {initial_val_outputs_all_np[i]:.4f}, Label: {initial_val_labels_all_np[i]:.4f}, Diff: {abs(initial_val_outputs_all_np[i] - initial_val_labels_all_np[i]):.4f}"
                )

        # Initialize early stopping (only if validation data is provided)
        early_stopping = None
        scheduler = None
        if val_loader is not None:
            early_stopping = EarlyStopping(
                patience=max(
                    5, early_stopping_patience
                ),  # Reduce patience for fine-tuning
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
            )

            # Initialize learning rate scheduler for fine-tuning
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_mode,
                factor=lr_scheduler_factor,
                patience=max(
                    2, lr_scheduler_patience // 3
                ),  # Reduce patience for fine-tuning
                verbose=lr_scheduler_verbose,
                min_lr=lr_scheduler_min_lr,
            )

        # Fine-tuning loop
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            train_outputs_all = []
            train_labels_all = []

            # Training phase
            for batch_host, batch_guest, batch_labels in train_loader:
                optimizer.zero_grad()
                batch_labels = batch_labels.to(self.device)
                outputs = self.model(batch_host, batch_guest)
                if outputs.shape != batch_labels.shape:
                    outputs = outputs.view_as(batch_labels)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                # Gradient clipping for fine-tuning stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                optimizer.step()

                train_losses.append(loss.item())
                train_outputs_all.append(outputs.detach().cpu().numpy())
                train_labels_all.append(batch_labels.detach().cpu().numpy())

            # Calculate training metrics using standardized calculation
            train_outputs_all_np = np.concatenate(train_outputs_all)
            train_labels_all_np = np.concatenate(train_labels_all)

            # Use standardized metrics calculation like original training
            train_metrics = StandardizedMetrics.calculate_standard_metrics(
                train_outputs_all_np, train_labels_all_np
            )

            train_loss = train_metrics["rmse"]  # Use standardized RMSE calculation
            train_mae = train_metrics["mae"]  # Use standardized MAE calculation

            # Validation phase (if validation data is provided)
            val_loss = None
            val_mae = None
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                val_outputs_all = []
                val_labels_all = []

                with torch.no_grad():
                    for val_host, val_guest, val_labels_batch in val_loader:
                        val_labels_batch = val_labels_batch.to(self.device)
                        val_outputs = self.model(val_host, val_guest)
                        if val_outputs.shape != val_labels_batch.shape:
                            val_outputs = val_outputs.view_as(val_labels_batch)
                        val_loss_batch = criterion(val_outputs, val_labels_batch)
                        val_losses.append(val_loss_batch.item())
                        val_outputs_all.append(val_outputs.cpu().numpy())
                        val_labels_all.append(val_labels_batch.cpu().numpy())

                val_outputs_all_np = np.concatenate(val_outputs_all)
                val_labels_all_np = np.concatenate(val_labels_all)

                # Use standardized metrics calculation like original training
                val_metrics = StandardizedMetrics.calculate_standard_metrics(
                    val_outputs_all_np, val_labels_all_np
                )

                val_loss = val_metrics["rmse"]  # Use standardized RMSE calculation
                val_mae = val_metrics["mae"]  # Use standardized MAE calculation

                # Debug validation data ranges (only on first epoch)
                if epoch == 0:
                    logger.info("🔍 Fine-tuning Validation Debug Info:")
                    logger.info(
                        f"  Val predictions range: [{val_outputs_all_np.min():.4f}, {val_outputs_all_np.max():.4f}]"
                    )
                    logger.info(
                        f"  Val predictions mean: {val_outputs_all_np.mean():.4f}"
                    )
                    logger.info(
                        f"  Val labels range: [{val_labels_all_np.min():.4f}, {val_labels_all_np.max():.4f}]"
                    )
                    logger.info(f"  Val labels mean: {val_labels_all_np.mean():.4f}")
                    logger.info(f"  Val MSE: {val_metrics['mse']:.4f}")
                    logger.info(f"  Val RMSE (standardized): {val_metrics['rmse']:.4f}")
                    logger.info(f"  Val MAE (standardized): {val_metrics['mae']:.4f}")
                    logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
                    logger.info(f"  Val Correlation: {val_metrics['correlation']:.4f}")

                    # Sample a few predictions vs labels for inspection
                    sample_indices = np.random.choice(
                        len(val_outputs_all_np),
                        min(5, len(val_outputs_all_np)),
                        replace=False,
                    )
                    logger.info("  Sample predictions vs labels:")
                    for i in sample_indices:
                        logger.info(
                            f"    Pred: {val_outputs_all_np[i]:.4f}, Label: {val_labels_all_np[i]:.4f}, Diff: {abs(val_outputs_all_np[i] - val_labels_all_np[i]):.4f}"
                        )

                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step(val_loss)

                # Check early stopping
                if early_stopping is not None and early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Log metrics to wandb
            if wandb_log:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_mae": train_mae,
                    "train_r2": train_metrics["r2"],
                    "train_correlation": train_metrics["correlation"],
                    "learning_rate_transformer": optimizer.param_groups[0]["lr"]
                    if transformer_params
                    else transformer_learning_rate,
                    "learning_rate_fnn": optimizer.param_groups[-1][
                        "lr"
                    ],  # FNN is the last param group
                }
                if val_loss is not None:
                    log_dict.update(
                        {
                            "val_loss": val_loss,
                            "val_mae": val_mae,
                            "val_r2": val_metrics["r2"],
                            "val_correlation": val_metrics["correlation"],
                            "predictions_mean": val_metrics["predictions_mean"],
                            "predictions_std": val_metrics["predictions_std"],
                        }
                    )
                wandb.log(log_dict)

            # Log progress more frequently for fine-tuning
            if epoch % 5 == 0 or epoch == epochs - 1:
                log_msg = f"Fine-tune Epoch {epoch+1}/{epochs}: "
                log_msg += f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
                logger.info(log_msg)

        # Restore best weights if early stopping was used
        if early_stopping_restore_best_weights:
            early_stopping.restore_weights(self.model)

        # Save the fine-tuned model with a different filename
        original_model_path = Path(self.model_path)
        finetuned_model_path = (
            original_model_path.parent / f"{original_model_path.stem}_{save_suffix}.pth"
        )
        finetuned_config_path = (
            original_model_path.parent
            / f"{original_model_path.stem}_{save_suffix}_config.json"
        )

        # Update config with fine-tuning parameters
        finetuned_config = dict(self.best_params)
        finetuned_config.update(
            {
                "fine_tuned": True,
                "unfrozen_layers": unfreeze_layers,
                "fine_tune_epochs": epochs,
                "fine_tune_transformer_lr": transformer_learning_rate,
                "fine_tune_fnn_lr": fnn_learning_rate,
                "original_model": str(original_model_path.name),
            }
        )

        # Save fine-tuned model and config
        torch.save(self.model.state_dict(), finetuned_model_path)
        with open(finetuned_config_path, "w", encoding="utf-8") as f:
            json.dump(finetuned_config, f, indent=2)

        logger.info(f"Fine-tuned model saved to {finetuned_model_path}")
        logger.info(f"Fine-tuned config saved to {finetuned_config_path}")
        logger.info("Fine-tuning completed!")

        # Finish wandb run
        if wandb_log:
            # Log final model artifacts
            wandb.log(
                {
                    "final_train_loss": train_loss,
                    "final_train_mae": train_mae,
                    "total_epochs": epoch + 1,
                    "early_stopped": early_stopping is not None
                    and early_stopping.counter >= early_stopping.patience,
                }
            )
            if val_loss is not None:
                wandb.log(
                    {
                        "final_val_loss": val_loss,
                        "final_val_mae": val_mae,
                    }
                )

            # Save model as wandb artifact
            artifact = wandb.Artifact(
                name=f"{original_model_path.stem}_{save_suffix}_model",
                type="model",
                description=f"Fine-tuned model with unfrozen layers: {unfreeze_layers}",
            )
            artifact.add_file(str(finetuned_model_path))
            artifact.add_file(str(finetuned_config_path))
            wandb.log_artifact(artifact)

            wandb.finish()
            logger.info("Wandb run completed and artifacts saved")

        return self.model

    def finetune_hyperparameter_tuning(
        self,
        train_host_smiles: list[str],
        train_guest_smiles: list[str],
        train_labels: np.ndarray,
        val_host_smiles: list[str],
        val_guest_smiles: list[str],
        val_labels: np.ndarray,
        sweep_config: dict,
        data_type: str = "string",
        pretrained_model_path: Path = None,
        config_path: Path = None,
        num_workers: int = 0,
    ):
        """
        Perform hyperparameter tuning for fine-tuning an end2end model.

        Args:
            train_host_smiles: Training host SMILES strings
            train_guest_smiles: Training guest SMILES strings
            train_labels: Training labels
            val_host_smiles: Validation host SMILES strings
            val_guest_smiles: Validation guest SMILES strings
            val_labels: Validation labels
            sweep_config: Wandb sweep configuration
            data_type: Data type identifier for wandb project
            pretrained_model_path: Path to pretrained model
            config_path: Path to pretrained model config
            num_workers: Number of workers for DataLoader
        """
        # Store original model path for restoration
        original_model_path = self.model_path

        def train():
            # Initialize wandb run
            wandb.init()
            config = wandb.config

            # Set random seed for reproducibility
            torch.manual_seed(self.random_seed)

            # Load the pre-trained model fresh for each run
            if pretrained_model_path and config_path:
                self.load_pretrained_model(pretrained_model_path, config_path)
            elif self.model is None:
                raise ValueError(
                    "No pretrained model loaded and no pretrained_model_path provided"
                )

            # Use unfreeze_layers directly from config
            unfreeze_layers = getattr(config, "unfreeze_layers", [])

            if isinstance(unfreeze_layers, list) and len(unfreeze_layers) > 0:
                logger.info(
                    f"Using direct layer list with {len(unfreeze_layers)} layers"
                )
            else:
                logger.info(
                    "No layers to unfreeze - keeping all transformer layers frozen"
                )

            # Setup model freezing/unfreezing
            # First freeze all transformer parameters
            for _name, param in self.transformer.named_parameters():
                param.requires_grad = False

            # Then unfreeze specified layers
            unfrozen_count = 0
            for name, param in self.transformer.named_parameters():
                for layer_name in unfreeze_layers:
                    if layer_name in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                        break

            logger.info(f"Unfrozen {unfrozen_count} transformer parameters")

            # Ensure FNN and BatchNorm parameters are trainable
            for param in self.model.fnn.parameters():
                param.requires_grad = True
            for param in self.model.batchnorm.parameters():
                param.requires_grad = True

            # Setup loss function
            criterion = RMSELoss()

            # Separate parameter groups for optimizer
            transformer_params = [
                p for p in self.transformer.parameters() if p.requires_grad
            ]
            fnn_params = list(self.model.fnn.parameters())
            batchnorm_params = list(self.model.batchnorm.parameters())

            # Setup optimizer with different learning rates
            optimizer_params = []
            if transformer_params:
                optimizer_params.append(
                    {
                        "params": transformer_params,
                        "lr": config.transformer_learning_rate,
                    }
                )

            fnn_and_bn_params = fnn_params + batchnorm_params
            if fnn_and_bn_params:
                optimizer_params.append(
                    {
                        "params": fnn_and_bn_params,
                        "lr": config.fnn_learning_rate,
                    }
                )

            if not optimizer_params:
                raise ValueError("No trainable parameters found")

            optimizer = torch.optim.Adam(optimizer_params)

            # Setup DataLoaders
            class HostGuestDataset(torch.utils.data.Dataset):
                def __init__(
                    self,
                    host_smiles: list[str],
                    guest_smiles: list[str],
                    labels: np.ndarray,
                ):
                    if not (len(host_smiles) == len(guest_smiles) == len(labels)):
                        raise ValueError("Input lists must have the same length.")
                    self.host_smiles = host_smiles
                    self.guest_smiles = guest_smiles
                    self.labels = labels

                def __len__(self) -> int:
                    return len(self.labels)

                def __getitem__(self, idx: int):
                    label = self.labels[idx]
                    if isinstance(label, np.ndarray):
                        label = label.item()
                    return self.host_smiles[idx], self.guest_smiles[idx], float(label)

            def collate_fn(batch):
                host, guest, label = zip(*batch, strict=False)
                return list(host), list(guest), torch.tensor(label, dtype=torch.float32)

            # Create DataLoaders
            train_dataset = HostGuestDataset(
                train_host_smiles, train_guest_smiles, train_labels
            )
            val_dataset = HostGuestDataset(
                val_host_smiles, val_guest_smiles, val_labels
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

            # Initialize early stopping with sweep-specific parameters
            early_stopping = EarlyStopping(
                patience=getattr(
                    config, "lr_scheduler_patience", early_stopping_patience
                ),
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
            )

            # Initialize learning rate scheduler with sweep-specific parameters
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_mode,
                factor=getattr(config, "lr_scheduler_factor", lr_scheduler_factor),
                patience=getattr(
                    config, "lr_scheduler_patience", lr_scheduler_patience
                ),
                verbose=lr_scheduler_verbose,
                min_lr=lr_scheduler_min_lr,
            )

            # Training loop
            for epoch in range(config.epochs):
                self.model.train()
                train_losses = []
                train_outputs_all = []
                train_labels_all = []

                # Training phase
                for batch_host, batch_guest, batch_labels in train_loader:
                    optimizer.zero_grad()
                    batch_labels = batch_labels.to(self.device)
                    outputs = self.model(batch_host, batch_guest)
                    if outputs.shape != batch_labels.shape:
                        outputs = outputs.view_as(batch_labels)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()

                    # Apply gradient clipping with fixed value
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    optimizer.step()
                    train_losses.append(loss.item())
                    train_outputs_all.append(outputs.detach().cpu().numpy())
                    train_labels_all.append(batch_labels.detach().cpu().numpy())

                # Calculate training metrics
                train_outputs_all_np = np.concatenate(train_outputs_all)
                train_labels_all_np = np.concatenate(train_labels_all)
                train_metrics = StandardizedMetrics.calculate_standard_metrics(
                    train_outputs_all_np, train_labels_all_np
                )

                # Validation phase
                self.model.eval()
                val_losses = []
                val_outputs_all = []
                val_labels_all = []

                with torch.no_grad():
                    for val_host, val_guest, val_labels_batch in val_loader:
                        val_labels_batch = val_labels_batch.to(self.device)
                        val_outputs = self.model(val_host, val_guest)
                        if val_outputs.shape != val_labels_batch.shape:
                            val_outputs = val_outputs.view_as(val_labels_batch)
                        val_loss_batch = criterion(val_outputs, val_labels_batch)
                        val_losses.append(val_loss_batch.item())
                        val_outputs_all.append(val_outputs.cpu().numpy())
                        val_labels_all.append(val_labels_batch.cpu().numpy())

                # Calculate validation metrics
                val_outputs_all_np = np.concatenate(val_outputs_all)
                val_labels_all_np = np.concatenate(val_labels_all)
                val_metrics = StandardizedMetrics.calculate_standard_metrics(
                    val_outputs_all_np, val_labels_all_np
                )

                # Use standardized RMSE for consistency
                val_rmse = val_metrics["rmse"]

                # Update learning rate scheduler
                scheduler.step(val_rmse)

                # Check early stopping
                if early_stopping(val_rmse, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                # Log metrics to wandb
                log_dict = {
                    "train_rmse": train_metrics["rmse"],
                    "val_rmse": val_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "val_mae": val_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "train_correlation": train_metrics["correlation"],
                    "val_correlation": val_metrics["correlation"],
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "unfrozen_transformer_params": unfrozen_count,
                }
                wandb.log(log_dict)

            # Restore best weights if early stopping was used
            if early_stopping_restore_best_weights:
                early_stopping.restore_weights(self.model)

        # Run the sweep
        project_name = f"end2end_{data_type}_finetune_sweep"
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train, count=end2end_finetune_sweep_count)

        # Get best parameters
        api = wandb.Api()
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        self.best_finetune_params = sweep.best_run().config
        logger.info(f"Best fine-tuning hyperparameters: {self.best_finetune_params}")

        # Restore original model path
        self.model_path = original_model_path
        wandb.finish()

    def get_best_finetune_params_from_project(
        self,
        project_name: str,
        data_type: str = "string",
    ) -> dict:
        """
        Retrieve the best hyperparameters from a completed fine-tuning sweep project.

        Args:
            project_name: Name of the wandb project containing the sweep
            data_type: Data type identifier for wandb project

        Returns:
            Dictionary containing the best hyperparameters
        """
        api = wandb.Api()

        # Get all runs from the project
        full_project_name = (
            f"end2end_{data_type}_finetune_sweep"
            if project_name == "auto"
            else project_name
        )
        runs = api.runs(full_project_name, filters={"state": "finished"})

        if not runs:
            error_msg = f"No finished runs found in project {full_project_name}"
            raise ValueError(error_msg)

        # Find best run based on val_rmse
        best_run = None
        best_val_rmse = float("inf")

        for run in runs:
            if "val_rmse" in run.summary:
                val_rmse = run.summary["val_rmse"]
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_run = run

        if best_run is None:
            error_msg = (
                f"No runs with val_rmse metric found in project {full_project_name}"
            )
            raise ValueError(error_msg)

        logger.info(
            f"Best run found: {best_run.name} with val_rmse: {best_val_rmse:.4f}"
        )
        self.best_finetune_params = best_run.config

        return self.best_finetune_params

    def get_best_finetune_params_from_sweep(
        self,
        sweep_id: str,
        project_name: str = None,
    ) -> dict:
        """
        Retrieve the best hyperparameters from a specific sweep.

        Args:
            sweep_id: The sweep ID (can be just the ID or full path)
            project_name: Optional project name if sweep_id is just the ID

        Returns:
            Dictionary containing the best hyperparameters
        """
        api = wandb.Api()

        # Handle both full sweep paths and just IDs
        if "/" in sweep_id:
            sweep_path = sweep_id
        else:
            if project_name is None:
                raise ValueError(
                    "project_name must be provided when using just sweep ID"
                )
            sweep_path = f"{project_name}/{sweep_id}"

        try:
            sweep = api.sweep(sweep_path)
            best_run = sweep.best_run()

            if best_run is None:
                error_msg = f"No best run found for sweep {sweep_path}"
                raise ValueError(error_msg)

            logger.info(f"Best run from sweep: {best_run.name}")
            if "val_rmse" in best_run.summary:
                logger.info(f"Best val_rmse: {best_run.summary['val_rmse']:.4f}")

            self.best_finetune_params = best_run.config
            return self.best_finetune_params

        except Exception as e:
            error_msg = f"Failed to retrieve sweep {sweep_path}: {e}"
            raise ValueError(error_msg) from e

    def train_final_finetuned_model(
        self,
        train_host_smiles: list[str],
        train_guest_smiles: list[str],
        train_labels: np.ndarray,
        val_host_smiles: list[str] = None,
        val_guest_smiles: list[str] = None,
        val_labels: np.ndarray = None,
        pretrained_model_path: Path = None,
        config_path: Path = None,
        save_suffix: str = "finetuned_final",
        wandb_log: bool = True,
        num_workers: int = 0,
    ):
        """
        Train a final fine-tuned model using the best hyperparameters from a sweep.

        Args:
            train_host_smiles: Training host SMILES strings
            train_guest_smiles: Training guest SMILES strings
            train_labels: Training labels
            val_host_smiles: Validation host SMILES strings (optional)
            val_guest_smiles: Validation guest SMILES strings (optional)
            val_labels: Validation labels (optional)
            pretrained_model_path: Path to pretrained model
            config_path: Path to pretrained model config
            save_suffix: Suffix for saved model filename
            wandb_log: Whether to log to wandb
            num_workers: Number of workers for DataLoader

        Returns:
            The trained model
        """
        if self.best_finetune_params is None:
            raise ValueError(
                "No best fine-tuning parameters found. Run finetune_hyperparameter_tuning or get_best_finetune_params_from_project first."
            )

        logger.info("Training final fine-tuned model with best hyperparameters...")
        logger.info(f"Best parameters: {self.best_finetune_params}")

        # Load the pre-trained model if paths provided
        if pretrained_model_path and config_path:
            self.load_pretrained_model(pretrained_model_path, config_path)
        elif self.model is None:
            raise ValueError(
                "No pretrained model loaded and no pretrained_model_path provided"
            )

        # Extract best parameters
        unfreeze_layers = self.best_finetune_params.get("unfreeze_layers", [])
        transformer_learning_rate = self.best_finetune_params.get(
            "transformer_learning_rate", 1e-6
        )
        fnn_learning_rate = self.best_finetune_params.get("fnn_learning_rate", 1e-5)
        epochs = self.best_finetune_params.get("epochs", 100)
        batch_size = self.best_finetune_params.get("batch_size", 64)

        # Use the fine_tune_model method with best parameters
        return self.fine_tune_model(
            train_host_smiles=train_host_smiles,
            train_guest_smiles=train_guest_smiles,
            train_labels=train_labels,
            val_host_smiles=val_host_smiles,
            val_guest_smiles=val_guest_smiles,
            val_labels=val_labels,
            unfreeze_layers=unfreeze_layers,
            transformer_learning_rate=transformer_learning_rate,
            fnn_learning_rate=fnn_learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            save_suffix=save_suffix,
            wandb_log=wandb_log,
            num_workers=num_workers,
        )

    def build_model(
        self,
        fnn_hidden_dim: int = 128,
        fnn_output_dim: int = 1,
        fnn_num_layers: int = 2,
        fnn_dropout: float = 0.2,
        fnn_activation: str = "relu",
        unfreeze_layers: list[str] = None,
        transformer_learning_rate: float = 1e-5,
        verbose: bool = None,
    ):
        """
        Build an end-to-end model that encodes host and guest SMILES using a pretrained transformer and passes their concatenated representations through a feedforward neural network.
        Allows selective unfreezing of transformer layers and custom transformer learning rate.

        Args:
            fnn_hidden_dim: Hidden dimension for the FNN
            fnn_output_dim: Output dimension for the FNN
            fnn_num_layers: Number of layers in the FNN
            fnn_dropout: Dropout rate for the FNN
            fnn_activation: Activation function for the FNN
            unfreeze_layers: List of transformer layer names to unfreeze
            transformer_learning_rate: Learning rate for transformer parameters
            verbose: Whether to print detailed model information
        """
        # Load pretrained tokenizer and model for SMILES
        tokenizer, transformer = self.build_SMILESTokenizer_layer()
        # Freeze all transformer parameters by default
        for param in transformer.parameters():
            param.requires_grad = False
        # Unfreeze specified layers if provided
        if unfreeze_layers is not None and len(unfreeze_layers) > 0:
            for name, param in transformer.named_parameters():
                for layer_name in unfreeze_layers:
                    if layer_name in name:
                        param.requires_grad = True

        # Define a module to encode a SMILES string using the transformer
        class SMILESEncoder(nn.Module):
            def __init__(self, transformer, tokenizer, device):
                super().__init__()
                self.transformer = transformer
                self.tokenizer = tokenizer
                self.device = device

            def forward(self, smiles_list):
                # Tokenize and encode a batch of SMILES strings
                encoded = self.tokenizer(
                    smiles_list,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                return cls_embedding

        # Build FNN for concatenated embeddings
        transformer_dim = transformer.config.hidden_size
        fnn_input_dim = 2 * transformer_dim
        fnn = self.build_FNN_layer(
            self.device,
            input_dim=fnn_input_dim,
            hidden_dim=fnn_hidden_dim,
            output_dim=fnn_output_dim,
            num_layers=fnn_num_layers,
            dropout=fnn_dropout,
            activation=fnn_activation,
        )

        # Define the full end-to-end model
        class EndToEndModel(nn.Module):
            def __init__(self, host_encoder, guest_encoder, fnn, concat_dim):
                super().__init__()
                self.host_encoder = host_encoder
                self.guest_encoder = guest_encoder
                self.fnn = fnn
                self.batchnorm = nn.BatchNorm1d(concat_dim)

            def forward(self, host_smiles, guest_smiles):
                host_emb = self.host_encoder(host_smiles)
                guest_emb = self.guest_encoder(guest_smiles)
                x = torch.cat([host_emb, guest_emb], dim=1)
                x = self.batchnorm(x)
                out = self.fnn(x)
                return out.squeeze(-1)

        host_encoder = SMILESEncoder(transformer, tokenizer, self.device)
        guest_encoder = SMILESEncoder(transformer, tokenizer, self.device)

        model = EndToEndModel(host_encoder, guest_encoder, fnn, fnn_input_dim).to(
            self.device
        )
        self.model = model
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.unfreeze_layers = unfreeze_layers
        self.transformer_learning_rate = transformer_learning_rate

        # Determine verbosity: argument overrides instance attribute
        _verbose = self.verbose if verbose is None else verbose
        if _verbose:
            print("\n========== End-to-End Model Architecture ==========")
            print(self.model)
            print("\n========== Host Encoder Architecture ==============")
            print(host_encoder)
            print("\n========== Guest Encoder Architecture =============")
            print(guest_encoder)
            print("\n========== Transformer Encoder Architecture ========")
            print(self.transformer)
            print("==================================================\n")

            self.print_transformer_layers()
        return model

    def hyperparameter_tuning(
        self,
        sweep_config: dict,
        train_host_smiles,
        train_guest_smiles,
        train_labels,
        val_host_smiles,
        val_guest_smiles,
        val_labels,
        data_type: str,
        batch_size: int = 256,
        num_workers: int = 0,
    ):
        """
        Perform hyperparameter tuning for the end-to-end model using all sweep config parameters.
        Uses a DataLoader for batching to reduce memory usage.
        """

        # Define a custom Dataset for host/guest SMILES and labels
        class HostGuestDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                host_smiles: list[str],
                guest_smiles: list[str],
                labels: np.ndarray,
            ):
                if not (len(host_smiles) == len(guest_smiles) == len(labels)):
                    raise ValueError("Input lists must have the same length.")
                self.host_smiles = host_smiles
                self.guest_smiles = guest_smiles
                self.labels = labels

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int):
                label = self.labels[idx]
                # If label is a numpy array, extract the scalar
                if isinstance(label, np.ndarray):
                    label = label.item()
                return self.host_smiles[idx], self.guest_smiles[idx], float(label)

        # Collate function for batching SMILES and labels
        def collate_fn(batch):
            host, guest, label = zip(*batch, strict=False)
            return list(host), list(guest), torch.tensor(label, dtype=torch.float32)

        def train():
            wandb.init()
            config = wandb.config
            torch.manual_seed(self.random_seed)
            # Build the model with current sweep config, including transformer and FNN params
            self.build_model(
                fnn_hidden_dim=config.hidden_dim,
                fnn_output_dim=1,
                fnn_num_layers=config.num_layers,
                fnn_dropout=config.dropout,
                fnn_activation=config.activation,
                unfreeze_layers=getattr(config, "unfreeze_layers", []),
                transformer_learning_rate=getattr(
                    config, "transformer_learning_rate", 1e-5
                ),
            )
            criterion = RMSELoss()
            # Separate parameter groups for optimizer: transformer and FNN
            transformer_params = [
                p for n, p in self.transformer.named_parameters() if p.requires_grad
            ]
            fnn_params = [p for n, p in self.model.fnn.named_parameters()]
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": transformer_params,
                        "lr": config.transformer_learning_rate,
                    },
                    {
                        "params": fnn_params,
                        "lr": config.learning_rate,
                        "weight_decay": config.regularization,
                    },
                ]
            )

            # Create DataLoaders for training and validation
            train_dataset = HostGuestDataset(
                train_host_smiles, train_guest_smiles, train_labels
            )
            val_dataset = HostGuestDataset(
                val_host_smiles, val_guest_smiles, val_labels
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

            # Initialize early stopping
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
            )

            # Initialize learning rate scheduler
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=lr_scheduler_mode,
                factor=lr_scheduler_factor,
                patience=lr_scheduler_patience,
                verbose=lr_scheduler_verbose,
                min_lr=lr_scheduler_min_lr,
            )

            for epoch in range(config.epochs):
                self.model.train()
                train_losses = []
                train_outputs_all = []
                train_labels_all = []
                for batch_host, batch_guest, batch_labels in train_loader:
                    optimizer.zero_grad()
                    batch_labels = batch_labels.to(self.device)
                    outputs = self.model(batch_host, batch_guest)
                    if outputs.shape != batch_labels.shape:
                        outputs = outputs.view_as(batch_labels)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    train_losses.append(loss.item())
                    train_outputs_all.append(outputs.detach().cpu().numpy())
                    train_labels_all.append(batch_labels.detach().cpu().numpy())

                # Aggregate train outputs and labels
                train_outputs_all_np = np.concatenate(train_outputs_all)
                train_labels_all_np = np.concatenate(train_labels_all)

                # Use standardized metrics calculation
                train_metrics = StandardizedMetrics.calculate_standard_metrics(
                    train_outputs_all_np, train_labels_all_np
                )

                # Validation
                self.model.eval()
                val_losses = []
                val_outputs_all = []
                val_labels_all = []
                with torch.no_grad():
                    for val_host, val_guest, val_labels_batch in val_loader:
                        val_labels_batch = val_labels_batch.to(self.device)
                        val_outputs = self.model(val_host, val_guest)
                        if val_outputs.shape != val_labels_batch.shape:
                            val_outputs = val_outputs.view_as(val_labels_batch)
                        val_loss = criterion(val_outputs, val_labels_batch)
                        val_losses.append(val_loss.item())
                        val_outputs_all.append(val_outputs.detach().cpu().numpy())
                        val_labels_all.append(val_labels_batch.detach().cpu().numpy())

                # Aggregate validation outputs and labels
                val_outputs_all_np = np.concatenate(val_outputs_all)
                val_labels_all_np = np.concatenate(val_labels_all)

                # Use standardized metrics calculation
                val_metrics = StandardizedMetrics.calculate_standard_metrics(
                    val_outputs_all_np, val_labels_all_np
                )

                # Use standardized RMSE for consistency
                val_rmse = val_metrics["rmse"]

                # Update learning rate scheduler using standardized RMSE
                scheduler.step(val_rmse)

                # Check early stopping using standardized RMSE
                if early_stopping(val_rmse, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                # Log standardized metrics
                log_dict = {
                    "train_rmse": train_metrics["rmse"],  # Use standardized RMSE
                    "val_rmse": val_metrics["rmse"],  # Use standardized RMSE
                    "train_mae": train_metrics["mae"],
                    "val_mae": val_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "val_r2": val_metrics["r2"],
                    "train_correlation": train_metrics["correlation"],
                    "val_correlation": val_metrics["correlation"],
                    "epoch": epoch + 1,
                    "predictions_mean": val_metrics["predictions_mean"],
                    "predictions_std": val_metrics["predictions_std"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_dict)

            # Restore best weights if early stopping was used
            if early_stopping_restore_best_weights:
                early_stopping.restore_weights(self.model)

        sweep_id = wandb.sweep(sweep_config, project=f"end2end_{data_type}")
        wandb.agent(sweep_id, function=train, count=end2end_sweep_count)
        api = wandb.Api()
        sweep = api.sweep(f"end2end_{data_type}/{sweep_id}")
        self.best_params = sweep.best_run().config
        logger.info(f"Best hyperparameters: {self.best_params}")
        wandb.finish()

    def train_final_model(
        self,
        train_host_smiles: list[str],
        train_guest_smiles: list[str],
        train_labels: np.ndarray,
        val_host_smiles: list[str] = None,
        val_guest_smiles: list[str] = None,
        val_labels: np.ndarray = None,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        """
        Train the final end-to-end model with the best hyperparameters.

        Args:
            train_host_smiles: List of host SMILES strings for training
            train_guest_smiles: List of guest SMILES strings for training
            train_labels: Training labels
            val_host_smiles: List of host SMILES strings for validation (optional)
            val_guest_smiles: List of guest SMILES strings for validation (optional)
            val_labels: Validation labels (optional)
            batch_size: Batch size for training
            num_workers: Number of workers for DataLoader

        Returns:
            The trained model
        """
        if self.best_params is None:
            raise ValueError(
                "No best parameters found. Run hyperparameter_tuning first."
            )

        logger.info("Training final end-to-end model with best hyperparameters...")
        logger.info(f"Best parameters: {self.best_params}")

        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)

        # Build the model with best hyperparameters
        self.build_model(
            fnn_hidden_dim=self.best_params["hidden_dim"],
            fnn_output_dim=1,
            fnn_num_layers=self.best_params["num_layers"],
            fnn_dropout=self.best_params["dropout"],
            fnn_activation=self.best_params["activation"],
            unfreeze_layers=self.best_params.get("unfreeze_layers", []),
            transformer_learning_rate=self.best_params.get(
                "transformer_learning_rate", 1e-5
            ),
        )

        # Setup loss function
        criterion = RMSELoss()

        # Separate parameter groups for optimizer: transformer and FNN
        transformer_params = [
            p for n, p in self.transformer.named_parameters() if p.requires_grad
        ]
        fnn_params = [p for n, p in self.model.fnn.named_parameters()]

        # Setup optimizer with separate learning rates
        optimizer = torch.optim.Adam(
            [
                {
                    "params": transformer_params,
                    "lr": self.best_params.get("transformer_learning_rate", 1e-5),
                },
                {
                    "params": fnn_params,
                    "lr": self.best_params["learning_rate"],
                    "weight_decay": self.best_params["regularization"],
                },
            ]
        )

        # Define Dataset and DataLoader classes
        class HostGuestDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                host_smiles: list[str],
                guest_smiles: list[str],
                labels: np.ndarray,
            ):
                if not (len(host_smiles) == len(guest_smiles) == len(labels)):
                    raise ValueError("Input lists must have the same length.")
                self.host_smiles = host_smiles
                self.guest_smiles = guest_smiles
                self.labels = labels

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int):
                label = self.labels[idx]
                if isinstance(label, np.ndarray):
                    label = label.item()
                return self.host_smiles[idx], self.guest_smiles[idx], float(label)

        def collate_fn(batch):
            host, guest, label = zip(*batch, strict=False)
            return list(host), list(guest), torch.tensor(label, dtype=torch.float32)

        # Create training DataLoader
        train_dataset = HostGuestDataset(
            train_host_smiles, train_guest_smiles, train_labels
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        # Create validation DataLoader if validation data is provided
        val_loader = None
        if (
            val_host_smiles is not None
            and val_guest_smiles is not None
            and val_labels is not None
        ):
            val_dataset = HostGuestDataset(
                val_host_smiles, val_guest_smiles, val_labels
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        # Initialize early stopping (only if validation data is provided)
        early_stopping = None
        if val_loader is not None:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=early_stopping_restore_best_weights,
            )

        # Training loop
        for epoch in range(self.best_params["epochs"]):
            self.model.train()
            train_losses = []
            train_outputs_all = []
            train_labels_all = []

            # Training phase
            for batch_host, batch_guest, batch_labels in train_loader:
                optimizer.zero_grad()
                batch_labels = batch_labels.to(self.device)
                outputs = self.model(batch_host, batch_guest)
                if outputs.shape != batch_labels.shape:
                    outputs = outputs.view_as(batch_labels)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_outputs_all.append(outputs.detach().cpu().numpy())
                train_labels_all.append(batch_labels.detach().cpu().numpy())

            # Calculate training metrics using standardized calculation
            train_outputs_all_np = np.concatenate(train_outputs_all)
            train_labels_all_np = np.concatenate(train_labels_all)

            # Use standardized metrics calculation
            train_metrics = StandardizedMetrics.calculate_standard_metrics(
                train_outputs_all_np, train_labels_all_np
            )

            train_loss = train_metrics["rmse"]  # Use standardized RMSE calculation
            train_mae = train_metrics["mae"]  # Use standardized MAE calculation

            # Validation phase (if validation data is provided)
            val_loss = None
            val_mae = None
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                val_outputs_all = []
                val_labels_all = []

                with torch.no_grad():
                    for val_host, val_guest, val_labels_batch in val_loader:
                        val_labels_batch = val_labels_batch.to(self.device)
                        val_outputs = self.model(val_host, val_guest)
                        if val_outputs.shape != val_labels_batch.shape:
                            val_outputs = val_outputs.view_as(val_labels_batch)
                        val_loss_batch = criterion(val_outputs, val_labels_batch)
                        val_losses.append(val_loss_batch.item())
                        val_outputs_all.append(val_outputs.cpu().numpy())
                        val_labels_all.append(val_labels_batch.cpu().numpy())

                val_outputs_all_np = np.concatenate(val_outputs_all)
                val_labels_all_np = np.concatenate(val_labels_all)

                # Use standardized metrics calculation
                val_metrics = StandardizedMetrics.calculate_standard_metrics(
                    val_outputs_all_np, val_labels_all_np
                )

                val_loss = val_metrics["rmse"]  # Use standardized RMSE calculation
                val_mae = val_metrics["mae"]  # Use standardized MAE calculation

                # Check early stopping
                if early_stopping is not None and early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Log progress periodically
            if epoch % 50 == 0 or epoch == self.best_params["epochs"] - 1:
                log_msg = f"Epoch {epoch+1}/{self.best_params['epochs']}: "
                log_msg += f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
                logger.info(log_msg)

        # Restore best weights if early stopping was used
        if early_stopping is not None and early_stopping_restore_best_weights:
            early_stopping.restore_weights(self.model)

        logger.info("Final model training completed!")
        return self.model

    def build_FNN_layer(
        self,
        device: torch.device,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ):
        """
        Build a simple feedforward neural network model with the specified number of layers.
        """
        layers = [torch.nn.Linear(input_dim, hidden_dim)]
        layers.append(nn.BatchNorm1d(hidden_dim))
        activation_fn = torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh()
        layers.append(activation_fn)
        layers.append(torch.nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation_fn)
            layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        return torch.nn.Sequential(*layers).to(device)

    def build_SMILESTokenizer_layer(self):
        """
        Build a differentiable SMILES tokenizer and encoder model using a pretrained transformer.
        Returns:
            tokenizer: The AutoTokenizer instance for SMILES strings.
            model: The AutoModel instance, set to eval mode and moved to the appropriate device.
        """

        # Load pretrained tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model

    def extract_embeddings(
        self,
        data_path: Path,
        model_path: Path = None,
        output_path: Path = None,
        skip_model_loading: bool = False,
    ):
        """
        Extract embeddings from the SMILES tokenizer before FNN processing.

        This method extracts separate embeddings for host and guest SMILES from end2end models.
        For fine-tuned models, it uses the respective transformer heads for host and guest.

        Args:
            data_path: Path to input CSV file with Host_SMILES and Guest_SMILES columns
            model_path: Path to trained model (can be None if skip_model_loading=True)
            output_path: Path to save embeddings CSV (defaults to same directory as data_path)
            skip_model_loading: If True, assumes model is already loaded and configured
        """
        logger.info(
            f"🔬 Extracting embeddings from model: {model_path if model_path else 'pre-loaded model'}"
        )

        # Set default output path if not provided
        if output_path is None:
            output_path = data_path.parent / f"{data_path.stem}_embeddings.csv"

        # Load the trained model only if not skipping model loading
        if not skip_model_loading and model_path is not None:
            if not model_path.exists():
                error_msg = f"Model not found: {model_path}"
                raise FileNotFoundError(error_msg)

            checkpoint = torch.load(model_path, map_location=self.device)

            # Load model state
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Load and prepare data
        logger.info(f"📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Prepare SMILES data - end2end models expect Host_SMILES and Guest_SMILES
        required_columns = ["Host_SMILES", "Guest_SMILES"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            # Check if single 'smiles' column exists as fallback
            if "smiles" in df.columns and len(missing_columns) == len(required_columns):
                logger.warning(
                    "⚠️ Host_SMILES and Guest_SMILES columns not found, using 'smiles' column for both host and guest"
                )
                host_smiles_list = df["smiles"].tolist()
                guest_smiles_list = df["smiles"].tolist()  # Use same SMILES for both
            else:
                error_msg = f"Missing required columns: {missing_columns}. End2end models require 'Host_SMILES' and 'Guest_SMILES' columns."
                raise ValueError(error_msg)
        else:
            host_smiles_list = df["Host_SMILES"].tolist()
            guest_smiles_list = df["Guest_SMILES"].tolist()

        logger.info(f"📊 Processing {len(host_smiles_list)} host-guest pairs")

        # Extract embeddings in batches
        logger.info("🧠 Extracting embeddings...")
        host_embeddings_list = []
        guest_embeddings_list = []

        batch_size = min(32, len(host_smiles_list))

        with torch.no_grad():
            for i in tqdm(
                range(0, len(host_smiles_list), batch_size),
                desc="Extracting embeddings",
            ):
                batch_host_smiles = host_smiles_list[i : i + batch_size]
                batch_guest_smiles = guest_smiles_list[i : i + batch_size]

                # Extract host embeddings
                host_encoded = self.tokenizer(
                    batch_host_smiles,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )

                host_input_ids = host_encoded["input_ids"].to(self.device)
                host_attention_mask = host_encoded["attention_mask"].to(self.device)

                # Forward pass through transformer for host SMILES
                host_transformer_outputs = self.transformer(
                    input_ids=host_input_ids, attention_mask=host_attention_mask
                )

                # Get host embeddings (use [CLS] token - first token)
                host_embeddings = host_transformer_outputs.last_hidden_state[:, 0, :]
                host_embeddings_list.append(host_embeddings.cpu().numpy())

                # Extract guest embeddings
                guest_encoded = self.tokenizer(
                    batch_guest_smiles,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )

                guest_input_ids = guest_encoded["input_ids"].to(self.device)
                guest_attention_mask = guest_encoded["attention_mask"].to(self.device)

                # Forward pass through transformer for guest SMILES
                guest_transformer_outputs = self.transformer(
                    input_ids=guest_input_ids, attention_mask=guest_attention_mask
                )

                # Get guest embeddings (use [CLS] token - first token)
                guest_embeddings = guest_transformer_outputs.last_hidden_state[:, 0, :]
                guest_embeddings_list.append(guest_embeddings.cpu().numpy())

        # Concatenate all embeddings
        all_host_embeddings = np.concatenate(host_embeddings_list, axis=0)
        all_guest_embeddings = np.concatenate(guest_embeddings_list, axis=0)
        logger.info(f"✅ Extracted host embeddings shape: {all_host_embeddings.shape}")
        logger.info(
            f"✅ Extracted guest embeddings shape: {all_guest_embeddings.shape}"
        )

        # Create DataFrame with embeddings
        host_embedding_columns = [
            f"host_embedding_{i}" for i in range(all_host_embeddings.shape[1])
        ]
        guest_embedding_columns = [
            f"guest_embedding_{i}" for i in range(all_guest_embeddings.shape[1])
        ]

        # Combine host and guest embeddings
        embeddings_df = pd.DataFrame(
            all_host_embeddings, columns=host_embedding_columns
        )
        guest_embeddings_df = pd.DataFrame(
            all_guest_embeddings, columns=guest_embedding_columns
        )
        embeddings_df = pd.concat([embeddings_df, guest_embeddings_df], axis=1)

        # Save embeddings (only embedding columns, no original data)
        logger.info(f"💾 Saving embeddings to: {output_path}")
        logger.info(f"� Final embeddings shape: {embeddings_df.shape}")
        embeddings_df.to_csv(output_path, index=False)

        return embeddings_df


class LOOCVTrainer:
    """Leave-One-Out Cross Validation Trainer for external validation datasets."""

    def __init__(
        self,
        cd_val_path: Path,
        pfas_val_path: Path,
        model_path: Path,
        random_seed: int = 42,
    ):
        self.cd_val_path = cd_val_path
        self.pfas_val_path = pfas_val_path
        self.model_path = model_path
        self.random_seed = random_seed
        self.best_params = None
        self.check_directories()
        self.data, self.labels = self.load_combined_data()
        np.random.seed(self.random_seed)

    def check_directories(self):
        """Create the model directory if it doesn't exist."""
        if not self.model_path.parent.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.model_path.parent}")

    def load_combined_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load and combine the cd_val and pfas_val datasets.

        Returns:
            tuple: Combined features and labels
        """
        logger.info(f"Loading CD validation data from: {self.cd_val_path}")
        cd_features = pd.read_csv(self.cd_val_path)

        logger.info(f"Loading PFAS validation data from: {self.pfas_val_path}")
        pfas_features = pd.read_csv(self.pfas_val_path)

        # Load labels from the corresponding directories
        cd_val_labels_path = self.cd_val_path.parent.parent / "cd_val.csv"
        pfas_val_labels_path = self.pfas_val_path.parent.parent / "pfas_val.csv"

        logger.info(f"Loading CD labels from: {cd_val_labels_path}")
        cd_labels_df = pd.read_csv(cd_val_labels_path)
        cd_labels = cd_labels_df["delG"]

        logger.info(f"Loading PFAS labels from: {pfas_val_labels_path}")
        pfas_labels_df = pd.read_csv(pfas_val_labels_path)
        pfas_labels = pfas_labels_df["delG"]

        # Combine features and labels
        combined_features = pd.concat(
            [cd_features, pfas_features], axis=0, ignore_index=True
        )
        combined_labels = pd.concat([cd_labels, pfas_labels], axis=0, ignore_index=True)

        logger.info(f"Combined dataset shape: {combined_features.shape}")
        logger.info(f"Combined labels shape: {combined_labels.shape}")

        return combined_features, combined_labels

    def run_loocv_lgbm(
        self, lgbm_params: dict = None, save_final_model: bool = False
    ) -> dict:
        """
        Run Leave-One-Out Cross Validation with LightGBM.

        Args:
            lgbm_params: LightGBM parameters to use. If None, uses default parameters.
            save_final_model: Whether to train and save a final model on the full dataset after LOOCV.

        Returns:
            dict: Dictionary containing LOOCV results including predictions and metrics
        """
        if lgbm_params is None:
            # Default LightGBM parameters optimized for small datasets
            lgbm_params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.1,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "random_state": self.random_seed,
                "n_estimators": 100,
            }

        n_samples = len(self.data)
        predictions = np.zeros(n_samples)
        actual_values = np.zeros(n_samples)

        logger.info(f"Starting LOOCV with {n_samples} samples...")

        for i in tqdm(range(n_samples), desc="LOOCV Progress"):
            # Create train and test splits for LOOCV
            train_indices = np.arange(n_samples)
            train_indices = train_indices[train_indices != i]
            test_index = i

            X_train = self.data.iloc[train_indices]
            y_train = self.labels.iloc[train_indices]
            X_test = self.data.iloc[[test_index]]
            y_test = self.labels.iloc[test_index]

            # Create LightGBM datasets
            train_dataset = lgb.Dataset(X_train, label=y_train)

            # Train model
            model = lgb.train(
                lgbm_params,
                train_dataset,
                valid_sets=[train_dataset],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
            )

            # Make prediction
            prediction = model.predict(X_test)[0]
            predictions[i] = prediction
            actual_values[i] = y_test

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)

        results = {
            "predictions": predictions,
            "actual_values": actual_values,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": n_samples,
            "model_params": lgbm_params,
        }

        logger.info("LOOCV Results:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MSE: {mse:.4f}")

        # Train final model on full dataset if requested
        if save_final_model:
            logger.info("Training final model on full dataset with best parameters...")
            full_dataset = lgb.Dataset(self.data, label=self.labels)

            final_model = lgb.train(
                lgbm_params,
                full_dataset,
                valid_sets=[full_dataset],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
            )

            # Save the final model
            joblib.dump(final_model, self.model_path)
            logger.info(f"✅ Final model saved to: {self.model_path}")

        return results

    def hyperparameter_tuning(
        self,
        sweep_config: dict,
        representation: str,
        sweep_count: int = 50,
    ) -> dict:
        """
        Run hyperparameter tuning using wandb sweeps for LOOCV LightGBM models.

        After the sweep completes, this method retrieves the best parameters from
        wandb based on the run with the lowest validation RMSE. Callers should
        then save these parameters using save_best_params() if needed.

        Args:
            sweep_config: Wandb sweep configuration
            representation: Data representation name (for project naming)
            sweep_count: Number of sweep iterations to run

        Returns:
            dict: Best hyperparameters found from wandb sweep
        """
        project_name = f"LOOCV-LightGBM-{representation}-external-validation"

        logger.info(
            f"Starting hyperparameter sweep with {sweep_count} trials for {representation}"
        )
        logger.info(f"Project: {project_name}")

        def objective():
            """Objective function for wandb sweep."""
            wandb.init()
            config = wandb.config

            # Convert wandb config to regular dict for LightGBM
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "verbose": -1,
                "seed": self.random_seed,
                "num_leaves": config.num_leaves,
                "max_depth": config.max_depth,
                "learning_rate": config.learning_rate,
                "n_estimators": config.n_estimators,
                "reg_alpha": config.reg_alpha,
                "reg_lambda": config.reg_lambda,
                "bagging_fraction": config.bagging_fraction,
                "feature_fraction": config.feature_fraction,
            }

            # Run LOOCV with these parameters
            results = self.run_loocv_lgbm(lgbm_params=params)

            # Log metrics to wandb
            wandb.log(
                {
                    "val_rmse": results["rmse"],
                    "val_mae": results["mae"],
                    "val_r2": results["r2"],
                    "val_mse": results["mse"],
                    "n_samples": results["n_samples"],
                }
            )

            # Save parameters for later use
            wandb.log(params)

        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        logger.info(f"Created sweep with ID: {sweep_id}")
        logger.info(
            f"Sweep URL: https://wandb.ai/{wandb.Api().default_entity}/{project_name}/sweeps/{sweep_id}"
        )

        # Run sweep
        wandb.agent(sweep_id, function=objective, count=sweep_count)

        # Retrieve best parameters from wandb after sweep completes
        logger.info("Retrieving best parameters from wandb...")
        api = wandb.Api()
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        runs = sweep.runs
        best_run = min(runs, key=lambda run: run.summary.get("val_rmse", float("inf")))

        # Extract best hyperparameters from the best run
        self.best_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": self.random_seed,
            "num_leaves": best_run.config["num_leaves"],
            "max_depth": best_run.config["max_depth"],
            "learning_rate": best_run.config["learning_rate"],
            "n_estimators": best_run.config["n_estimators"],
            "reg_alpha": best_run.config["reg_alpha"],
            "reg_lambda": best_run.config["reg_lambda"],
            "bagging_fraction": best_run.config["bagging_fraction"],
            "feature_fraction": best_run.config["feature_fraction"],
        }

        logger.info("Best parameters found:")
        for key, value in self.best_params.items():
            if key not in ["objective", "metric", "boosting_type", "verbose", "seed"]:
                logger.info(f"  {key}: {value}")

        logger.info(f"Best RMSE: {best_run.summary.get('val_rmse', 'N/A'):.4f}")
        logger.info(f"Best R²: {best_run.summary.get('val_r2', 'N/A'):.4f}")

        # Automatically save best parameters after retrieving from wandb
        # Note: suffix should be set via save_best_params() call after this method
        # if custom suffix is needed, but we store them in self.best_params

        return self.best_params

    def save_best_params(self, suffix: str = ""):
        """
        Save the best hyperparameters to a JSON file.

        Args:
            suffix: Optional suffix for the filename
        """
        if self.best_params is None:
            logger.warning(
                "No best parameters to save. Run hyperparameter tuning first."
            )
            return

        params_path = self.model_path.parent / f"loocv_best_params{suffix}.json"
        with open(params_path, "w") as f:
            json.dump(self.best_params, f, indent=4)
        logger.info(f"Saved best parameters to: {params_path}")

    def load_best_params(self, suffix: str = "") -> dict:
        """
        Load the best hyperparameters from a JSON file.

        Args:
            suffix: Optional suffix for the filename

        Returns:
            dict: Loaded hyperparameters
        """
        params_path = self.model_path.parent / f"loocv_best_params{suffix}.json"
        if not params_path.exists():
            logger.error(f"Best parameters file not found: {params_path}")
            return None

        with open(params_path) as f:
            self.best_params = json.load(f)
        logger.info(f"Loaded best parameters from: {params_path}")
        return self.best_params

    def train_final_model_with_best_params(
        self, suffix: str = "", save_model: bool = True, save_results: bool = True
    ) -> dict:
        """
        Train final LOOCV model using the best hyperparameters.

        Args:
            suffix: Optional suffix for result filenames
            save_model: Whether to save the final model trained on full dataset (default: True)
            save_results: Whether to save CSV and JSON result files (default: True)

        Returns:
            dict: Training results
        """
        if self.best_params is None:
            logger.error(
                "No best parameters available. Run hyperparameter tuning first or load existing parameters."
            )
            raise ValueError("Best parameters not available")

        logger.info("Training final LOOCV model with best hyperparameters...")

        # Run LOOCV with best parameters, save final model on full dataset
        results = self.run_loocv_lgbm(
            lgbm_params=self.best_params, save_final_model=save_model
        )

        # Save results with best parameters if requested
        if save_results:
            self.save_results(results, suffix=suffix)
        else:
            logger.info("Skipping results file generation (save_results=False)")

        logger.info("✅ Final LOOCV model completed with best parameters")
        logger.info(f"📊 Final RMSE: {results['rmse']:.4f}")
        logger.info(f"📈 Final R²: {results['r2']:.4f}")

        return results

    def save_results(self, results: dict, suffix: str = ""):
        """
        Save LOOCV results to files.

        Args:
            results: Results dictionary from run_loocv_lgbm
            suffix: Optional suffix for result filenames
        """
        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "actual": results["actual_values"],
                "predicted": results["predictions"],
                "residual": results["actual_values"] - results["predictions"],
            }
        )

        # Save detailed results
        results_path = self.model_path.parent / f"loocv_results{suffix}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved detailed results to: {results_path}")

        # Save summary metrics
        metrics = {
            "rmse": results["rmse"],
            "mae": results["mae"],
            "r2": results["r2"],
            "mse": results["mse"],
            "n_samples": results["n_samples"],
        }

        metrics_path = self.model_path.parent / f"loocv_metrics{suffix}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_path}")

        # Save model parameters
        params_path = self.model_path.parent / f"loocv_params{suffix}.json"
        with open(params_path, "w") as f:
            json.dump(results["model_params"], f, indent=2)
        logger.info(f"Saved model parameters to: {params_path}")


@app.command("train")
def train(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        help="Data representation (ecfp, ecfp_plus, original_enriched, unimol, grover, string)"
    ),
    model_type: str = typer.Option(help="Model type to train (lgbm, fnn, end2end)"),
    num_pcs: int = typer.Option(
        None,
        help="Number of PCs of the data to use, these must be generated prior to training",
    ),
) -> None:
    """
    Train a model with specified data source, representation, and model type.

    This function replaces individual training functions by dynamically setting paths
    and selecting the appropriate trainer based on the provided arguments.
    """
    logger.info(
        f"Training {model_type} model with {representation} representation from {data_source}"
    )

    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"

    # Set model path based on model type
    if num_pcs is None:
        model_dir = MODELS_DIR / data_source / representation
        model_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_dir / f"{model_type}_{representation}"
    else:
        model_dir = MODELS_DIR / data_source / representation / f"PCA_{num_pcs}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_dir / f"{model_type}_{representation}_PC{num_pcs}"

    if model_type == "lgbm":
        model_filename = f"{model_name}.pkl"
        model_path = model_dir / model_filename
        logger.info("Loading data...")
        trainer = LightGBMTrainer(features_path, labels_path, model_path)
        if num_pcs is None:
            logger.info("Splitting data...")
            (
                train_features,
                val_features,
                test_features,
                train_labels,
                val_labels,
                test_labels,
            ) = trainer.split_data()
        else:
            train_features_path = (
                PROCESSED_DATA_DIR
                / data_source
                / representation
                / f"PCA_{num_pcs}"
                / "train_features.csv"
            )
            val_features_path = (
                PROCESSED_DATA_DIR
                / data_source
                / representation
                / f"PCA_{num_pcs}"
                / "val_features.csv"
            )
            train_labels_path = (
                PROCESSED_DATA_DIR / data_source / representation / "train_labels.csv"
            )
            val_labels_path = (
                PROCESSED_DATA_DIR / data_source / representation / "val_labels.csv"
            )
            logger.info("Loading PCA-transformed data...")
            train_features = pd.read_csv(train_features_path)
            val_features = pd.read_csv(val_features_path)
            train_labels = pd.read_csv(train_labels_path)
            val_labels = pd.read_csv(val_labels_path)
            representation = representation + f"_PC{num_pcs}"
        sweep_config = lightgbm_sweep_config
        logger.info(
            f"Running wandb sweep for {model_type} model with {representation} data..."
        )
        trainer.hyperparameter_tuning(
            sweep_config,
            train_features,
            train_labels,
            val_features,
            val_labels,
            f"{representation.lower()}",
        )
        logger.info(f"Training final {model_type} model...")
        final_model = trainer.train_final_model(train_features, train_labels)
        logger.info(f"Saving {model_type} {representation} model...")
        joblib.dump(final_model, model_path)
        logger.success(
            f"{model_type.upper()} {representation} model training complete."
        )

    elif model_type == "fnn":
        model_filename = f"{model_name}.pth"
        model_path = model_dir / model_filename
        logger.info("Loading data...")
        trainer = FNNTrainer(features_path, labels_path, model_path)
        if num_pcs is None:
            logger.info("Splitting data...")
            (
                train_features,
                val_features,
                test_features,
                train_labels,
                val_labels,
                test_labels,
            ) = trainer.split_data()
        else:
            train_features_path = (
                PROCESSED_DATA_DIR
                / data_source
                / representation
                / f"PCA_{num_pcs}"
                / "train_features.csv"
            )
            val_features_path = (
                PROCESSED_DATA_DIR
                / data_source
                / representation
                / f"PCA_{num_pcs}"
                / "val_features.csv"
            )
            train_labels_path = (
                PROCESSED_DATA_DIR / data_source / representation / "train_labels.csv"
            )
            val_labels_path = (
                PROCESSED_DATA_DIR / data_source / representation / "val_labels.csv"
            )
            logger.info("Loading PCA-transformed data...")
            train_features = pd.read_csv(train_features_path)
            val_features = pd.read_csv(val_features_path)
            train_labels = pd.read_csv(train_labels_path)
            val_labels = pd.read_csv(val_labels_path)
            representation = representation + f"_PC{num_pcs}"
        logger.info("Scaling data (features)...")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        scaler_path = model_path.parent / f"{model_type}_{representation}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Feature scaler saved to {scaler_path}")
        sweep_config = fnn_sweep_config
        logger.info(
            f"Running wandb sweep for {model_type} model with {representation} data..."
        )
        trainer.hyperparameter_tuning(
            sweep_config,
            train_features,
            train_labels,
            val_features,
            val_labels,
            f"{representation.lower()}",
        )
        logger.info(f"Training final {model_type} model...")
        final_model = trainer.train_final_model(train_features, train_labels)
        logger.info(f"Saving {model_type} {representation} model...")
        torch.save(final_model.state_dict(), model_path)
        logger.success(
            f"{model_type.upper()} {representation} model training complete."
        )

    elif model_type == "end2end":
        model_filename = f"{model_name}.pth"
        model_path = model_dir / model_filename
        trainer = EndToEndTrainer(features_path, labels_path, model_path, verbose=True)

        logger.info("Splitting data...")
        (
            train_features,
            val_features,
            test_features,
            train_labels,
            val_labels,
            test_labels,
        ) = trainer.split_data()

        train_host_smiles = train_features["Host_SMILES"].tolist()
        train_guest_smiles = train_features["Guest_SMILES"].tolist()
        val_host_smiles = val_features["Host_SMILES"].tolist()
        val_guest_smiles = val_features["Guest_SMILES"].tolist()

        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        sweep_config = end2end_sweep_config
        logger.info(
            f"Running wandb sweep for {model_type} model with {representation} data..."
        )
        trainer.hyperparameter_tuning(
            sweep_config,
            train_host_smiles,
            train_guest_smiles,
            train_labels,
            val_host_smiles,
            val_guest_smiles,
            val_labels,
            f"{representation.lower()}",
        )
        logger.info(f"Training final {model_type} model...")
        # Train final model with best hyperparameters
        final_model = trainer.train_final_model(
            train_host_smiles,
            train_guest_smiles,
            train_labels,
            val_host_smiles,
            val_guest_smiles,
            val_labels,
        )
        logger.info(f"Saving {model_type} {representation} model...")
        torch.save(trainer.model.state_dict(), model_path)
        logger.success(
            f"{model_type.upper()} {representation} model training complete."
        )
    else:
        msg = f"Unsupported model type: {model_type}"
        raise ValueError(msg)


@app.command("train-final-model")
def train_final_model(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        help="Data representation (ecfp, ecfp_plus, original_enriched, unimol, grover, string)"
    ),
    model_type: str = typer.Option(help="Model type to train (lgbm, fnn, end2end)"),
    num_pcs: int = typer.Option(
        None,
        help="Number of PCs of the data to use, these must be generated prior to training",
    ),
) -> None:
    """
    Train a final model using the best hyperparameters from config files, without wandb sweeps.
    """
    logger.info(
        f"Training final {model_type} model with {representation} representation from {data_source} using best config"
    )

    # Set up paths
    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"

    if num_pcs is None:
        model_dir = MODELS_DIR / data_source / representation
        model_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_dir / f"{model_type}_{representation}"
    else:
        model_dir = MODELS_DIR / data_source / representation / f"PCA_{num_pcs}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_name = model_dir / f"{model_type}_{representation}_PC{num_pcs}"

    if model_type == "lgbm":
        model_filename = f"{model_name}.pkl"
        config_filename = f"{model_name}_config.json"
    elif model_type == "fnn" or model_type == "end2end":
        model_filename = f"{model_name}.pth"
        config_filename = f"{model_name}_config.json"
    else:
        msg = f"Unsupported model type: {model_type}"
        raise ValueError(msg)

    model_path = model_dir / model_filename
    config_path = model_dir / config_filename

    # Load best hyperparameters
    if not config_path.exists():
        msg = f"Config file not found at {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Load best hyperparameters from JSON file
    with open(config_path, encoding="utf-8") as f:
        best_params = json.load(f)
    logger.info(f"Loaded best hyperparameters from {config_path}")

    # Load data
    if num_pcs is None:
        # Split data
        if model_type == "lgbm":
            trainer = LightGBMTrainer(features_path, labels_path, model_path)
        elif model_type == "fnn":
            trainer = FNNTrainer(features_path, labels_path, model_path)
        elif model_type == "end2end":
            trainer = EndToEndTrainer(
                features_path, labels_path, model_path, verbose=True
            )
        (
            train_features,
            val_features,
            test_features,
            train_labels,
            val_labels,
            test_labels,
        ) = trainer.split_data()
    else:
        # Load PCA-transformed data
        train_features_path = (
            PROCESSED_DATA_DIR
            / data_source
            / representation
            / f"PCA_{num_pcs}"
            / "train_features.csv"
        )
        val_features_path = (
            PROCESSED_DATA_DIR
            / data_source
            / representation
            / f"PCA_{num_pcs}"
            / "val_features.csv"
        )
        train_labels_path = (
            PROCESSED_DATA_DIR / data_source / representation / "train_labels.csv"
        )
        val_labels_path = (
            PROCESSED_DATA_DIR / data_source / representation / "val_labels.csv"
        )

        train_features = pd.read_csv(train_features_path)
        val_features = pd.read_csv(val_features_path)
        train_labels = pd.read_csv(train_labels_path)
        val_labels = pd.read_csv(val_labels_path)

        if model_type == "lgbm":
            trainer = LightGBMTrainer(features_path, labels_path, model_path)
        elif model_type == "fnn":
            trainer = FNNTrainer(features_path, labels_path, model_path)
        elif model_type == "end2end":
            trainer = EndToEndTrainer(
                features_path, labels_path, model_path, verbose=True
            )

    # Apply scaling for FNN models
    if model_type == "fnn":
        scaler_path = model_path.parent / f"{model_type}_{representation}_scaler.pkl"
        if not scaler_path.exists():
            msg = f"Scaler file not found at {scaler_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        scaler = joblib.load(scaler_path)
        train_features = scaler.transform(train_features)
        val_features = scaler.transform(val_features)
        train_labels_raw = np.array(train_labels)
        val_labels_raw = np.array(val_labels)
    elif model_type == "end2end":
        # For end2end models, we need SMILES strings
        train_host_smiles = train_features["Host_SMILES"].tolist()
        train_guest_smiles = train_features["Guest_SMILES"].tolist()
        val_host_smiles = val_features["Host_SMILES"].tolist()
        val_guest_smiles = val_features["Guest_SMILES"].tolist()
        train_labels_raw = np.array(train_labels)
        val_labels_raw = np.array(val_labels)
    # Set best_params in trainer
    trainer.best_params = best_params

    logger.info(f"Training final {model_type} model with loaded hyperparameters...")
    if model_type == "lgbm":
        final_model = trainer.train_final_model(
            pd.concat([train_features, val_features], ignore_index=True),
            pd.concat([train_labels, val_labels], ignore_index=True),
        )
        joblib.dump(final_model, model_path)
        logger.success(f"Final LightGBM model saved to {model_path}")
    elif model_type == "fnn":
        final_model = trainer.train_final_model(
            np.vstack([train_features, val_features]),
            np.concatenate([train_labels_raw, val_labels_raw]),
        )
        torch.save(final_model.state_dict(), model_path)
        logger.success(f"Final FNN model saved to {model_path}")
    elif model_type == "end2end":
        # Combine training and validation data for final model
        all_host_smiles = train_host_smiles + val_host_smiles
        all_guest_smiles = train_guest_smiles + val_guest_smiles
        all_labels = np.concatenate([train_labels_raw, val_labels_raw])

        final_model = trainer.train_final_model(
            all_host_smiles,
            all_guest_smiles,
            all_labels,
        )
        torch.save(trainer.model.state_dict(), model_path)
        logger.success(f"Final End-to-End model saved to {model_path}")


@app.command("fine-tune-end2end")
def fine_tune_end2end(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        "string", help="Data representation (should be 'string' for end2end models)"
    ),
    pretrained_model_name: str = typer.Option(
        help="Name of the pre-trained model file (without .pth extension)"
    ),
    save_suffix: str = typer.Option(
        "finetuned", help="Suffix to add to the fine-tuned model filename"
    ),
    wandb_log: bool = typer.Option(
        True, help="Whether to log fine-tuning metrics to wandb"
    ),
) -> None:
    """
    Fine-tune a pre-trained end2end model using hyperparameters from the config file.
    The fine-tuned model will be saved with a suffix to avoid overwriting the original.
    """
    logger.info(f"Starting fine-tuning of {pretrained_model_name}")

    # Set up paths
    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"

    model_dir = MODELS_DIR / data_source / representation
    pretrained_model_path = model_dir / f"{pretrained_model_name}.pth"
    config_path = model_dir / f"{pretrained_model_name}_config.json"

    # Load fine-tuning hyperparameters from config
    finetune_config = end2end_finetune_config
    logger.info("Using fine-tuning configuration from config.py")
    logger.info(f"Fine-tuning config: {finetune_config}")

    # Extract fine-tuning parameters from config
    unfreeze_layers = finetune_config.get("unfreeze_layers", ["encoder.layer.11"])
    transformer_learning_rate = finetune_config.get("transformer_learning_rate", 1e-6)
    fnn_learning_rate = finetune_config.get("fnn_learning_rate", None)
    epochs = finetune_config.get("epochs", None)
    batch_size = finetune_config.get("batch_size", 128)

    logger.info(f"Unfrozen layers: {unfreeze_layers}")
    logger.info(f"Transformer LR: {transformer_learning_rate}")
    logger.info(f"FNN LR: {fnn_learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")

    # Create trainer instance
    trainer = EndToEndTrainer(
        features_path, labels_path, pretrained_model_path, verbose=False
    )

    # Load the pre-trained model
    logger.info(f"Loading pre-trained model from {pretrained_model_path}")
    trainer.load_pretrained_model(pretrained_model_path, config_path)

    # Load data from predefined split files to prevent data leakage
    logger.info("Loading data from predefined split files...")
    processed_data_dir = PROCESSED_DATA_DIR / data_source / representation

    # Load training data
    train_features = pd.read_csv(processed_data_dir / "train_features.csv")
    train_labels = pd.read_csv(processed_data_dir / "train_labels.csv")

    # Load validation data
    val_features = pd.read_csv(processed_data_dir / "val_features.csv")
    val_labels = pd.read_csv(processed_data_dir / "val_labels.csv")

    logger.info("Loaded predefined splits:")
    logger.info(f"  Training samples: {len(train_features)}")
    logger.info(f"  Validation samples: {len(val_features)}")

    # Extract SMILES strings
    train_host_smiles = train_features["Host_SMILES"].tolist()
    train_guest_smiles = train_features["Guest_SMILES"].tolist()
    val_host_smiles = val_features["Host_SMILES"].tolist()
    val_guest_smiles = val_features["Guest_SMILES"].tolist()

    # Extract labels (assuming single column or first column contains the target)
    if len(train_labels.columns) == 1:
        train_labels = np.array(train_labels.iloc[:, 0])
        val_labels = np.array(val_labels.iloc[:, 0])
    else:
        # If multiple columns, assume the target is in a column named 'target' or first column
        target_col = (
            "target" if "target" in train_labels.columns else train_labels.columns[0]
        )
        train_labels = np.array(train_labels[target_col])
        val_labels = np.array(val_labels[target_col])

    # Fine-tune the model
    logger.info("Starting fine-tuning...")
    trainer.fine_tune_model(
        train_host_smiles=train_host_smiles,
        train_guest_smiles=train_guest_smiles,
        train_labels=train_labels,
        val_host_smiles=val_host_smiles,
        val_guest_smiles=val_guest_smiles,
        val_labels=val_labels,
        unfreeze_layers=unfreeze_layers,
        transformer_learning_rate=transformer_learning_rate,
        fnn_learning_rate=fnn_learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        save_suffix=save_suffix,
        wandb_log=wandb_log,
    )

    logger.success("Fine-tuning completed successfully!")


@app.command("sweep-finetune-end2end")
def sweep_finetune_end2end(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        "string", help="Data representation (should be 'string' for end2end models)"
    ),
    pretrained_model_name: str = typer.Option(
        help="Name of the pre-trained model file (without .pth extension)"
    ),
) -> None:
    """
    Run a hyperparameter sweep for fine-tuning a pre-trained end2end model.
    This will optimize unfreeze strategies, learning rates, and other hyperparameters.
    """
    logger.info(
        f"Starting fine-tuning hyperparameter sweep for {pretrained_model_name}"
    )

    # Set up paths
    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"

    model_dir = MODELS_DIR / data_source / representation
    pretrained_model_path = model_dir / f"{pretrained_model_name}.pth"
    config_path = model_dir / f"{pretrained_model_name}_config.json"

    # Validate paths
    if not pretrained_model_path.exists():
        logger.error(f"Pre-trained model not found: {pretrained_model_path}")
        raise typer.Exit(1)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(1)

    logger.info("Using fine-tuning sweep configuration from config.py")
    logger.info(f"Fine-tuning sweep config: {end2end_finetune_sweep_config}")

    # Create trainer instance
    trainer = EndToEndTrainer(
        features_path, labels_path, pretrained_model_path, verbose=False
    )

    # Load data from predefined split files to prevent data leakage
    logger.info("Loading data from predefined split files...")
    processed_data_dir = PROCESSED_DATA_DIR / data_source / representation

    # Load training data
    train_features = pd.read_csv(processed_data_dir / "train_features.csv")
    train_labels = pd.read_csv(processed_data_dir / "train_labels.csv")

    # Load validation data
    val_features = pd.read_csv(processed_data_dir / "val_features.csv")
    val_labels = pd.read_csv(processed_data_dir / "val_labels.csv")

    logger.info("Loaded predefined splits:")
    logger.info(f"  Training samples: {len(train_features)}")
    logger.info(f"  Validation samples: {len(val_features)}")

    # Extract SMILES strings
    train_host_smiles = train_features["Host_SMILES"].tolist()
    train_guest_smiles = train_features["Guest_SMILES"].tolist()
    val_host_smiles = val_features["Host_SMILES"].tolist()
    val_guest_smiles = val_features["Guest_SMILES"].tolist()

    # Extract labels (assuming single column or first column contains the target)
    if len(train_labels.columns) == 1:
        train_labels = np.array(train_labels.iloc[:, 0])
        val_labels = np.array(val_labels.iloc[:, 0])
    else:
        # If multiple columns, assume the target is in a column named 'target' or first column
        target_col = (
            "target" if "target" in train_labels.columns else train_labels.columns[0]
        )
        train_labels = np.array(train_labels[target_col])
        val_labels = np.array(val_labels[target_col])

    # Run hyperparameter sweep for fine-tuning
    logger.info("Starting fine-tuning hyperparameter sweep...")
    trainer.finetune_hyperparameter_tuning(
        train_host_smiles=train_host_smiles,
        train_guest_smiles=train_guest_smiles,
        train_labels=train_labels,
        val_host_smiles=val_host_smiles,
        val_guest_smiles=val_guest_smiles,
        val_labels=val_labels,
        sweep_config=end2end_finetune_sweep_config,
        data_type=representation,
        pretrained_model_path=pretrained_model_path,
        config_path=config_path,
    )

    logger.success("Fine-tuning hyperparameter sweep completed successfully!")
    logger.info(f"Best parameters: {trainer.best_finetune_params}")


@app.command("finetune-with-best-params")
def finetune_with_best_params(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        "string", help="Data representation (should be 'string' for end2end models)"
    ),
    pretrained_model_name: str = typer.Option(
        help="Name of the pre-trained model file (without .pth extension)"
    ),
    sweep_run_id: str = typer.Option(
        help="Wandb run ID from the sweep to use for best parameters"
    ),
    save_suffix: str = typer.Option(
        "finetuned_best", help="Suffix to add to the fine-tuned model filename"
    ),
    wandb_log: bool = typer.Option(
        True, help="Whether to log fine-tuning metrics to wandb"
    ),
) -> None:
    """
    Fine-tune a model using the best hyperparameters from a completed sweep.
    """
    logger.info(
        f"Fine-tuning {pretrained_model_name} with best parameters from sweep run {sweep_run_id}"
    )

    # Set up paths
    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"

    model_dir = MODELS_DIR / data_source / representation
    pretrained_model_path = model_dir / f"{pretrained_model_name}.pth"
    config_path = model_dir / f"{pretrained_model_name}_config.json"

    # Load best parameters from wandb
    api = wandb.Api()
    run = api.run(sweep_run_id)
    best_params = run.config
    logger.info(f"Loaded best parameters: {best_params}")

    # Create trainer instance
    trainer = EndToEndTrainer(
        features_path, labels_path, pretrained_model_path, verbose=False
    )

    # Load the pre-trained model
    logger.info(f"Loading pre-trained model from {pretrained_model_path}")
    trainer.load_pretrained_model(pretrained_model_path, config_path)

    # Load data from predefined split files
    logger.info("Loading data from predefined split files...")
    processed_data_dir = PROCESSED_DATA_DIR / data_source / representation

    train_features = pd.read_csv(processed_data_dir / "train_features.csv")
    train_labels = pd.read_csv(processed_data_dir / "train_labels.csv")
    val_features = pd.read_csv(processed_data_dir / "val_features.csv")
    val_labels = pd.read_csv(processed_data_dir / "val_labels.csv")

    # Extract SMILES strings and labels
    train_host_smiles = train_features["Host_SMILES"].tolist()
    train_guest_smiles = train_features["Guest_SMILES"].tolist()
    val_host_smiles = val_features["Host_SMILES"].tolist()
    val_guest_smiles = val_features["Guest_SMILES"].tolist()

    if len(train_labels.columns) == 1:
        train_labels = np.array(train_labels.iloc[:, 0])
        val_labels = np.array(val_labels.iloc[:, 0])
    else:
        target_col = (
            "target" if "target" in train_labels.columns else train_labels.columns[0]
        )
        train_labels = np.array(train_labels[target_col])
        val_labels = np.array(val_labels[target_col])

    # Extract best parameters
    unfreeze_layers = best_params.get("unfreeze_layers", [])

    transformer_learning_rate = best_params.get("transformer_learning_rate", 1e-6)
    fnn_learning_rate = best_params.get("fnn_learning_rate", 1e-5)
    epochs = best_params.get("epochs", 100)
    batch_size = best_params.get("batch_size", 64)

    logger.info("Using unfreeze_layers directly")
    logger.info(f"Unfreezing {len(unfreeze_layers)} layers")
    logger.info(
        f"Transformer LR: {transformer_learning_rate}, FNN LR: {fnn_learning_rate}"
    )
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

    # Fine-tune the model with best parameters
    trainer.fine_tune_model(
        train_host_smiles=train_host_smiles,
        train_guest_smiles=train_guest_smiles,
        train_labels=train_labels,
        val_host_smiles=val_host_smiles,
        val_guest_smiles=val_guest_smiles,
        val_labels=val_labels,
        unfreeze_layers=unfreeze_layers,
        transformer_learning_rate=transformer_learning_rate,
        fnn_learning_rate=fnn_learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        save_suffix=save_suffix,
        wandb_log=wandb_log,
    )

    logger.success("Fine-tuning with best parameters completed successfully!")


@app.command("train-final-finetuned")
def train_final_finetuned(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        "string", help="Data representation (should be 'string' for end2end models)"
    ),
    pretrained_model_name: str = typer.Option(
        help="Name of the pre-trained model file (without .pth extension)"
    ),
    source_type: str = typer.Option(
        "project",
        help="Source for best params: 'project' (search entire project), 'sweep' (specific sweep ID), or 'run' (specific run ID)",
    ),
    source_id: str = typer.Option(
        "auto",
        help="Source identifier: project name (for 'project'), sweep ID (for 'sweep'), or run ID (for 'run'). Use 'auto' for default project naming.",
    ),
    project_name: str = typer.Option(
        None,
        help="Project name (required when using sweep ID without full path, or to override default project naming)",
    ),
    save_suffix: str = typer.Option(
        "finetuned_final", help="Suffix to add to the final fine-tuned model filename"
    ),
    wandb_log: bool = typer.Option(
        True, help="Whether to log fine-tuning metrics to wandb"
    ),
) -> None:
    """
    Train a final fine-tuned model using the best hyperparameters from various sources.

    Examples:
        # Auto-find best from default project
        train-final-finetuned --pretrained-model-name model --source-type project

        # Best from specific project
        train-final-finetuned --pretrained-model-name model --source-type project --source-id my_project

        # Best from specific sweep
        train-final-finetuned --pretrained-model-name model --source-type sweep --source-id abc123

        # Best from specific run
        train-final-finetuned --pretrained-model-name model --source-type run --source-id run_abc123
    """
    logger.info(
        f"Training final fine-tuned model using best parameters from {source_type}: {source_id}"
    )

    # Set up paths
    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"

    model_dir = MODELS_DIR / data_source / representation
    pretrained_model_path = model_dir / f"{pretrained_model_name}.pth"
    config_path = model_dir / f"{pretrained_model_name}_config.json"

    # Create trainer instance
    trainer = EndToEndTrainer(
        features_path, labels_path, pretrained_model_path, verbose=False
    )

    # Get best parameters based on source type
    logger.info("Retrieving best hyperparameters from wandb...")

    if source_type == "project":
        project_id = source_id if source_id != "auto" else "auto"
        best_params = trainer.get_best_finetune_params_from_project(
            project_name=project_id, data_type=representation
        )
        logger.info(f"Retrieved best parameters from project: {project_id}")

    elif source_type == "sweep":
        best_params = trainer.get_best_finetune_params_from_sweep(
            sweep_id=source_id, project_name=project_name
        )
        logger.info(f"Retrieved best parameters from sweep: {source_id}")

    elif source_type == "run":
        # Load best parameters from specific wandb run
        api = wandb.Api()
        run = api.run(source_id)
        best_params = run.config
        trainer.best_finetune_params = best_params
        logger.info(f"Retrieved best parameters from run: {source_id}")

    else:
        error_msg = (
            f"Invalid source_type: {source_type}. Must be 'project', 'sweep', or 'run'"
        )
        raise ValueError(error_msg)

    logger.info(f"Best parameters retrieved: {best_params}")

    # Load data from predefined split files
    logger.info("Loading data from predefined split files...")
    processed_data_dir = PROCESSED_DATA_DIR / data_source / representation

    train_features = pd.read_csv(processed_data_dir / "train_features.csv")
    train_labels = pd.read_csv(processed_data_dir / "train_labels.csv")
    val_features = pd.read_csv(processed_data_dir / "val_features.csv")
    val_labels = pd.read_csv(processed_data_dir / "val_labels.csv")

    # Extract SMILES strings and labels
    train_host_smiles = train_features["Host_SMILES"].tolist()
    train_guest_smiles = train_features["Guest_SMILES"].tolist()
    val_host_smiles = val_features["Host_SMILES"].tolist()
    val_guest_smiles = val_features["Guest_SMILES"].tolist()

    if len(train_labels.columns) == 1:
        train_labels = np.array(train_labels.iloc[:, 0])
        val_labels = np.array(val_labels.iloc[:, 0])
    else:
        target_col = (
            "target" if "target" in train_labels.columns else train_labels.columns[0]
        )
        train_labels = np.array(train_labels[target_col])
        val_labels = np.array(val_labels[target_col])

    # Train final model with best parameters
    logger.info("Training final fine-tuned model...")
    trainer.train_final_finetuned_model(
        train_host_smiles=train_host_smiles,
        train_guest_smiles=train_guest_smiles,
        train_labels=train_labels,
        val_host_smiles=val_host_smiles,
        val_guest_smiles=val_guest_smiles,
        val_labels=val_labels,
        pretrained_model_path=pretrained_model_path,
        config_path=config_path,
        save_suffix=save_suffix,
        wandb_log=wandb_log,
    )

    logger.success("Final fine-tuned model training completed successfully!")


@app.command("list-transformer-layers")
def list_transformer_layers(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    representation: str = typer.Option(
        "string", help="Data representation (should be 'string' for end2end models)"
    ),
) -> None:
    """
    List all available transformer layer names for fine-tuning configuration.
    This helps users identify which layers they want to unfreeze.
    """
    logger.info("Loading transformer to display layer names...")

    # Set up paths (we just need to create a trainer to access transformer layers)
    features_path = (
        INTERIM_DATA_DIR / data_source / representation / f"{representation}.csv"
    )
    labels_path = INTERIM_DATA_DIR / data_source / "labels" / "labels.csv"
    model_path = MODELS_DIR / data_source / representation / "dummy.pth"  # Dummy path

    trainer = EndToEndTrainer(features_path, labels_path, model_path, verbose=False)

    # Build a model to get access to transformer layers
    trainer.build_model()

    # Print all transformer layer names
    trainer.print_transformer_layers()


@app.command("train-all-final-models")
def train_all_final_models(
    data_source: str = typer.Option(
        "OpenCycloDB", help="Data source to use (OpenCycloDB)"
    ),
    model_types: list[str] = typer.Option(
        ["lgbm", "fnn", "end2end"],
        help="List of model types to train (lgbm, fnn, end2end)",
    ),
    representations: list[str] = typer.Option(
        [
            "ecfp",
            "ecfp_plus",
            "grover",
            "original_enriched",
            "unimol",
            "string",
        ],
        help="List of data representations to use (string is used for end2end models)",
    ),
    num_pcs: int = typer.Option(
        None,
        help="Number of PCs of the data to use, these must be generated prior to training",
    ),
) -> None:
    """
    Train all final models for all combinations of model types and representations.
    """
    for model_type in model_types:
        for representation in representations:
            try:
                logger.info(
                    f"Training final model for type: {model_type}, representation: {representation}"
                )
                train_final_model(
                    data_source=data_source,
                    representation=representation,
                    model_type=model_type,
                    num_pcs=num_pcs,
                )
            except Exception as e:
                logger.error(
                    f"Failed to train final model for type: {model_type}, representation: {representation}. Error: {e}"
                )


@app.command("extract-embeddings")
def extract_embeddings_cli(
    data_path: str = typer.Option(help="Path to CSV file containing SMILES data"),
    model_path: str = typer.Option(help="Path to trained end2end model (.pth file)"),
    labels_path: str = typer.Option(
        help="Path to labels CSV file (required for trainer initialization)"
    ),
    output_path: str = typer.Option(
        None, help="Path to save embeddings CSV (optional)"
    ),
) -> None:
    """
    Extract embeddings from end2end models for t-SNE analysis.

    This command loads a trained end2end model and extracts the transformer embeddings
    (before FNN processing) for each host-guest SMILES pair in the input data. For
    fine-tuned models, it uses the respective fine-tuned transformer heads for host
    and guest SMILES. The embeddings are saved as a CSV file for downstream analysis
    like t-SNE visualization.

    Note: This command requires the corresponding config file (e.g., model_config.json)
    to be present in the same directory as the model file to build the architecture correctly.

    Args:
        data_path: Path to input CSV file with 'Host_SMILES' and 'Guest_SMILES' columns
        model_path: Path to trained end2end model (.pth file)
        labels_path: Path to labels CSV file (required for trainer initialization)
        output_path: Optional path for output embeddings CSV (defaults to input directory)
    """
    logger.info("🔬 Starting embedding extraction for t-SNE analysis")

    # Convert string paths to Path objects
    data_path_obj = Path(data_path)
    model_path_obj = Path(model_path)
    labels_path_obj = Path(labels_path)
    output_path_obj = Path(output_path) if output_path else None

    # Validate input files exist
    if not data_path_obj.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        raise typer.Exit(1)

    if not model_path_obj.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        raise typer.Exit(1)

    if not labels_path_obj.exists():
        logger.error(f"❌ Labels file not found: {labels_path}")
        raise typer.Exit(1)

    # Determine model configuration from path
    # Assume model path contains info about the model type
    if "end2end_string" in str(model_path_obj):
        logger.info("🤖 Detected end2end_string model")

        # Look for the corresponding config file
        config_path = model_path_obj.parent / f"{model_path_obj.stem}_config.json"

        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            logger.info(
                "💡 Config file is required to build the model with correct architecture"
            )
            raise typer.Exit(1)

        logger.info(f"📋 Loading model config from: {config_path}")

        # Create trainer with the provided paths
        trainer = EndToEndTrainer(
            features_path=data_path_obj,
            labels_path=labels_path_obj,
            model_path=model_path_obj,
            verbose=False,
        )

        # Load config and build the model architecture to match the saved model
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        logger.info(f"🏗️ Building model with config: {config}")

        # Build the model with the loaded configuration but with fresh pretrained SMILES Tokenizer
        trainer.build_model(
            fnn_hidden_dim=config["hidden_dim"],
            fnn_output_dim=1,
            fnn_num_layers=config["num_layers"],
            fnn_dropout=config["dropout"],
            fnn_activation=config["activation"],
            unfreeze_layers=[],  # Use fresh pretrained weights, don't unfreeze any layers
            transformer_learning_rate=config.get("transformer_learning_rate", 1e-5),
        )

        # Load only the FNN and BatchNorm weights from the saved model
        logger.info(f"🔄 Loading trained FNN weights from: {model_path_obj}")
        checkpoint = torch.load(model_path_obj, map_location=trainer.device)

        # Load model state
        saved_state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Filter out transformer weights and only keep FNN and BatchNorm weights
        fnn_and_bn_state_dict = {}
        for key, value in saved_state_dict.items():
            if key.startswith(("fnn.", "batchnorm.")):
                fnn_and_bn_state_dict[key] = value
                logger.info(f"  Loading weight: {key}")

        # Load only the filtered weights (FNN and BatchNorm), keeping transformer fresh
        trainer.model.load_state_dict(fnn_and_bn_state_dict, strict=False)
        logger.info(
            f"✅ Loaded {len(fnn_and_bn_state_dict)} FNN/BatchNorm parameters, keeping transformer pretrained"
        )

    else:
        logger.error(f"❌ Could not determine model type from path: {model_path}")
        logger.info(
            "💡 Model path should contain 'end2end_string' to identify model type"
        )
        raise typer.Exit(1)

    try:
        # Set the correct output path with finetuned prefix if not explicitly provided
        if output_path_obj is None:
            output_path_obj = (
                data_path_obj.parent
                / f"{data_path_obj.stem}_{'finetuned_' if 'finetuned' in str(model_path_obj) else ''}embeddings.csv"
            )

        # Extract embeddings using the model with fresh transformer and trained FNN
        embeddings_df = trainer.extract_embeddings(
            data_path=data_path_obj,
            model_path=None,  # Model already loaded
            output_path=output_path_obj,
            skip_model_loading=True,  # Skip model loading since we already configured it
        )

        logger.info(f"✅ Successfully extracted {len(embeddings_df)} embeddings")
        logger.info(
            f"📊 Embedding dimensions: {embeddings_df.shape[1] - len([col for col in embeddings_df.columns if not col.startswith('embedding_')])}"
        )

        if output_path_obj:
            logger.info(f"💾 Embeddings saved to: {output_path_obj}")
        else:
            default_output = (
                data_path_obj.parent
                / f"{data_path_obj.stem}_{'finetuned_' if 'finetuned' in str(model_path_obj) else ''}embeddings.csv"
            )
            logger.info(f"💾 Embeddings saved to: {default_output}")

    except Exception as e:
        logger.error(f"❌ Failed to extract embeddings: {e}")
        raise typer.Exit(1) from e


@app.command("loocv")
def run_loocv(
    representation: str = typer.Option(
        help="Data representation (unimol, grover, ecfp, end2end, end2end_finetuned)"
    ),
    model_type: str = typer.Option(
        "lgbm", help="Model type to use (currently only lgbm supported)"
    ),
    custom_params: str = typer.Option(
        None, help="Path to JSON file with custom LightGBM parameters"
    ),
) -> None:
    """
    Run Leave-One-Out Cross Validation on combined cd_val and pfas_val external validation datasets.
    """
    if model_type != "lgbm":
        logger.error("Currently only LightGBM (lgbm) models are supported for LOOCV")
        raise typer.Exit(1)

    if representation not in [
        "unimol",
        "grover",
        "ecfp",
        "end2end",
        "end2end_finetuned",
    ]:
        logger.error(
            f"Unsupported representation: {representation}. Use: unimol, grover, ecfp, end2end, or end2end_finetuned"
        )
        raise typer.Exit(1)

    logger.info(
        f"Running LOOCV with {representation} representation using {model_type}"
    )

    # Set up paths for external validation data
    cd_val_features_path = (
        EXTERNAL_DATA_DIR
        / "validation"
        / "cd_val"
        / representation
        / f"cd_val_canonical_{representation}.csv"
    )

    pfas_val_features_path = (
        EXTERNAL_DATA_DIR
        / "validation"
        / "pfas_val"
        / representation
        / f"pfas_val_canonical_{representation}.csv"
    )

    # Check if files exist
    if not cd_val_features_path.exists():
        logger.error(f"CD validation features file not found: {cd_val_features_path}")
        raise typer.Exit(1)

    if not pfas_val_features_path.exists():
        logger.error(
            f"PFAS validation features file not found: {pfas_val_features_path}"
        )
        raise typer.Exit(1)

    # Set up model path
    model_dir = MODELS_DIR / "external_validation" / representation
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"loocv_{model_type}_{representation}.pkl"

    # Load custom parameters if provided
    lgbm_params = None
    if custom_params:
        custom_params_path = Path(custom_params)
        if custom_params_path.exists():
            with open(custom_params_path) as f:
                lgbm_params = json.load(f)
            logger.info(f"Loaded custom parameters from: {custom_params_path}")
        else:
            logger.error(f"Custom parameters file not found: {custom_params_path}")
            raise typer.Exit(1)

    # Initialize LOOCV trainer
    trainer = LOOCVTrainer(
        cd_val_path=cd_val_features_path,
        pfas_val_path=pfas_val_features_path,
        model_path=model_path,
        random_seed=42,
    )

    # Run LOOCV
    try:
        results = trainer.run_loocv_lgbm(lgbm_params=lgbm_params)

        # Save results
        suffix = f"_{representation}"
        trainer.save_results(results, suffix=suffix)

        logger.info(
            f"✅ LOOCV completed successfully for {representation} representation"
        )
        logger.info(f"📊 Final RMSE: {results['rmse']:.4f}")
        logger.info(f"📈 Final R²: {results['r2']:.4f}")

    except Exception as e:
        logger.error(f"❌ LOOCV failed: {e}")
        raise typer.Exit(1) from e


@app.command("loocv-all")
def run_loocv_all(
    model_type: str = typer.Option(
        "lgbm", help="Model type to use (currently only lgbm supported)"
    ),
    custom_params: str = typer.Option(
        None, help="Path to JSON file with custom LightGBM parameters"
    ),
) -> None:
    """
    Run Leave-One-Out Cross Validation for all supported representations.
    """
    representations = ["unimol", "grover", "ecfp", "end2end", "end2end_finetuned"]

    logger.info(f"Running LOOCV for all representations: {representations}")

    results_summary = {}

    for representation in representations:
        logger.info(f"\n🔄 Processing representation: {representation}")

        try:
            # Set up paths for external validation data
            cd_val_features_path = (
                EXTERNAL_DATA_DIR
                / "validation"
                / "cd_val"
                / representation
                / f"cd_val_canonical_{representation}.csv"
            )

            pfas_val_features_path = (
                EXTERNAL_DATA_DIR
                / "validation"
                / "pfas_val"
                / representation
                / f"pfas_val_canonical_{representation}.csv"
            )

            # Check if files exist
            if not cd_val_features_path.exists():
                logger.error(
                    f"CD validation features file not found: {cd_val_features_path}"
                )
                logger.warning(f"Skipping {representation}")
                continue

            if not pfas_val_features_path.exists():
                logger.error(
                    f"PFAS validation features file not found: {pfas_val_features_path}"
                )
                logger.warning(f"Skipping {representation}")
                continue

            # Set up model path
            model_dir = MODELS_DIR / "external_validation" / representation
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"loocv_{model_type}_{representation}.pkl"

            # Load custom parameters if provided
            lgbm_params = None
            if custom_params:
                custom_params_path = Path(custom_params)
                if custom_params_path.exists():
                    with open(custom_params_path) as f:
                        lgbm_params = json.load(f)
                    logger.info(f"Using custom parameters from: {custom_params_path}")
                else:
                    logger.error(
                        f"Custom parameters file not found: {custom_params_path}"
                    )
                    # Continue with default parameters

            # Initialize LOOCV trainer
            trainer = LOOCVTrainer(
                cd_val_path=cd_val_features_path,
                pfas_val_path=pfas_val_features_path,
                model_path=model_path,
                random_seed=42,
            )

            # Run LOOCV
            results = trainer.run_loocv_lgbm(lgbm_params=lgbm_params)

            # Save results
            suffix = f"_{representation}"
            trainer.save_results(results, suffix=suffix)

            # Store summary
            results_summary[representation] = {
                "rmse": results["rmse"],
                "mae": results["mae"],
                "r2": results["r2"],
                "n_samples": results["n_samples"],
            }

            logger.info(
                f"✅ {representation} completed - RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to process {representation}: {e}")
            logger.warning("Continuing with next representation...")
            continue

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("🏆 LOOCV SUMMARY RESULTS")
    logger.info("=" * 60)

    if results_summary:
        # Create summary DataFrame for easy comparison
        summary_df = pd.DataFrame(results_summary).T
        summary_df = summary_df.round(4)

        logger.info(f"\n{summary_df.to_string()}")

        # Save summary
        summary_path = MODELS_DIR / "external_validation" / "loocv_summary.csv"
        summary_df.to_csv(summary_path)
        logger.info(f"\n📊 Summary saved to: {summary_path}")

        # Find best performing model
        best_representation = summary_df["rmse"].idxmin()
        best_rmse = summary_df.loc[best_representation, "rmse"]
        best_r2 = summary_df.loc[best_representation, "r2"]

        logger.info(f"\n🥇 Best performing representation: {best_representation}")
        logger.info(f"   RMSE: {best_rmse:.4f}")
        logger.info(f"   R²: {best_r2:.4f}")
    else:
        logger.error("❌ No representations were successfully processed")


@app.command("loocv-tune")
def run_loocv_hyperparameter_tuning(
    representation: str = typer.Option(
        help="Data representation (unimol, grover, ecfp, end2end, end2end_finetuned)"
    ),
    model_type: str = typer.Option(
        "lgbm", help="Model type to use (currently only lgbm supported)"
    ),
    sweep_count: int = typer.Option(
        50, help="Number of hyperparameter sweep iterations"
    ),
) -> None:
    """
    Run hyperparameter tuning for LOOCV models using wandb sweeps.
    """
    if model_type != "lgbm":
        logger.error("Currently only LightGBM (lgbm) models are supported for LOOCV")
        raise typer.Exit(1)

    if representation not in [
        "unimol",
        "grover",
        "ecfp",
        "end2end",
        "end2end_finetuned",
    ]:
        logger.error(
            f"Unsupported representation: {representation}. Use: unimol, grover, ecfp, end2end, or end2end_finetuned"
        )
        raise typer.Exit(1)

    logger.info(
        f"Starting hyperparameter tuning for {representation} LOOCV with {sweep_count} iterations"
    )

    # Set up paths for external validation data
    if representation == "end2end_finetuned":
        # Special handling for finetuned embeddings
        cd_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "cd_val"
            / "string"
            / "cd_val_canonical_finetuned_embeddings.csv"
        )

        pfas_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "pfas_val"
            / "string"
            / "pfas_val_canonical_finetuned_embeddings.csv"
        )
    else:
        cd_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "cd_val"
            / representation
            / f"cd_val_canonical_{representation}.csv"
        )

        pfas_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "pfas_val"
            / representation
            / f"pfas_val_canonical_{representation}.csv"
        )

    # Check if files exist
    if not cd_val_features_path.exists():
        logger.error(f"CD validation features file not found: {cd_val_features_path}")
        raise typer.Exit(1)

    if not pfas_val_features_path.exists():
        logger.error(
            f"PFAS validation features file not found: {pfas_val_features_path}"
        )
        raise typer.Exit(1)

    # Set up model path
    model_dir = MODELS_DIR / "external_validation" / representation
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"loocv_{model_type}_{representation}.pkl"

    # Initialize LOOCV trainer
    trainer = LOOCVTrainer(
        cd_val_path=cd_val_features_path,
        pfas_val_path=pfas_val_features_path,
        model_path=model_path,
        random_seed=42,
    )

    # Run hyperparameter tuning (retrieves best params from wandb after sweep)
    try:
        # Save best parameters retrieved from wandb
        suffix = f"_{representation}"
        trainer.save_best_params(suffix=suffix)

        logger.info(
            f"✅ Hyperparameter tuning completed successfully for {representation}"
        )
        logger.info(
            "🔧 Best parameters retrieved from wandb and saved for final training"
        )

    except Exception as e:
        logger.error(f"❌ Hyperparameter tuning failed: {e}")
        raise typer.Exit(1) from e


@app.command("loocv-tune-all")
def run_loocv_hyperparameter_tuning_all(
    model_type: str = typer.Option(
        "lgbm", help="Model type to use (currently only lgbm supported)"
    ),
    sweep_count: int = typer.Option(
        50, help="Number of hyperparameter sweep iterations per representation"
    ),
) -> None:
    """
    Run hyperparameter tuning for all LOOCV representations.
    """
    representations = ["unimol", "grover", "ecfp", "end2end", "end2end_finetuned"]

    logger.info(
        f"Running hyperparameter tuning for all representations: {representations}"
    )
    logger.info(f"Each representation will run {sweep_count} iterations")

    tuning_results = {}

    for representation in representations:
        logger.info(f"\n🔄 Tuning hyperparameters for: {representation}")

        try:
            # Set up paths for external validation data
            if representation == "end2end_finetuned":
                # Special handling for finetuned embeddings
                cd_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "cd_val"
                    / "string"
                    / "cd_val_canonical_finetuned_embeddings.csv"
                )

                pfas_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "pfas_val"
                    / "string"
                    / "pfas_val_canonical_finetuned_embeddings.csv"
                )
            else:
                cd_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "cd_val"
                    / representation
                    / f"cd_val_canonical_{representation}.csv"
                )

                pfas_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "pfas_val"
                    / representation
                    / f"pfas_val_canonical_{representation}.csv"
                )

            # Check if files exist
            if not cd_val_features_path.exists():
                logger.warning(
                    f"⚠️ Skipping {representation}: CD validation file not found"
                )
                continue

            if not pfas_val_features_path.exists():
                logger.warning(
                    f"⚠️ Skipping {representation}: PFAS validation file not found"
                )
                continue

            # Set up model path
            model_dir = MODELS_DIR / "external_validation" / representation
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"loocv_{model_type}_{representation}.pkl"

            # Initialize LOOCV trainer
            trainer = LOOCVTrainer(
                cd_val_path=cd_val_features_path,
                pfas_val_path=pfas_val_features_path,
                model_path=model_path,
                random_seed=42,
            )

            # Run hyperparameter tuning (retrieves best params from wandb after sweep completes)
            best_params = trainer.hyperparameter_tuning(
                sweep_config=lightgbm_sweep_config,
                representation=representation,
                sweep_count=sweep_count,
            )

            # Save best parameters retrieved from wandb
            suffix = f"_{representation}"
            trainer.save_best_params(suffix=suffix)

            # Store results summary
            tuning_results[representation] = {
                "best_params": best_params,
                "tuning_completed": True,
            }

            logger.info(f"✅ {representation} hyperparameter tuning completed")

        except Exception as e:
            logger.error(f"❌ Failed to tune hyperparameters for {representation}: {e}")
            tuning_results[representation] = {
                "tuning_completed": False,
                "error": str(e),
            }
            logger.warning("Continuing with next representation...")
            continue

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("🔧 HYPERPARAMETER TUNING SUMMARY")
    logger.info("=" * 60)

    completed_count = sum(
        1 for r in tuning_results.values() if r.get("tuning_completed", False)
    )
    failed_count = len(representations) - completed_count

    logger.info(
        f"✅ Successfully completed: {completed_count}/{len(representations)} representations"
    )
    if failed_count > 0:
        logger.info(f"❌ Failed: {failed_count}/{len(representations)} representations")

    for representation, result in tuning_results.items():
        if result.get("tuning_completed", False):
            logger.info(f"  ✅ {representation}: Best parameters saved")
        else:
            logger.info(
                f"  ❌ {representation}: {result.get('error', 'Unknown error')}"
            )


@app.command("loocv-final")
def run_loocv_with_best_params(
    representation: str = typer.Option(
        help="Data representation (unimol, grover, ecfp, end2end, end2end_finetuned)"
    ),
    model_type: str = typer.Option(
        "lgbm", help="Model type to use (currently only lgbm supported)"
    ),
    save_results: bool = typer.Option(
        True, help="Whether to save CSV and JSON result files (default: True)"
    ),
) -> None:
    """
    Run final LOOCV training using previously tuned hyperparameters.
    """
    if model_type != "lgbm":
        logger.error("Currently only LightGBM (lgbm) models are supported for LOOCV")
        raise typer.Exit(1)

    if representation not in [
        "unimol",
        "grover",
        "ecfp",
        "end2end",
        "end2end_finetuned",
    ]:
        logger.error(
            f"Unsupported representation: {representation}. Use: unimol, grover, ecfp, end2end, or end2end_finetuned"
        )
        raise typer.Exit(1)

    logger.info(f"Running final LOOCV for {representation} with best hyperparameters")

    # Set up paths for external validation data
    if representation == "end2end_finetuned":
        # Special handling for finetuned embeddings
        cd_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "cd_val"
            / "string"
            / "cd_val_canonical_finetuned_embeddings.csv"
        )

        pfas_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "pfas_val"
            / "string"
            / "pfas_val_canonical_finetuned_embeddings.csv"
        )
    else:
        cd_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "cd_val"
            / representation
            / f"cd_val_canonical_{representation}.csv"
        )

        pfas_val_features_path = (
            EXTERNAL_DATA_DIR
            / "validation"
            / "pfas_val"
            / representation
            / f"pfas_val_canonical_{representation}.csv"
        )

    # Check if files exist
    if not cd_val_features_path.exists():
        logger.error(f"CD validation features file not found: {cd_val_features_path}")
        raise typer.Exit(1)

    if not pfas_val_features_path.exists():
        logger.error(
            f"PFAS validation features file not found: {pfas_val_features_path}"
        )
        raise typer.Exit(1)

    # Set up model path
    model_dir = MODELS_DIR / "external_validation" / representation
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"loocv_{model_type}_{representation}_final.pkl"

    # Initialize LOOCV trainer
    trainer = LOOCVTrainer(
        cd_val_path=cd_val_features_path,
        pfas_val_path=pfas_val_features_path,
        model_path=model_path,
        random_seed=42,
    )

    # Load best parameters
    try:
        suffix = f"_{representation}"
        best_params = trainer.load_best_params(suffix=suffix)

        if best_params is None:
            logger.error(
                "No best parameters found. Run hyperparameter tuning first with 'loocv-tune' command."
            )
            raise typer.Exit(1)

        # Train final model with best parameters
        results = trainer.train_final_model_with_best_params(
            suffix=f"_{representation}_final", save_results=save_results
        )

        logger.info(f"✅ Final LOOCV completed successfully for {representation}")
        logger.info(f"📊 Final RMSE: {results['rmse']:.4f}")
        logger.info(f"📈 Final R²: {results['r2']:.4f}")

    except Exception as e:
        logger.error(f"❌ Final LOOCV training failed: {e}")
        raise typer.Exit(1) from e


@app.command("loocv-download-params")
def download_loocv_best_params(
    representation: str = typer.Option(
        None,
        help="Specific representation to download params for (unimol, grover, ecfp, end2end_finetuned). If None, downloads for all.",
    ),
    project_prefix: str = typer.Option(
        "LOOCV-LightGBM",
        help="Project name prefix in wandb (default: LOOCV-LightGBM)",
    ),
    project_suffix: str = typer.Option(
        "external-validation",
        help="Project name suffix in wandb (default: external-validation)",
    ),
) -> None:
    """
    Download best hyperparameters from completed wandb sweeps for LOOCV models.

    This command retrieves the best parameters from wandb based on the lowest
    validation RMSE across all runs in each sweep, and saves them to the
    appropriate directory for use with loocv-final commands.
    """
    # Determine which representations to process
    if representation:
        representations = [representation]
    else:
        representations = ["unimol", "grover", "ecfp", "smiles_tokenizer", "finetuned"]

    logger.info("Downloading best parameters from wandb for LOOCV models")
    logger.info(f"Representations: {representations}")

    api = wandb.Api()
    download_results = {}

    for rep in representations:
        logger.info(f"\n🔄 Processing {rep}...")

        try:
            # Construct project name
            project_name = f"{project_prefix}-{rep}-{project_suffix}"
            logger.info(f"Searching wandb project: {project_name}")

            # Get all sweeps for this project
            try:
                sweeps = api.project(project_name).sweeps()
                sweep_list = list(sweeps)

                if not sweep_list:
                    logger.warning(f"⚠️ No sweeps found in project {project_name}")
                    download_results[rep] = {
                        "success": False,
                        "error": "No sweeps found",
                    }
                    continue

                # Get the most recent sweep (assuming it's the one we want)
                latest_sweep = sweep_list[0]
                logger.info(f"Found sweep: {latest_sweep.id}")
                logger.info(f"Sweep name: {latest_sweep.name}")

            except Exception as e:
                logger.error(f"❌ Could not access project {project_name}: {e}")
                download_results[rep] = {"success": False, "error": str(e)}
                continue

            # Get best run from sweep
            runs = latest_sweep.runs
            if not runs:
                logger.warning(f"⚠️ No runs found in sweep {latest_sweep.id}")
                download_results[rep] = {"success": False, "error": "No runs in sweep"}
                continue

            best_run = min(
                runs, key=lambda run: run.summary.get("val_rmse", float("inf"))
            )

            logger.info(f"Best run: {best_run.name}")
            logger.info(f"  val_rmse: {best_run.summary.get('val_rmse', 'N/A'):.4f}")
            logger.info(f"  val_r2: {best_run.summary.get('val_r2', 'N/A'):.4f}")

            # Check if best_run has valid RMSE
            if best_run.summary.get("val_rmse") is None:
                logger.warning("⚠️ Best run has no val_rmse metric")
                download_results[rep] = {
                    "success": False,
                    "error": "No valid val_rmse in best run",
                }
                continue

            # Extract best hyperparameters with error handling
            try:
                best_params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "boosting_type": "gbdt",
                    "verbose": -1,
                    "seed": 42,
                    "num_leaves": best_run.config["num_leaves"],
                    "max_depth": best_run.config["max_depth"],
                    "learning_rate": best_run.config["learning_rate"],
                    "n_estimators": best_run.config["n_estimators"],
                    "reg_alpha": best_run.config["reg_alpha"],
                    "reg_lambda": best_run.config["reg_lambda"],
                    "bagging_fraction": best_run.config["bagging_fraction"],
                    "feature_fraction": best_run.config["feature_fraction"],
                }
            except KeyError as e:
                logger.error(f"❌ Missing hyperparameter in config: {e}")
                logger.error(f"Available config keys: {list(best_run.config.keys())}")
                download_results[rep] = {
                    "success": False,
                    "error": f"Missing config key: {e}",
                }
                continue

            # Determine the folder name for saving
            # Map project names to their corresponding folder names
            if rep == "smiles_tokenizer":
                folder_name = "string"
            elif rep == "finetuned":
                folder_name = "string_finetuned"
            else:
                folder_name = rep

            # Save parameters
            model_dir = MODELS_DIR / "external_validation" / folder_name
            model_dir.mkdir(parents=True, exist_ok=True)
            params_path = model_dir / f"loocv_best_params_{folder_name}.json"

            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=4)

            logger.info(f"✅ Saved best parameters to: {params_path}")

            download_results[rep] = {
                "success": True,
                "sweep_id": latest_sweep.id,
                "best_run": best_run.name,
                "val_rmse": best_run.summary.get("val_rmse"),
                "val_r2": best_run.summary.get("val_r2"),
                "params_path": str(params_path),
            }

        except Exception as e:
            logger.error(f"❌ Failed to download params for {rep}: {e}")
            download_results[rep] = {"success": False, "error": str(e)}
            continue

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📥 DOWNLOAD SUMMARY")
    logger.info("=" * 60)

    successful = [
        rep for rep, result in download_results.items() if result.get("success")
    ]
    failed = [
        rep for rep, result in download_results.items() if not result.get("success")
    ]

    logger.info(f"✅ Successfully downloaded: {len(successful)}/{len(representations)}")
    if successful:
        for rep in successful:
            result = download_results[rep]
            logger.info(f"  {rep}:")
            logger.info(
                f"    RMSE: {result['val_rmse']:.4f}, R²: {result['val_r2']:.4f}"
            )
            logger.info(f"    Saved to: {result['params_path']}")

    if failed:
        logger.info(f"\n❌ Failed: {len(failed)}/{len(representations)}")
        for rep in failed:
            logger.info(
                f"  {rep}: {download_results[rep].get('error', 'Unknown error')}"
            )

    if successful:
        logger.info(
            "\n✅ Ready to run: python cd_host_guest/modeling/train.py loocv-final-all"
        )


@app.command("loocv-final-all")
def run_loocv_final_all(
    model_type: str = typer.Option(
        "lgbm", help="Model type to use (currently only lgbm supported)"
    ),
    save_results: bool = typer.Option(
        True, help="Whether to save CSV and JSON result files (default: True)"
    ),
) -> None:
    """
    Run final LOOCV training for all representations using their best hyperparameters.
    """
    representations = ["unimol", "grover", "ecfp", "string", "string_finetuned"]

    logger.info(
        f"Running final LOOCV for all representations with best hyperparameters: {representations}"
    )

    final_results = {}

    for representation in representations:
        logger.info(f"\n🔄 Running final LOOCV for: {representation}")

        try:
            # Set up paths for external validation data
            if representation == "string_finetuned":
                # Special handling for finetuned embeddings
                cd_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "cd_val"
                    / "string"
                    / "cd_val_canonical_finetuned_embeddings.csv"
                )

                pfas_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "pfas_val"
                    / "string"
                    / "pfas_val_canonical_finetuned_embeddings.csv"
                )
            elif representation == "string":
                # Special handling for string embeddings
                cd_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "cd_val"
                    / "string"
                    / "cd_val_canonical_embeddings.csv"
                )

                pfas_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "pfas_val"
                    / "string"
                    / "pfas_val_canonical_embeddings.csv"
                )
            else:
                cd_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "cd_val"
                    / representation
                    / f"cd_val_canonical_{representation}.csv"
                )

                pfas_val_features_path = (
                    EXTERNAL_DATA_DIR
                    / "validation"
                    / "pfas_val"
                    / representation
                    / f"pfas_val_canonical_{representation}.csv"
                )

            # Check if files exist
            if not cd_val_features_path.exists():
                logger.warning(
                    f"⚠️ Skipping {representation}: CD validation file not found"
                )
                continue

            if not pfas_val_features_path.exists():
                logger.warning(
                    f"⚠️ Skipping {representation}: PFAS validation file not found"
                )
                continue

            # Set up model path
            model_dir = MODELS_DIR / "external_validation" / representation
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"loocv_{model_type}_{representation}_final.pkl"

            # Initialize LOOCV trainer
            trainer = LOOCVTrainer(
                cd_val_path=cd_val_features_path,
                pfas_val_path=pfas_val_features_path,
                model_path=model_path,
                random_seed=42,
            )

            # Load best parameters
            suffix = f"_{representation}"
            best_params = trainer.load_best_params(suffix=suffix)

            if best_params is None:
                logger.warning(
                    f"⚠️ No best parameters found for {representation}. Skipping."
                )
                logger.info(f"Run 'loocv-tune --representation {representation}' first")
                continue

            # Train final model with best parameters
            results = trainer.train_final_model_with_best_params(
                suffix=f"_{representation}_final", save_results=save_results
            )

            # Store summary
            final_results[representation] = {
                "rmse": results["rmse"],
                "mae": results["mae"],
                "r2": results["r2"],
                "n_samples": results["n_samples"],
                "completed": True,
            }

            logger.info(
                f"✅ {representation} completed - RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to run final LOOCV for {representation}: {e}")
            final_results[representation] = {"completed": False, "error": str(e)}
            logger.warning("Continuing with next representation...")
            continue

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("🏆 FINAL LOOCV RESULTS (WITH TUNED HYPERPARAMETERS)")
    logger.info("=" * 60)

    if final_results:
        completed_results = {
            k: v for k, v in final_results.items() if v.get("completed", False)
        }

        if completed_results:
            # Create summary DataFrame for easy comparison
            summary_df = pd.DataFrame(completed_results).T
            summary_df = summary_df[["rmse", "mae", "r2", "n_samples"]].round(4)

            logger.info(f"\n{summary_df.to_string()}")

            # Save summary
            summary_path = (
                MODELS_DIR / "external_validation" / "loocv_final_summary.csv"
            )
            summary_df.to_csv(summary_path)
            logger.info(f"\n📊 Final summary saved to: {summary_path}")

            # Find best performing model
            best_representation = summary_df["rmse"].idxmin()
            best_rmse = summary_df.loc[best_representation, "rmse"]
            best_r2 = summary_df.loc[best_representation, "r2"]

            logger.info(
                f"\n🥇 Best performing representation (tuned): {best_representation}"
            )
            logger.info(f"   RMSE: {best_rmse:.4f}")
            logger.info(f"   R²: {best_r2:.4f}")

        # Report any failures
        failed_results = {
            k: v for k, v in final_results.items() if not v.get("completed", False)
        }
        if failed_results:
            logger.info("\n❌ Failed representations:")
            for representation, result in failed_results.items():
                logger.info(
                    f"  {representation}: {result.get('error', 'Unknown error')}"
                )

    else:
        logger.error("❌ No representations were successfully processed")


if __name__ == "__main__":
    app()

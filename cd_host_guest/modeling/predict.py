import json
from abc import abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
import wandb
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from cd_host_guest.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class Predictor:
    """Base class for model prediction and evaluation"""

    def __init__(
        self,
        features_path: Path,
        model_path: Path,
        predictions_path: Path,
        wandb_log: bool = False,
        scaler_path: Path | None = None,
        label_scaler_path: Path | None = None,
    ):
        self.features_path = features_path
        self.model_path = model_path
        self.predictions_path = predictions_path
        self.wandb_log = wandb_log
        self.scaler_path = scaler_path
        self.label_scaler_path = label_scaler_path
        self.model: object | None = None
        self.scaler: StandardScaler | None = None
        self.label_scaler: StandardScaler | None = None

    def check_directories(self):
        """
        Check if the directories for model and features paths exist.
        Create the directory for the predictions path if it does not exist.
        """
        if not self.features_path.parent.exists():
            msg = (
                f"Directory for features at {self.features_path.parent} does not exist."
            )
            raise FileNotFoundError(msg)

        if not self.model_path.parent.exists():
            msg = f"Directory for model at {self.model_path.parent} does not exist."
            raise FileNotFoundError(msg)

        if not self.predictions_path.parent.exists():
            logger.info(
                f"Creating directory for model at {self.predictions_path.parent}"
            )
            self.predictions_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_model(self) -> None:
        """Load the trained model"""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the model"""

    def save_predictions(self, predictions: np.ndarray) -> None:
        """Save predictions to CSV in the model directory."""
        predictions_df = pd.DataFrame(predictions, columns=["DeltaG"])
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(self.predictions_path, index=False)
        logger.info(f"Predictions saved to {self.predictions_path}")


class LightGBMPredictor(Predictor):
    """LightGBM-specific predictor implementation"""

    def load_model(self) -> None:
        """Load the trained LightGBM model from disk"""
        try:
            self.model = joblib.load(str(self.model_path))
        except Exception as e:
            msg = f"Failed to load model: {e!s}"
            logger.error(msg)
            raise

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the LightGBM model"""
        if self.model is None:
            msg = "Model not loaded. Call load_model() first."
            raise ValueError(msg)
        return self.model.predict(features, num_iteration=self.model.best_iteration)


class FNNPredictor(Predictor):
    """FNN-specific predictor implementation"""

    def __init__(
        self,
        features_path: Path,
        model_path: Path,
        predictions_path: Path,
        wandb_log: bool = False,
        scaler_path: Path | None = None,
        label_scaler_path: Path | None = None,
        config_path: Path | None = None,
    ):
        super().__init__(
            features_path,
            model_path,
            predictions_path,
            wandb_log,
            scaler_path,
            label_scaler_path,
        )
        self.config_path = config_path
        self.input_dim: int | None = None
        self.hidden_dim: int = 128
        self.output_dim: int = 1
        self.num_layers: int = 2
        self.dropout: float = 0.2
        self.activation: str = "relu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self) -> torch.nn.Module:
        """Build FNN model with BatchNorm1d layers to match training architecture"""
        layers = [torch.nn.Linear(self.input_dim, self.hidden_dim)]
        layers.append(
            torch.nn.BatchNorm1d(self.hidden_dim)
        )  # Add BatchNorm after first layer
        activation_fn = (
            torch.nn.ReLU() if self.activation == "relu" else torch.nn.Tanh()
        )
        layers.append(activation_fn)
        layers.append(torch.nn.Dropout(self.dropout))

        for _ in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(
                torch.nn.BatchNorm1d(self.hidden_dim)
            )  # Add BatchNorm after each hidden layer
            layers.append(activation_fn)
            layers.append(torch.nn.Dropout(self.dropout))

        layers.append(torch.nn.Linear(self.hidden_dim, self.output_dim))
        return torch.nn.Sequential(*layers).to(self.device)

    def load_model(self) -> None:
        # Load scalers
        if self.scaler_path is None:
            raise ValueError("Scaler path must be provided for FNNPredictor.")
        self.scaler = joblib.load(self.scaler_path)
        # Load model hyperparameters from config JSON
        if self.config_path is None:
            # Try to infer config path from model_path
            config_guess = self.model_path.parent / (
                self.model_path.stem + "_config.json"
            )
            if config_guess.exists():
                self.config_path = config_guess
            else:
                msg = f"Config file not found for FNN model at {config_guess}"
                raise FileNotFoundError(msg)

        with open(self.config_path, encoding="utf-8") as f:
            config = json.load(f)
        self.hidden_dim = int(config.get("hidden_dim", 128))
        self.num_layers = int(config.get("num_layers", 2))
        self.dropout = float(config.get("dropout", 0.2))
        self.activation = str(config.get("activation", "relu"))
        # Load model architecture
        features = pd.read_csv(self.features_path)
        self.input_dim = features.shape[1]
        self.model = self.build_model()
        # Load weights
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.eval()
        # Load label scaler if available
        if self.label_scaler_path is not None and self.label_scaler_path.exists():
            self.label_scaler = joblib.load(self.label_scaler_path)
        else:
            self.label_scaler = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise ValueError(
                "Model and feature scaler must be loaded before prediction."
            )
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            preds = self.model(features_tensor).cpu().numpy().flatten()
        # Inverse transform predictions to original label scale if label scaler exists
        if self.label_scaler is not None:
            preds_orig = self.label_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).flatten()
            return preds_orig
        return preds


class End2EndPredictor(Predictor):
    """End2End-specific predictor implementation for SMILES string input"""

    def __init__(
        self,
        features_path: Path,
        model_path: Path,
        predictions_path: Path,
        wandb_log: bool = False,
        scaler_path: Path | None = None,
        label_scaler_path: Path | None = None,
        config_path: Path | None = None,
    ):
        super().__init__(
            features_path,
            model_path,
            predictions_path,
            wandb_log,
            scaler_path,
            label_scaler_path,
        )
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.transformer = None

    def load_model(self) -> None:
        """Load the trained end2end model from disk"""
        # Import required modules for model building

        # Load model hyperparameters from config JSON if available
        if self.config_path is not None and self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                config = json.load(f)
            hidden_dim = int(config.get("hidden_dim", 128))
            num_layers = int(config.get("num_layers", 2))
            dropout = float(config.get("dropout", 0.2))
            activation = str(config.get("activation", "relu"))
        else:
            # Default values if no config file
            hidden_dim = 128
            num_layers = 2
            dropout = 0.2
            activation = "relu"

        # Load pretrained tokenizer and transformer
        self.tokenizer, self.transformer = self.build_SMILESTokenizer_layer()

        # Freeze all transformer parameters by default
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Check the saved state_dict to determine the model architecture
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Check if it's the new architecture with BatchNorm
        model_uses_batchnorm = "batchnorm.weight" in state_dict

        logger.info(
            f"Model was trained with {'BatchNorm' if model_uses_batchnorm else 'LayerNorm/Scaler'} architecture"
        )

        # Define SMILES encoder module
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
                # Use the [CLS] token embedding as the representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                return cls_embedding

        # Define the full end2end model
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
                # Concatenate as [host, guest] to match training
                x = torch.cat([host_emb, guest_emb], dim=1)
                # Apply BatchNorm to normalize the embeddings
                x = self.batchnorm(x)
                out = self.fnn(x)
                return out.squeeze(-1)

        # Build FNN component
        transformer_dim = self.transformer.config.hidden_size
        fnn_input_dim = 2 * transformer_dim

        fnn = self.build_FNN_layer(
            self.device,
            input_dim=fnn_input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

        # Create encoders
        host_encoder = SMILESEncoder(self.transformer, self.tokenizer, self.device)
        guest_encoder = SMILESEncoder(self.transformer, self.tokenizer, self.device)

        # Create the full model with BatchNorm architecture
        self.model = EndToEndModel(
            host_encoder,
            guest_encoder,
            fnn,
            fnn_input_dim,
        ).to(self.device)

        # Load the trained weights
        try:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Successfully loaded end2end model from {self.model_path}")
        except Exception as e:
            msg = f"Failed to load end2end model: {e}"
            logger.error(msg)
            raise

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
        Build a feedforward neural network with BatchNorm layers to match training architecture.
        """

        layers = [torch.nn.Linear(input_dim, hidden_dim)]
        activation_fn = torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh()
        layers.append(nn.BatchNorm1d(hidden_dim))
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
        model.to(self.device)
        return tokenizer, model

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the end2end model with SMILES strings"""
        if self.model is None:
            msg = "Model not loaded. Call load_model() first."
            raise ValueError(msg)

        # Extract host and guest SMILES from the dataframe
        # Try different possible column names for host and guest SMILES
        host_col = None
        guest_col = None

        # Check for various possible column names
        possible_host_cols = ["Host_SMILES", "Host", "host_smiles", "host"]
        possible_guest_cols = ["Guest_SMILES", "Guest", "guest_smiles", "guest"]

        for col in possible_host_cols:
            if col in features.columns:
                host_col = col
                break

        for col in possible_guest_cols:
            if col in features.columns:
                guest_col = col
                break

        if host_col is None or guest_col is None:
            available_cols = list(features.columns)
            msg = f"Features dataframe must contain host and guest SMILES columns. Available columns: {available_cols}. Expected one of {possible_host_cols} for host and one of {possible_guest_cols} for guest."
            raise ValueError(msg)

        host_smiles = features[host_col].tolist()
        guest_smiles = features[guest_col].tolist()

        predictions = []
        batch_size = 32  # Process in batches to manage memory

        with torch.no_grad():
            for i in range(0, len(host_smiles), batch_size):
                batch_host = host_smiles[i : i + batch_size]
                batch_guest = guest_smiles[i : i + batch_size]

                batch_preds = self.model(batch_host, batch_guest)
                # The model already applies squeeze(-1), so the output is 1D
                predictions.extend(batch_preds.cpu().numpy())

        predictions = np.array(predictions)

        # Inverse transform predictions if label scaler exists
        if self.label_scaler is not None:
            predictions = self.label_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions


def get_training_metrics_from_wandb(
    model_type: str,
    representation: str,
    data_source: str = "OpenCycloDB",  # noqa: ARG001
) -> dict:
    """
    Fetch training and validation metrics from WandB for a specific model.

    Args:
        model_type: Type of model (e.g., 'fnn', 'lgbm', 'end2end')
        representation: Data representation (e.g., 'ecfp', 'grover', 'string')
        data_source: Data source used for training

    Returns:
        Dictionary containing training metrics or empty dict if not found
    """
    try:
        api = wandb.Api()
        project_name = f"{model_type}_{representation}"

        # Get finished runs from the project
        runs = api.runs(f"cbrzakala-tu-delft/{project_name}")
        finished_runs = [run for run in runs if run.state == "finished"]

        if not finished_runs:
            logger.warning(f"No finished runs found for project {project_name}")
            return {}

        # Find the best run (lowest validation loss/rmse)
        val_rmse_runs = [run for run in finished_runs if "val_rmse" in run.summary]
        if val_rmse_runs:
            best_run = min(val_rmse_runs, key=lambda r: r.summary["val_rmse"])
            metric_used = "val_rmse"
        else:
            val_loss_runs = [run for run in finished_runs if "val_loss" in run.summary]
            if val_loss_runs:
                best_run = min(val_loss_runs, key=lambda r: r.summary["val_loss"])
                metric_used = "val_loss"
            else:
                logger.warning(
                    f"No validation metrics found for project {project_name}"
                )
                return {}

        # Extract metrics from the best run's summary
        training_metrics = {}

        # Get final training and validation metrics with comprehensive pattern matching
        summary = best_run.summary
        logger.debug(
            f"Available summary keys for {project_name}: {list(summary.keys())}"
        )

        for key, value in summary.items():
            # More comprehensive keyword matching for training-related metrics
            key_lower = key.lower()
            if any(
                metric in key_lower
                for metric in [
                    "train",
                    "val",
                    "validation",
                    "loss",
                    "rmse",
                    "mae",
                    "r2",
                    "mse",
                    "error",
                    "score",
                    "metric",
                    "eval",
                ]
            ):
                try:
                    training_metrics[key] = (
                        float(value) if isinstance(value, int | float) else value
                    )
                except (ValueError, TypeError):
                    training_metrics[key] = value

        # Add run metadata
        training_metrics["wandb_run_id"] = best_run.id
        training_metrics["wandb_run_name"] = best_run.name
        training_metrics["best_metric_used"] = metric_used
        training_metrics["best_metric_value"] = float(best_run.summary[metric_used])

        if training_metrics:
            logger.info(
                f"Retrieved training metrics for {project_name} (run: {best_run.name})"
            )
            logger.debug(
                f"Training metrics keys found: {[k for k in training_metrics if k not in ['wandb_run_id', 'wandb_run_name', 'best_metric_used', 'best_metric_value']]}"
            )
        else:
            logger.warning(
                f"No training-related metrics found in summary for {project_name}"
            )

        return training_metrics

    except Exception as e:
        logger.error(
            f"Failed to fetch training metrics for {model_type}_{representation}: {e}"
        )
        return {}


def calculate_prediction_metrics(
    predictions: np.ndarray, true_values: np.ndarray, model_name: str = "Model"
) -> dict:
    """
    Calculate comprehensive metrics for model predictions.

    Args:
        predictions: Array of predicted values
        true_values: Array of true/observed values
        model_name: Name of the model for logging purposes

    Returns:
        Dictionary containing all calculated metrics
    """
    # Ensure arrays are flattened and of the same length
    predictions = np.array(predictions).flatten()
    true_values = np.array(true_values).flatten()

    if len(predictions) != len(true_values):
        error_msg = f"Predictions and true values must have the same length. Got {len(predictions)} and {len(true_values)}"
        raise ValueError(error_msg)

    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)

    # Pearson correlation
    correlation, p_value = pearsonr(predictions, true_values)

    # Additional statistics
    residuals = true_values - predictions
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Prediction statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    true_mean = np.mean(true_values)
    true_std = np.std(true_values)

    metrics = {
        "model_name": model_name,
        "n_samples": len(predictions),
        "rmse": float(rmse),
        "mae": float(mae),
        "mse": float(mse),
        "r2_score": float(r2),
        "pearson_correlation": float(correlation),
        "correlation_p_value": float(p_value),
        "mean_residual": float(mean_residual),
        "std_residual": float(std_residual),
        "predictions_mean": float(pred_mean),
        "predictions_std": float(pred_std),
        "true_values_mean": float(true_mean),
        "true_values_std": float(true_std),
        "predictions_range": [float(np.min(predictions)), float(np.max(predictions))],
        "true_values_range": [float(np.min(true_values)), float(np.max(true_values))],
    }

    return metrics


def compare_all_model_predictions(
    data_source: str = "OpenCycloDB",
    use_finetuned_end2end: bool = False,
    save_results: bool = True,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Compare predictions from all available models against true values.

    Args:
        data_source: Data source to use (default: OpenCycloDB)
        use_finetuned_end2end: Whether to use fine-tuned end2end model
        save_results: Whether to save results to CSV
        output_dir: Directory to save results (default: models directory)

    Returns:
        DataFrame containing metrics for all models
    """
    # Define model configurations
    model_configs = [
        # LGBM models
        ("lgbm", "ecfp", False),
        ("lgbm", "ecfp_plus", False),
        ("lgbm", "original_enriched", False),
        ("lgbm", "grover", False),
        ("lgbm", "unimol", False),
        # FNN models
        ("fnn", "ecfp", False),
        ("fnn", "ecfp_plus", False),
        ("fnn", "original_enriched", False),
        ("fnn", "grover", False),
        ("fnn", "unimol", False),
        # End2End models
        ("end2end", "string", False),
    ]

    # Add fine-tuned end2end if requested
    if use_finetuned_end2end:
        model_configs.append(("end2end", "string", True))

    all_metrics = []

    for model_type, representation, use_finetuned in model_configs:
        try:
            logger.info(
                f"Evaluating {model_type} with {representation} representation{' (fine-tuned)' if use_finetuned else ''}"
            )

            # Determine paths
            if model_type == "end2end":
                model_dir = MODELS_DIR / data_source / "string"
                processed_dir = PROCESSED_DATA_DIR / data_source / "string"

                if use_finetuned:
                    model_basename = "end2end_string_finetuned_final"
                    model_name = "End2End (Fine-tuned)"
                else:
                    model_basename = "end2end_string"
                    model_name = "End2End"

                predictions_path = (
                    model_dir / "predictions" / f"{model_basename}_predictions.csv"
                )
            else:
                model_dir = MODELS_DIR / data_source / representation
                processed_dir = PROCESSED_DATA_DIR / data_source / representation
                predictions_path = (
                    model_dir
                    / "predictions"
                    / f"{model_type}_{representation}_predictions.csv"
                )
                model_name = f"{model_type.upper()} ({representation})"

            # Load predictions
            if not predictions_path.exists():
                logger.warning(f"Predictions file not found: {predictions_path}")
                continue

            predictions_df = pd.read_csv(predictions_path)
            predictions = predictions_df["DeltaG"].values

            # Load true values
            labels_path = processed_dir / "test_labels.csv"
            if not labels_path.exists():
                logger.warning(f"Test labels file not found: {labels_path}")
                continue

            true_values_df = pd.read_csv(labels_path)
            # Handle different possible column names for labels
            label_cols = ["DeltaG", "labels", "target", "y"]
            true_values = None
            for col in label_cols:
                if col in true_values_df.columns:
                    true_values = true_values_df[col].values
                    break

            if true_values is None:
                logger.warning(
                    f"No recognized label column found in {labels_path}. Available columns: {list(true_values_df.columns)}"
                )
                continue

            # Calculate prediction metrics
            metrics = calculate_prediction_metrics(predictions, true_values, model_name)

            # Fetch training metrics from WandB if available
            wandb_representation = (
                representation if model_type != "end2end" else "string"
            )
            training_metrics = get_training_metrics_from_wandb(
                model_type, wandb_representation, data_source
            )

            # Merge training metrics with prediction metrics
            if training_metrics:
                # Add training metrics to the main metrics dict
                training_count = 0
                for key, value in training_metrics.items():
                    if (
                        key not in metrics
                    ):  # Don't overwrite existing prediction metrics
                        metrics[f"training_{key}"] = value
                        training_count += 1

                logger.info(
                    f"  Added {training_count} training metrics from WandB run: {training_metrics.get('wandb_run_name', 'unknown')}"
                )

                # Log which specific training/validation metrics were found
                train_keys = [
                    k
                    for k in training_metrics
                    if "train" in k.lower() and k not in ["wandb_run_name"]
                ]
                val_keys = [
                    k
                    for k in training_metrics
                    if any(x in k.lower() for x in ["val", "validation"])
                    and k not in ["wandb_run_name"]
                ]

                if train_keys or val_keys:
                    logger.debug(
                        f"  Training metrics: {train_keys}, Validation metrics: {val_keys}"
                    )
            else:
                logger.warning(
                    f"  No training metrics found in WandB for {model_type}_{wandb_representation}"
                )

            all_metrics.append(metrics)

            # Log key metrics
            logger.info(
                f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2_score']:.4f}, Correlation: {metrics['pearson_correlation']:.4f}"
            )

            # Log training metrics if available - check for various possible naming patterns
            val_metric = None
            train_metric = None

            # Look for validation metrics with different naming patterns
            for key in metrics:
                if key.startswith("training_"):
                    key_suffix = key[9:]  # Remove 'training_' prefix
                    if any(
                        val_pattern in key_suffix.lower()
                        for val_pattern in [
                            "val_rmse",
                            "val_loss",
                            "validation_rmse",
                            "validation_loss",
                        ]
                    ):
                        if val_metric is None:  # Take the first one found
                            val_metric = (key, metrics[key])
                    elif (
                        any(
                            train_pattern in key_suffix.lower()
                            for train_pattern in [
                                "train_rmse",
                                "train_loss",
                                "training_rmse",
                                "training_loss",
                            ]
                        )
                        and train_metric is None
                    ):  # Take the first one found
                        train_metric = (key, metrics[key])

            # Log the training metrics if found
            if val_metric or train_metric:
                training_info = ""
                if val_metric:
                    training_info += f"Val: {val_metric[1]:.4f}"
                if train_metric:
                    if training_info:
                        training_info += " | "
                    training_info += f"Train: {train_metric[1]:.4f}"
                logger.info(f"  Training - {training_info}")
            else:
                # Log individual metrics if the paired approach didn't work
                found_any = False
                for key in metrics:
                    if key.startswith("training_"):
                        key_suffix = key[9:]
                        if (
                            any(
                                pattern in key_suffix.lower()
                                for pattern in ["val", "validation", "train"]
                            )
                            and not found_any
                        ):
                            logger.info(
                                f"  Training - {key_suffix}: {metrics[key]:.4f}"
                            )
                            found_any = True

        except Exception as e:
            logger.error(f"Failed to evaluate {model_type} {representation}: {e}")
            continue

    if not all_metrics:
        logger.error("No models could be evaluated successfully")
        return pd.DataFrame()

    # Create results DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Sort by RMSE (best first)
    results_df = results_df.sort_values("rmse")

    # Add ranking column
    results_df["rmse_rank"] = range(1, len(results_df) + 1)

    # Reorder columns for better readability
    # Define base columns that should always be first
    base_columns = [
        "rmse_rank",
        "model_name",
        "rmse",
        "mae",
        "r2_score",
        "pearson_correlation",
        "n_samples",
    ]

    # Define training/validation columns
    training_columns = [
        col for col in results_df.columns if col.startswith("training_")
    ]

    # Define other prediction metrics columns
    other_columns = [
        "mse",
        "correlation_p_value",
        "mean_residual",
        "std_residual",
        "predictions_mean",
        "predictions_std",
        "true_values_mean",
        "true_values_std",
        "predictions_range",
        "true_values_range",
    ]

    # Build final column order, only including columns that exist
    column_order = []
    for col_list in [base_columns, training_columns, other_columns]:
        column_order.extend([col for col in col_list if col in results_df.columns])

    results_df = results_df[column_order]

    # Save results if requested
    if save_results:
        if output_dir is None:
            output_dir = MODELS_DIR / data_source

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_path = output_dir / "model_comparison_detailed.csv"
        results_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed model comparison saved to {detailed_path}")

        # Save summary results (key metrics including training metrics)
        summary_cols = [
            "rmse_rank",
            "model_name",
            "rmse",
            "mae",
            "r2_score",
            "pearson_correlation",
            "n_samples",
        ]

        # Add key training metrics if available - check for various naming patterns
        training_metric_patterns = [
            "training_val_rmse",
            "training_validation_rmse",
            "training_val_loss",
            "training_validation_loss",
            "training_train_rmse",
            "training_training_rmse",
            "training_train_loss",
            "training_training_loss",
        ]

        for pattern in training_metric_patterns:
            if pattern in results_df.columns:
                summary_cols.append(pattern)

        summary_df = results_df[summary_cols]
        summary_path = output_dir / "model_comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary model comparison saved to {summary_path}")

        # Debug info
        logger.debug(f"Summary DataFrame shape: {summary_df.shape}")
        logger.debug(f"Summary DataFrame columns: {list(summary_df.columns)}")

        # Log summary to console
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 80)

        try:
            for _idx, row in summary_df.iterrows():
                # Basic metrics
                basic_info = f"{row['rmse_rank']:2d}. {row['model_name']:<25} | RMSE: {row['rmse']:.4f} | MAE: {row['mae']:.4f} | R²: {row['r2_score']:.4f} | Corr: {row['pearson_correlation']:.4f}"

                # Add training metrics if available
                training_info = ""
                try:
                    # Look for validation metrics
                    val_cols = [
                        col
                        for col in summary_df.columns
                        if col.startswith("training_")
                        and any(x in col.lower() for x in ["val", "validation"])
                    ]
                    train_cols = [
                        col
                        for col in summary_df.columns
                        if col.startswith("training_")
                        and "train" in col.lower()
                        and not any(x in col.lower() for x in ["val", "validation"])
                    ]

                    # Use the first available validation and training metrics
                    if val_cols and pd.notna(row.get(val_cols[0])):
                        training_info += f" | Val: {row[val_cols[0]]:.4f}"
                    if train_cols and pd.notna(row.get(train_cols[0])):
                        training_info += f" | Train: {row[train_cols[0]]:.4f}"

                except Exception as e:
                    logger.debug(f"Error adding training metrics for row: {e}")

                logger.info(basic_info + training_info)
        except Exception as e:
            logger.error(f"Error during summary logging: {e}")

        logger.info("=" * 80)
        logger.info(f"Total models compared: {len(summary_df)}")

    return results_df


def evaluate_external_validation_predictions(
    val_set: str,
    save_results: bool = True,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Evaluate predictions from all available models on an external validation dataset.

    Args:
        val_set: Name of the validation set (e.g., 'cd_val', 'pfas_val')
        save_results: Whether to save results to CSV
        output_dir: Directory to save results (default: validation set directory)

    Returns:
        DataFrame containing metrics for all models on the validation set
    """
    from cd_host_guest.config import EXTERNAL_DATA_DIR

    # Define model configurations for external validation
    model_configs = [
        # LGBM models
        ("lgbm", "ecfp", False, "lgbm_ecfp_predictions.csv"),
        ("lgbm", "ecfp_plus", False, "lgbm_ecfp_plus_predictions.csv"),
        ("lgbm", "grover", False, "lgbm_grover_predictions.csv"),
        ("lgbm", "unimol", False, "lgbm_unimol_predictions.csv"),
        # FNN models
        ("fnn", "ecfp", False, "fnn_ecfp_predictions.csv"),
        ("fnn", "ecfp_plus", False, "fnn_ecfp_plus_predictions.csv"),
        ("fnn", "grover", False, "fnn_grover_predictions.csv"),
        ("fnn", "unimol", False, "fnn_unimol_predictions.csv"),
        # End2End models
        ("end2end", "string", False, "end2end_string_predictions.csv"),
        ("end2end", "string", True, "end2end_string_finetuned_final_predictions.csv"),
    ]

    # Load true values from canonical CSV
    base_dir = EXTERNAL_DATA_DIR / "validation" / val_set
    canonical_csv = base_dir / f"{val_set}_canonical.csv"

    if not canonical_csv.exists():
        logger.error(
            f"Validation set '{val_set}' not found or missing canonical CSV at {canonical_csv}"
        )
        return pd.DataFrame()

    try:
        df_true = pd.read_csv(canonical_csv)
        if "delG" not in df_true.columns:
            logger.error(
                f"'delG' column not found in {canonical_csv}. Available columns: {list(df_true.columns)}"
            )
            return pd.DataFrame()
        true_values = df_true["delG"].astype(float).values
        logger.info(f"Loaded {len(true_values)} true values from {canonical_csv}")
    except Exception as e:
        logger.error(f"Failed to load true values from {canonical_csv}: {e}")
        return pd.DataFrame()

    all_metrics = []

    for model_type, representation, is_finetuned, pred_filename in model_configs:
        try:
            # Determine prediction file path
            if model_type == "end2end":
                pred_dir = base_dir / "string"
                model_name = f"End2End{' (Fine-tuned)' if is_finetuned else ''}"
            else:
                pred_dir = base_dir / representation
                model_name = f"{model_type.upper()} ({representation})"

            pred_file = pred_dir / pred_filename

            logger.info(f"Evaluating {model_name} on {val_set}")

            # Load predictions
            if not pred_file.exists():
                logger.warning(f"Predictions file not found: {pred_file}")
                continue

            try:
                predictions_df = pd.read_csv(pred_file)
                if "DeltaG" not in predictions_df.columns:
                    logger.warning(
                        f"'DeltaG' column not found in {pred_file}. Available columns: {list(predictions_df.columns)}"
                    )
                    continue

                predictions = predictions_df["DeltaG"].astype(float).values

                # Ensure same length (truncate to minimum length if necessary)
                min_len = min(len(true_values), len(predictions))
                true_vals_subset = true_values[:min_len]
                pred_vals_subset = predictions[:min_len]

                if len(true_values) != len(predictions):
                    logger.warning(
                        f"Length mismatch for {model_name}: true={len(true_values)}, pred={len(predictions)}. Using first {min_len} samples."
                    )

            except Exception as e:
                logger.error(f"Failed to load predictions from {pred_file}: {e}")
                continue

            # Calculate prediction metrics
            metrics = calculate_prediction_metrics(
                pred_vals_subset, true_vals_subset, model_name
            )

            # Add validation set information
            metrics["validation_set"] = val_set
            metrics["model_type"] = model_type
            metrics["representation"] = representation
            metrics["is_finetuned"] = is_finetuned
            metrics["predictions_file"] = str(pred_file)

            all_metrics.append(metrics)

            # Log key metrics
            logger.info(
                f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2_score']:.4f}, Correlation: {metrics['pearson_correlation']:.4f}"
            )

        except Exception as e:
            logger.error(
                f"Failed to evaluate {model_type} {representation}{' (fine-tuned)' if is_finetuned else ''}: {e}"
            )
            continue

    if not all_metrics:
        logger.error(f"No models could be evaluated successfully on {val_set}")
        return pd.DataFrame()

    # Create results DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Sort by RMSE (best first)
    results_df = results_df.sort_values("rmse")

    # Add ranking column
    results_df["rmse_rank"] = range(1, len(results_df) + 1)

    # Reorder columns for better readability
    base_columns = [
        "rmse_rank",
        "validation_set",
        "model_name",
        "model_type",
        "representation",
        "is_finetuned",
        "rmse",
        "mae",
        "r2_score",
        "pearson_correlation",
        "n_samples",
    ]

    other_columns = [
        "mse",
        "correlation_p_value",
        "mean_residual",
        "std_residual",
        "predictions_mean",
        "predictions_std",
        "true_values_mean",
        "true_values_std",
        "predictions_range",
        "true_values_range",
        "predictions_file",
    ]

    # Build final column order, only including columns that exist
    column_order = []
    for col_list in [base_columns, other_columns]:
        column_order.extend([col for col in col_list if col in results_df.columns])

    results_df = results_df[column_order]

    # Save results if requested
    if save_results:
        if output_dir is None:
            output_dir = base_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_path = output_dir / f"{val_set}_model_evaluation_detailed.csv"
        results_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed evaluation results saved to {detailed_path}")

        # Save summary results
        summary_cols = [
            "rmse_rank",
            "validation_set",
            "model_name",
            "rmse",
            "mae",
            "r2_score",
            "pearson_correlation",
            "n_samples",
        ]

        summary_df = results_df[summary_cols]
        summary_path = output_dir / f"{val_set}_model_evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary evaluation results saved to {summary_path}")

        # Log summary to console
        logger.info("\n" + "=" * 80)
        logger.info(f"EXTERNAL VALIDATION RESULTS: {val_set.upper()}")
        logger.info("=" * 80)

        for _idx, row in summary_df.iterrows():
            model_info = f"{row['rmse_rank']:2d}. {row['model_name']:<25}"
            metrics_info = f"RMSE: {row['rmse']:.4f} | MAE: {row['mae']:.4f} | R²: {row['r2_score']:.4f} | Corr: {row['pearson_correlation']:.4f}"
            logger.info(f"{model_info} | {metrics_info}")

        logger.info("=" * 80)
        logger.info(
            f"Total models evaluated: {len(summary_df)} | Best model: {summary_df.iloc[0]['model_name']}"
        )

    return results_df


def get_comprehensive_model_metrics(
    data_source: str = "OpenCycloDB",
    save_results: bool = True,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Get comprehensive training, validation, and test metrics for all models.

    This function retrieves training and validation metrics from WandB (unbiased),
    and calculates test metrics by making predictions on the test set.

    Args:
        data_source: Data source to use (default: OpenCycloDB)
        save_results: Whether to save results to CSV
        output_dir: Directory to save results (default: models directory)

    Returns:
        DataFrame containing comprehensive metrics for all models
    """
    # Define all model configurations including both end2end models for string representation
    model_configs = [
        # LGBM models
        ("lgbm", "ecfp", False),
        ("lgbm", "ecfp_plus", False),
        ("lgbm", "original_enriched", False),
        ("lgbm", "grover", False),
        ("lgbm", "unimol", False),
        # FNN models
        ("fnn", "ecfp", False),
        ("fnn", "ecfp_plus", False),
        ("fnn", "original_enriched", False),
        ("fnn", "grover", False),
        ("fnn", "unimol", False),
        # End2End models (both base and fine-tuned for string)
        ("end2end", "string", False),
        ("end2end", "string", True),  # Fine-tuned version
    ]

    all_metrics = []

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL METRICS EVALUATION")
    logger.info("Using WandB for train/val metrics, calculating test metrics directly")
    logger.info("=" * 80)

    for model_type, representation, use_finetuned in model_configs:
        try:
            # Create model identifier
            model_suffix = "_finetuned_final" if use_finetuned else ""
            model_name = f"{model_type}_{representation}{model_suffix}"

            logger.info(f"Processing: {model_name}")

            # 1. Get training and validation metrics from WandB
            # For fine-tuned models, we need to look at the finetune project
            if use_finetuned and model_type == "end2end":
                # Try both the finetune project and the regular project
                training_metrics = {}
                project_found = None
                for project_suffix in ["_finetuned_final", "_finetune", ""]:
                    try:
                        project_name = representation + project_suffix
                        logger.info(f"  Trying WandB project: end2end_{project_name}")
                        temp_metrics = get_training_metrics_from_wandb(
                            model_type=model_type,
                            representation=project_name,
                            data_source=data_source,
                        )
                        if temp_metrics:
                            training_metrics = temp_metrics
                            project_found = f"end2end_{project_name}"
                            break
                    except Exception as e:
                        logger.warning(
                            f"  Failed to get metrics from end2end_{project_name}: {e}"
                        )
                        continue

                # If still no metrics, try with the original representation name
                if not training_metrics:
                    logger.info(
                        f"  Falling back to original project: end2end_{representation}"
                    )
                    training_metrics = get_training_metrics_from_wandb(
                        model_type=model_type,
                        representation=representation,
                        data_source=data_source,
                    )
                    project_found = (
                        f"end2end_{representation}" if training_metrics else None
                    )

                if project_found:
                    logger.info(f"  Using metrics from WandB project: {project_found}")
                else:
                    logger.warning(
                        f"  No WandB metrics found for finetuned {model_name}"
                    )
            else:
                training_metrics = get_training_metrics_from_wandb(
                    model_type=model_type,
                    representation=representation,
                    data_source=data_source,
                )

            # Debug: log available training metrics
            if training_metrics:
                available_metrics = [
                    k
                    for k in training_metrics
                    if k
                    not in [
                        "wandb_run_id",
                        "wandb_run_name",
                        "best_metric_used",
                        "best_metric_value",
                    ]
                ]
                logger.info(
                    f"Available training metrics for {model_name}: {available_metrics}"
                )

                # Log specific metric values for debugging
                for metric_name in [
                    "train_rmse",
                    "val_rmse",
                    "training_rmse",
                    "validation_rmse",
                    "train_loss",
                    "val_loss",
                ]:
                    if metric_name in training_metrics:
                        logger.info(f"  {metric_name}: {training_metrics[metric_name]}")
            else:
                logger.warning(f"No training metrics found for {model_name}")

            # 2. Calculate missing train RMSE for LGBM models if not in WandB
            if model_type == "lgbm" and not any(
                k in training_metrics
                for k in ["train_rmse", "training_rmse", "train_loss"]
            ):
                logger.info(
                    f"  Train RMSE missing from WandB for {model_name}, calculating manually..."
                )
                try:
                    # Load training data
                    train_features_path = (
                        PROCESSED_DATA_DIR
                        / data_source
                        / representation
                        / "train_features.csv"
                    )
                    train_labels_path = (
                        PROCESSED_DATA_DIR
                        / data_source
                        / representation
                        / "train_labels.csv"
                    )

                    if train_features_path.exists() and train_labels_path.exists():
                        train_features = pd.read_csv(train_features_path)
                        train_labels = pd.read_csv(train_labels_path)["DeltaG"].values

                        # Determine model path
                        model_dir = MODELS_DIR / data_source / representation
                        model_filename = f"{model_type}_{representation}.pkl"
                        model_path = model_dir / model_filename

                        if model_path.exists():
                            # Create LGBM predictor and calculate train RMSE
                            predictions_path = (
                                model_dir
                                / "predictions"
                                / f"{model_type}_{representation}_predictions.csv"
                            )
                            predictor = LightGBMPredictor(
                                features_path=train_features_path,
                                model_path=model_path,
                                predictions_path=predictions_path,
                            )
                            predictor.load_model()
                            train_predictions = predictor.predict(train_features)

                            # Calculate train RMSE
                            train_rmse = np.sqrt(
                                np.mean((train_predictions - train_labels) ** 2)
                            )
                            training_metrics["train_rmse"] = train_rmse
                            logger.info(f"  Calculated train RMSE: {train_rmse:.4f}")
                        else:
                            logger.warning(f"  Could not find model file: {model_path}")
                    else:
                        logger.warning(
                            f"  Could not find training data for {model_name}"
                        )
                except Exception as e:
                    logger.error(
                        f"  Failed to calculate train RMSE for {model_name}: {e}"
                    )

            # 3. Calculate test metrics by making predictions
            test_metrics = {}
            try:
                # Load test data
                test_features_path = (
                    PROCESSED_DATA_DIR
                    / data_source
                    / representation
                    / "test_features.csv"
                )
                test_labels_path = (
                    PROCESSED_DATA_DIR
                    / data_source
                    / representation
                    / "test_labels.csv"
                )

                if not test_features_path.exists() or not test_labels_path.exists():
                    logger.warning(f"Test data not found for {model_name}")
                    test_metrics = {"test_error": "Test data not found"}
                else:
                    test_features = pd.read_csv(test_features_path)
                    test_labels = pd.read_csv(test_labels_path)["DeltaG"].values

                    # Determine model and prediction paths
                    model_dir = MODELS_DIR / data_source / representation

                    if model_type == "end2end":
                        model_filename = f"end2end_{representation}{model_suffix}.pth"
                        config_filename = (
                            f"end2end_{representation}{model_suffix}_config.json"
                        )
                        predictions_filename = (
                            f"end2end_{representation}{model_suffix}_predictions.csv"
                        )
                    else:
                        model_filename = f"{model_type}_{representation}.{'pkl' if model_type == 'lgbm' else 'pth'}"
                        config_filename = f"{model_type}_{representation}_config.json"
                        predictions_filename = (
                            f"{model_type}_{representation}_predictions.csv"
                        )

                    model_path = model_dir / model_filename
                    config_path = model_dir / config_filename
                    predictions_path = model_dir / "predictions" / predictions_filename

                    if not model_path.exists():
                        logger.warning(f"Model file not found: {model_path}")
                        test_metrics = {"test_error": "Model file not found"}
                    else:
                        # Create appropriate predictor and make predictions
                        if model_type == "lgbm":
                            predictor = LightGBMPredictor(
                                features_path=test_features_path,
                                model_path=model_path,
                                predictions_path=predictions_path,
                            )
                        elif model_type == "fnn":
                            scaler_path = (
                                model_dir / f"{model_type}_{representation}_scaler.pkl"
                            )
                            predictor = FNNPredictor(
                                features_path=test_features_path,
                                model_path=model_path,
                                predictions_path=predictions_path,
                                scaler_path=scaler_path,
                                config_path=config_path,
                            )
                        elif model_type == "end2end":
                            predictor = End2EndPredictor(
                                features_path=test_features_path,
                                model_path=model_path,
                                predictions_path=predictions_path,
                                config_path=config_path,
                            )

                        # Load model and make predictions
                        predictor.load_model()
                        test_predictions = predictor.predict(test_features)

                        # Calculate test metrics
                        test_metrics = calculate_prediction_metrics(
                            predictions=test_predictions,
                            true_values=test_labels,
                            model_name=f"{model_name}_test",
                        )

                        # Add test_ prefix to metrics
                        test_metrics = {
                            f"test_{k}"
                            if not k.startswith("test_") and k != "model_name"
                            else k: v
                            for k, v in test_metrics.items()
                        }

                        logger.info(
                            f"  Test RMSE: {test_metrics.get('test_rmse', 'N/A'):.4f}"
                        )

            except Exception as e:
                logger.error(f"Failed to calculate test metrics for {model_name}: {e}")
                test_metrics = {"test_error": str(e)}

            # 4. Standardize metric names and combine all metrics
            standardized_training_metrics = {}
            for key, value in training_metrics.items():
                # Standardize validation metric names
                if key == "val_loss":
                    standardized_training_metrics["val_rmse"] = value
                elif key == "train_loss":
                    standardized_training_metrics["train_rmse"] = value
                else:
                    standardized_training_metrics[key] = value

            combined_metrics = {
                "model_name": model_name,
                "model_type": model_type,
                "representation": representation,
                "is_finetuned": use_finetuned,
                **standardized_training_metrics,
                **test_metrics,
            }

            # 5. Standardize metric names for consistency
            # For end2end models, train_loss and val_loss are actually RMSE values
            # For LGBM models, val_loss is also RMSE
            if model_type == "end2end":
                if (
                    "train_loss" in combined_metrics
                    and "train_rmse" not in combined_metrics
                ):
                    combined_metrics["train_rmse"] = combined_metrics["train_loss"]
                if (
                    "val_loss" in combined_metrics
                    and "val_rmse" not in combined_metrics
                ):
                    combined_metrics["val_rmse"] = combined_metrics["val_loss"]
            elif model_type == "lgbm":
                # For LGBM, val_loss is RMSE but train_rmse is calculated separately
                if (
                    "val_loss" in combined_metrics
                    and "val_rmse" not in combined_metrics
                ):
                    combined_metrics["val_rmse"] = combined_metrics["val_loss"]

            all_metrics.append(combined_metrics)

            # Log summary for this model
            train_rmse = (
                training_metrics.get("train_rmse")
                or training_metrics.get("training_rmse")
                or training_metrics.get("train_loss")
                or "N/A"
            )
            val_rmse = (
                training_metrics.get("val_rmse")
                or training_metrics.get("validation_rmse")
                or training_metrics.get("val_loss")
                or "N/A"
            )
            test_rmse = test_metrics.get("test_rmse", "N/A")

            logger.info(f"  Train RMSE: {train_rmse}")
            logger.info(f"  Val RMSE: {val_rmse}")
            logger.info(f"  Test RMSE: {test_rmse}")

        except Exception as e:
            logger.error(
                f"Failed to process {model_type}_{representation}{model_suffix}: {e}"
            )
            # Add error entry
            all_metrics.append(
                {
                    "model_name": f"{model_type}_{representation}{model_suffix}",
                    "model_type": model_type,
                    "representation": representation,
                    "is_finetuned": use_finetuned,
                    "error": str(e),
                }
            )

    if not all_metrics:
        logger.error("No model metrics could be collected")
        return pd.DataFrame()

    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Sort by test RMSE (best first), handling missing values
    if "test_rmse" in results_df.columns:
        results_df["test_rmse_sort"] = pd.to_numeric(
            results_df["test_rmse"], errors="coerce"
        )
        # Sort with NaN values last (compatible with older pandas versions)
        results_df = results_df.sort_values("test_rmse_sort", ascending=True)
        results_df = results_df.drop("test_rmse_sort", axis=1)

    # Add ranking columns
    if "test_rmse" in results_df.columns:
        valid_test_rmse = pd.to_numeric(results_df["test_rmse"], errors="coerce")
        results_df["test_rmse_rank"] = valid_test_rmse.rank(method="min")

    # Reorder columns for better readability
    base_columns = ["model_name", "model_type", "representation", "is_finetuned"]

    # Test metrics columns
    test_columns = [col for col in results_df.columns if col.startswith("test_")]
    test_columns = sorted(
        test_columns, key=lambda x: ("rmse" in x, "mae" in x, "r2" in x)
    )

    # Training/validation metrics columns
    train_val_columns = [
        col
        for col in results_df.columns
        if any(prefix in col.lower() for prefix in ["train", "val", "validation"])
        and not col.startswith("test_")
    ]

    # Other columns
    other_columns = [
        col
        for col in results_df.columns
        if col not in base_columns + test_columns + train_val_columns
    ]

    # Build final column order
    column_order = [
        *base_columns,
        "test_rmse_rank",
        *test_columns,
        *train_val_columns,
        *other_columns,
    ]
    column_order = [col for col in column_order if col in results_df.columns]

    results_df = results_df[column_order]

    # Save results if requested
    if save_results:
        if output_dir is None:
            output_dir = MODELS_DIR / data_source
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comprehensive metrics
        comprehensive_path = output_dir / "comprehensive_model_metrics.csv"
        results_df.to_csv(comprehensive_path, index=False)
        logger.info(f"Comprehensive metrics saved to: {comprehensive_path}")

        # Save summary (key metrics only)
        # Create a combined model identifier that differentiates finetuned models
        results_df_copy = results_df.copy()
        results_df_copy["model_type"] = results_df_copy.apply(
            lambda row: f"{row['model_type']}_finetuned"
            if row.get("is_finetuned", False)
            else row["model_type"],
            axis=1,
        )

        summary_columns = ["model_type", "representation"]

        # Add train, val, and test metrics (RMSE, MAE, R²) if available
        metric_types = ["train", "val", "test"]
        metric_names = ["rmse", "mae", "r2_score"]

        for metric_type in metric_types:
            for metric_name in metric_names:
                col_name = f"{metric_type}_{metric_name}"
                if col_name in results_df_copy.columns:
                    summary_columns.append(col_name)

        summary_columns = [
            col for col in summary_columns if col in results_df_copy.columns
        ]
        summary_df = results_df_copy[summary_columns]

        summary_path = output_dir / "comprehensive_model_metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary metrics saved to: {summary_path}")

    # Log final summary
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE METRICS SUMMARY")
    logger.info("=" * 80)

    if "test_rmse" in results_df.columns:
        valid_models = results_df[
            pd.to_numeric(results_df["test_rmse"], errors="coerce").notna()
        ]
        if len(valid_models) > 0:
            best_model = valid_models.iloc[0]
            logger.info(f"Best model by test RMSE: {best_model['model_name']}")
            logger.info(f"  Test RMSE: {best_model['test_rmse']:.4f}")
            if "test_mae" in best_model:
                logger.info(f"  Test MAE: {best_model['test_mae']:.4f}")
            if "test_r2_score" in best_model:
                logger.info(f"  Test R²: {best_model['test_r2_score']:.4f}")

    logger.info(f"Total models evaluated: {len(results_df)}")
    logger.info(
        f"Models with valid test metrics: {len(results_df[pd.to_numeric(results_df.get('test_rmse', []), errors='coerce').notna()])}"
    )

    return results_df


def evaluate_loocv_models_on_base_validation(
    save_results: bool = True,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Evaluate all LOOCV LightGBM models on the base_cdpfas_val internal validation dataset.

    Args:
        save_results: Whether to save results to CSV file
        output_dir: Directory to save results (default: models/external_validation)

    Returns:
        DataFrame containing metrics for all LOOCV models on the base validation set
    """
    import json

    from cd_host_guest.config import EXTERNAL_DATA_DIR, MODELS_DIR

    # Define base paths
    loocv_models_dir = MODELS_DIR / "external_validation"
    base_val_dir = EXTERNAL_DATA_DIR / "validation" / "base_cdpfas_val"
    canonical_csv = base_val_dir / "base_cdpfas_val_canonical.csv"

    # Check if validation data exists
    if not canonical_csv.exists():
        logger.error(f"Validation data not found at {canonical_csv}")
        return pd.DataFrame()

    # Load true values from canonical CSV
    try:
        df_true = pd.read_csv(canonical_csv)
        if "delG" not in df_true.columns:
            logger.error("'delG' column not found in canonical validation data")
            return pd.DataFrame()
        true_values = df_true["delG"].astype(float).values
        logger.info(f"Loaded {len(true_values)} true values from {canonical_csv}")
    except Exception as e:
        logger.error(f"Failed to load true values from {canonical_csv}: {e}")
        return pd.DataFrame()

    # Find all LOOCV model directories (exclude CSV files)
    model_representations = []
    for item in loocv_models_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            model_representations.append(item.name)

    if not model_representations:
        logger.error(f"No LOOCV model directories found in {loocv_models_dir}")
        return pd.DataFrame()

    logger.info(f"Found LOOCV models for representations: {model_representations}")

    all_metrics = []

    for representation in model_representations:
        try:
            logger.info(f"Evaluating LOOCV model: {representation}")

            # Define file paths for this representation
            repr_dir = loocv_models_dir / representation
            model_file = (
                repr_dir / f"loocv_model_on_base_cdpfas_val_{representation}.pkl"
            )
            predictions_file = (
                repr_dir / f"loocv_predictions_on_base_cdpfas_val_{representation}.csv"
            )
            metrics_file = (
                repr_dir / f"loocv_metrics_on_base_cdpfas_val_{representation}.json"
            )

            # Check if all required files exist
            missing_files = []
            if not model_file.exists():
                missing_files.append(f"model: {model_file}")
            if not predictions_file.exists():
                missing_files.append(f"predictions: {predictions_file}")
            if not metrics_file.exists():
                missing_files.append(f"metrics: {metrics_file}")

            if missing_files:
                logger.warning(
                    f"Missing files for {representation}: {', '.join(missing_files)}"
                )
                all_metrics.append(
                    {
                        "representation": representation,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "mse": None,
                        "n_samples": len(true_values),
                        "success": False,
                        "error": f"Missing files: {', '.join(missing_files)}",
                        "predictions_file": str(predictions_file),
                        "metrics_file": str(metrics_file),
                        "model_file": str(model_file),
                    }
                )
                continue

            # Load existing metrics from JSON file
            try:
                with open(metrics_file) as f:
                    existing_metrics = json.load(f)
                logger.info(
                    f"  Loaded existing metrics: RMSE={existing_metrics.get('rmse', 'N/A'):.4f}"
                )
            except Exception as e:
                logger.warning(
                    f"  Failed to load existing metrics from {metrics_file}: {e}"
                )
                existing_metrics = {}

            # Load predictions from CSV file
            try:
                pred_df = pd.read_csv(predictions_file)
                if "DeltaG" not in pred_df.columns:
                    logger.error(
                        f"  'DeltaG' column not found in predictions file for {representation}"
                    )
                    continue
                predictions = pred_df["DeltaG"].astype(float).values
                logger.info(f"  Loaded {len(predictions)} predictions")
            except Exception as e:
                logger.error(
                    f"  Failed to load predictions from {predictions_file}: {e}"
                )
                all_metrics.append(
                    {
                        "representation": representation,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "mse": None,
                        "n_samples": len(true_values),
                        "success": False,
                        "error": f"Failed to load predictions: {e}",
                        "predictions_file": str(predictions_file),
                        "metrics_file": str(metrics_file),
                        "model_file": str(model_file),
                    }
                )
                continue

            # Verify data alignment
            if len(predictions) != len(true_values):
                logger.error(
                    f"  Mismatch in data length: {len(predictions)} predictions vs {len(true_values)} true values"
                )
                continue

            # Calculate metrics
            try:
                metrics = calculate_prediction_metrics(
                    predictions=predictions,
                    true_values=true_values,
                    model_name=f"LOOCV_LGBM_{representation}",
                )

                # Add additional information
                metrics.update(
                    {
                        "representation": representation,
                        "success": True,
                        "error": None,
                        "predictions_file": str(predictions_file),
                        "metrics_file": str(metrics_file),
                        "model_file": str(model_file),
                    }
                )

                # Add model information if available
                try:
                    model = joblib.load(model_file)
                    if hasattr(model, "num_feature"):
                        metrics["n_features"] = model.num_feature()
                    # Get training information from the model object if available
                    if hasattr(model, "train_set") and model.train_set is not None:
                        metrics["n_train_samples"] = model.train_set.num_data()
                    else:
                        # Default assumption for LOOCV: using all external validation data except the test sample
                        metrics["n_train_samples"] = 63  # Based on the existing CSV
                    metrics["n_eval_samples"] = len(true_values)
                except Exception as e:
                    logger.warning(
                        f"  Could not load model info for {representation}: {e}"
                    )
                    metrics.update(
                        {
                            "n_features": None,
                            "n_train_samples": None,
                            "n_eval_samples": len(true_values),
                        }
                    )

                all_metrics.append(metrics)

                logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  R²: {metrics['r2_score']:.4f}")

            except Exception as e:
                logger.error(f"  Failed to calculate metrics for {representation}: {e}")
                all_metrics.append(
                    {
                        "representation": representation,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "mse": None,
                        "n_samples": len(true_values),
                        "success": False,
                        "error": f"Failed to calculate metrics: {e}",
                        "predictions_file": str(predictions_file),
                        "metrics_file": str(metrics_file),
                        "model_file": str(model_file),
                    }
                )

        except Exception as e:
            logger.error(f"Failed to process {representation}: {e}")
            all_metrics.append(
                {
                    "representation": representation,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "mse": None,
                    "n_samples": len(true_values),
                    "success": False,
                    "error": str(e),
                    "predictions_file": "N/A",
                    "metrics_file": "N/A",
                    "model_file": "N/A",
                }
            )

    if not all_metrics:
        logger.error("No LOOCV models could be evaluated")
        return pd.DataFrame()

    # Create results DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Filter to only successful evaluations for ranking
    successful_results = results_df[results_df["success"]].copy()

    if len(successful_results) > 0:
        # Sort by RMSE (best first)
        successful_results = successful_results.sort_values("rmse", ascending=True)
        successful_results["rmse_rank"] = range(1, len(successful_results) + 1)

        # Merge ranks back to the full dataframe
        results_df = results_df.merge(
            successful_results[["representation", "rmse_rank"]],
            on="representation",
            how="left",
        )
    else:
        results_df["rmse_rank"] = None

    # Reorder columns for better readability
    base_columns = [
        "representation",
        "rmse",
        "mae",
        "r2_score",
        "mse",
        "n_features",
        "n_train_samples",
        "n_eval_samples",
        "success",
        "predictions_file",
        "metrics_file",
        "model_file",
        "rmse_rank",
    ]

    # Only include columns that exist
    column_order = [col for col in base_columns if col in results_df.columns]

    # Add any remaining columns not in base_columns
    remaining_columns = [col for col in results_df.columns if col not in column_order]
    column_order.extend(remaining_columns)

    results_df = results_df[column_order]

    # Save results if requested
    if save_results:
        output_dir = loocv_models_dir if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_path = output_dir / "loocv_base_cdpfas_val_evaluation_detailed.csv"
        results_df.to_csv(detailed_path, index=False)
        logger.info(f"Detailed LOOCV evaluation results saved to {detailed_path}")

        # Save summary results (successful models only)
        if len(successful_results) > 0:
            summary_cols = [
                "rmse_rank",
                "representation",
                "rmse",
                "mae",
                "r2_score",
                "n_features",
                "n_train_samples",
                "n_eval_samples",
            ]
            summary_cols = [
                col for col in summary_cols if col in successful_results.columns
            ]
            summary_df = successful_results[summary_cols]

            summary_path = output_dir / "loocv_base_cdpfas_val_evaluation_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary LOOCV evaluation results saved to {summary_path}")

        # Update the main summary file with new evaluation
        summary_path = output_dir / "loocv_base_cdpfas_val_summary.csv"
        results_df.to_csv(summary_path, index=False)
        logger.info(f"Updated main LOOCV summary saved to {summary_path}")

    # Log summary to console
    logger.info("\n" + "=" * 80)
    logger.info("LOOCV MODELS EVALUATION ON BASE_CDPFAS_VAL")
    logger.info("=" * 80)

    successful_models = len(successful_results) if len(successful_results) > 0 else 0
    total_models = len(results_df)

    if successful_models > 0:
        best_model = successful_results.iloc[0]
        logger.info(f"Best LOOCV model: {best_model['representation']}")
        logger.info(f"  RMSE: {best_model['rmse']:.4f}")
        logger.info(f"  MAE: {best_model['mae']:.4f}")
        logger.info(f"  R²: {best_model['r2_score']:.4f}")

        logger.info("\nTop 3 LOOCV models:")
        for i, (_, row) in enumerate(successful_results.head(3).iterrows()):
            logger.info(
                f"  {i+1}. {row['representation']}: RMSE={row['rmse']:.4f}, R²={row['r2_score']:.4f}"
            )

    logger.info("=" * 80)
    logger.info(f"Total LOOCV models evaluated: {total_models}")
    logger.info(f"Successful evaluations: {successful_models}")
    logger.info(f"Failed evaluations: {total_models - successful_models}")

    return results_df


@app.command("loocv-evaluate-all-on-base")
def loocv_evaluate_all_on_base_command(
    save_results: bool = typer.Option(True, help="Save results to CSV file"),
    output_dir: str | None = typer.Option(None, help="Directory to save results"),
) -> None:
    """Evaluate all LOOCV LightGBM models on the base_cdpfas_val internal validation dataset."""
    output_path = Path(output_dir) if output_dir else None
    results_df = evaluate_loocv_models_on_base_validation(
        save_results=save_results, output_dir=output_path
    )

    if results_df.empty:
        typer.echo("No LOOCV models could be evaluated.")
        raise typer.Exit(1)

    best_model = results_df.iloc[0]["representation"]
    best_rmse = results_df.iloc[0]["rmse"]
    typer.echo(
        f"Successfully evaluated {len(results_df)} LOOCV models on base_cdpfas_val. "
        f"Best model: {best_model} (RMSE: {best_rmse:.4f})"
    )


@app.command("predict-external-validation")
def predict_external_validation_cli(
    val_set: str = typer.Argument(
        ..., help="Validation set name (e.g., 'cd_val', 'pfas_val', 'base_cdpfas_val')"
    ),
    model_type: str = typer.Option(
        None, help="Specific model type (lgbm, fnn, end2end) or None for all"
    ),
    representation: str = typer.Option(
        None, help="Specific representation or None for all"
    ),
    overwrite: bool = typer.Option(False, help="Overwrite existing prediction files"),
) -> None:
    """Generate predictions for all available models on an external validation dataset."""

    results = generate_external_validation_predictions(
        val_set=val_set,
        model_type=model_type,
        representation=representation,
        overwrite=overwrite,
    )

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    typer.echo(f"Prediction generation complete: {successful}/{total} successful")

    if successful < total:
        failed = [r for r in results if not r["success"]]
        typer.echo("\nFailed predictions:")
        for f in failed:
            typer.echo(f"  - {f['model_name']}: {f['error']}")


def generate_external_validation_predictions(
    val_set: str,
    model_type: str = None,
    representation: str = None,
    overwrite: bool = False,
) -> list[dict]:
    """
    Generate predictions for all available models on an external validation dataset.
    Does not require true values - only generates predictions.

    Args:
        val_set: Name of the validation set (e.g., 'cd_val', 'pfas_val', 'base_cdpfas_val')
        model_type: Specific model type to run (None for all)
        representation: Specific representation to run (None for all)
        overwrite: Whether to overwrite existing prediction files

    Returns:
        List of dictionaries with prediction results
    """
    from cd_host_guest.config import EXTERNAL_DATA_DIR, MODELS_DIR

    # Define model configurations for external validation
    all_model_configs = [
        # LGBM models
        ("lgbm", "ecfp", False, "lgbm_ecfp_predictions.csv"),
        ("lgbm", "ecfp_plus", False, "lgbm_ecfp_plus_predictions.csv"),
        ("lgbm", "grover", False, "lgbm_grover_predictions.csv"),
        ("lgbm", "unimol", False, "lgbm_unimol_predictions.csv"),
        # FNN models
        ("fnn", "ecfp", False, "fnn_ecfp_predictions.csv"),
        ("fnn", "ecfp_plus", False, "fnn_ecfp_plus_predictions.csv"),
        ("fnn", "grover", False, "fnn_grover_predictions.csv"),
        ("fnn", "unimol", False, "fnn_unimol_predictions.csv"),
        # End2End models
        ("end2end", "string", False, "end2end_string_predictions.csv"),
        ("end2end", "string", True, "end2end_string_finetuned_final_predictions.csv"),
    ]

    # Filter configurations if specific model_type or representation requested
    model_configs = []
    for config in all_model_configs:
        cfg_model_type, cfg_representation, cfg_is_finetuned, cfg_filename = config

        # Filter by model_type if specified
        if model_type is not None and cfg_model_type != model_type:
            continue

        # Filter by representation if specified
        if representation is not None and cfg_representation != representation:
            continue

        model_configs.append(config)

    if not model_configs:
        logger.error("No model configurations match the specified criteria")
        return []

    # Validate validation set exists
    base_dir = EXTERNAL_DATA_DIR / "validation" / val_set
    canonical_csv = base_dir / f"{val_set}_canonical.csv"

    if not canonical_csv.exists():
        logger.error(
            f"Validation set '{val_set}' not found. Expected canonical CSV at {canonical_csv}"
        )
        return []

    logger.info(f"Generating predictions for validation set: {val_set}")
    logger.info(f"Found {len(model_configs)} model configurations to run")

    results = []

    for (
        cfg_model_type,
        cfg_representation,
        is_finetuned,
        pred_filename,
    ) in model_configs:
        try:
            # Build model name for logging
            if cfg_model_type == "end2end":
                model_name = f"End2End{' (Fine-tuned)' if is_finetuned else ''}"
            else:
                model_name = f"{cfg_model_type.upper()} ({cfg_representation})"

            logger.info(f"Generating predictions for {model_name}")

            # Determine paths
            if cfg_model_type == "end2end":
                # For end2end models, use string representation paths
                model_dir = MODELS_DIR / "OpenCycloDB" / "string"
                pred_dir = base_dir / "string"

                # Determine model and features files
                # End2End models need SMILES strings, not embeddings, so use canonical CSV
                model_basename = (
                    "end2end_string_finetuned_final"
                    if is_finetuned
                    else "end2end_string"
                )
                features_file = base_dir / f"{val_set}_canonical.csv"

                model_file = model_dir / f"{model_basename}.pth"
                config_file = model_dir / f"{model_basename}_config.json"

            else:
                # For other models (lgbm, fnn)
                model_dir = MODELS_DIR / "OpenCycloDB" / cfg_representation
                pred_dir = base_dir / cfg_representation

                # Model file extension depends on model type
                if cfg_model_type == "fnn":
                    model_file = (
                        model_dir / f"{cfg_model_type}_{cfg_representation}.pth"
                    )
                else:
                    model_file = (
                        model_dir / f"{cfg_model_type}_{cfg_representation}.pkl"
                    )
                features_file = (
                    base_dir
                    / cfg_representation
                    / f"{val_set}_canonical_{cfg_representation}.csv"
                )

                # For FNN models, also need scaler files
                if cfg_model_type == "fnn":
                    scaler_file = model_dir / f"fnn_{cfg_representation}_scaler.pkl"
                    label_scaler_file = (
                        model_dir / f"fnn_{cfg_representation}_label_scaler.pkl"
                    )
                    config_file = model_dir / f"fnn_{cfg_representation}_config.json"
                else:
                    scaler_file = None
                    label_scaler_file = None
                    config_file = None

            # Check if prediction file already exists
            pred_file = pred_dir / pred_filename
            if pred_file.exists() and not overwrite:
                logger.info(
                    f"  Predictions already exist: {pred_file} (use --overwrite to regenerate)"
                )
                results.append(
                    {
                        "model_name": model_name,
                        "model_type": cfg_model_type,
                        "representation": cfg_representation,
                        "is_finetuned": is_finetuned,
                        "prediction_file": str(pred_file),
                        "success": True,
                        "error": None,
                        "skipped": True,
                    }
                )
                continue

            # Check if required files exist
            missing_files = []
            if not model_file.exists():
                missing_files.append(f"model: {model_file}")
            if not features_file.exists():
                missing_files.append(f"features: {features_file}")
            if cfg_model_type == "fnn" and scaler_file and not scaler_file.exists():
                missing_files.append(f"scaler: {scaler_file}")

            if missing_files:
                error_msg = f"Missing files: {', '.join(missing_files)}"
                logger.warning(f"  {error_msg}")
                results.append(
                    {
                        "model_name": model_name,
                        "model_type": cfg_model_type,
                        "representation": cfg_representation,
                        "is_finetuned": is_finetuned,
                        "prediction_file": str(pred_file),
                        "success": False,
                        "error": error_msg,
                        "skipped": False,
                    }
                )
                continue

            # Create prediction directory if it doesn't exist
            pred_dir.mkdir(parents=True, exist_ok=True)

            # Create appropriate predictor
            if cfg_model_type == "lgbm":
                predictor = LightGBMPredictor(
                    features_path=features_file,
                    model_path=model_file,
                    predictions_path=pred_file,
                )
            elif cfg_model_type == "fnn":
                predictor = FNNPredictor(
                    features_path=features_file,
                    model_path=model_file,
                    predictions_path=pred_file,
                    scaler_path=scaler_file,
                    label_scaler_path=label_scaler_file,
                    config_path=config_file,
                )
            elif cfg_model_type == "end2end":
                predictor = End2EndPredictor(
                    features_path=features_file,
                    model_path=model_file,
                    predictions_path=pred_file,
                    config_path=config_file,
                )
            else:
                msg = f"Unknown model type: {cfg_model_type}"
                raise ValueError(msg)

            # Generate predictions
            predictor.check_directories()
            predictor.load_model()

            # Load features
            features_df = pd.read_csv(features_file)
            logger.info(f"  Loaded {len(features_df)} samples from {features_file}")

            # Generate predictions
            predictions = predictor.predict(features_df)
            logger.info(f"  Generated {len(predictions)} predictions")

            # Save predictions
            predictor.save_predictions(predictions)
            logger.info(f"  Saved predictions to {pred_file}")

            results.append(
                {
                    "model_name": model_name,
                    "model_type": cfg_model_type,
                    "representation": cfg_representation,
                    "is_finetuned": is_finetuned,
                    "prediction_file": str(pred_file),
                    "success": True,
                    "error": None,
                    "skipped": False,
                }
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"  Failed to generate predictions for {model_name}: {error_msg}"
            )
            results.append(
                {
                    "model_name": model_name
                    if "model_name" in locals()
                    else f"{cfg_model_type}_{cfg_representation}",
                    "model_type": cfg_model_type,
                    "representation": cfg_representation,
                    "is_finetuned": is_finetuned,
                    "prediction_file": str(pred_file)
                    if "pred_file" in locals()
                    else "unknown",
                    "success": False,
                    "error": error_msg,
                    "skipped": False,
                }
            )

    # Summary
    successful = sum(1 for r in results if r["success"])
    skipped = sum(1 for r in results if r.get("skipped", False))
    failed = len(results) - successful

    logger.info(f"\nPrediction generation summary for {val_set}:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Skipped (already exist): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(results)}")

    return results


@app.command("evaluate-external-validation")
def evaluate_external_validation_cli(
    val_set: str = typer.Argument(
        ..., help="Validation set name (e.g., 'cd_val', 'pfas_val')"
    ),
    save_results: bool = typer.Option(True, help="Save results to CSV files"),
    output_dir: str | None = typer.Option(None, help="Directory to save results"),
) -> None:
    """Evaluate predictions from all available models on an external validation dataset."""
    output_path = Path(output_dir) if output_dir else None
    results_df = evaluate_external_validation_predictions(
        val_set=val_set, save_results=save_results, output_dir=output_path
    )

    if results_df.empty:
        typer.echo(
            f"No models could be evaluated on validation set '{val_set}'.", err=True
        )
        raise typer.Exit(1)

    best_model = results_df.iloc[0]["model_name"]
    best_rmse = results_df.iloc[0]["rmse"]
    typer.echo(
        f"Successfully evaluated {len(results_df)} models on '{val_set}'. Best model: {best_model} (RMSE: {best_rmse:.4f})"
    )


@app.command("evaluate-all-external-validation")
def evaluate_all_external_validation_cli(
    save_results: bool = typer.Option(True, help="Save results to CSV files"),
    output_dir: str | None = typer.Option(None, help="Directory to save results"),
) -> None:
    """Evaluate predictions from all available models on all external validation datasets."""
    from cd_host_guest.config import EXTERNAL_DATA_DIR

    # Include all known external validation sets (extendable)
    validation_sets = ["cd_val", "pfas_val", "base_cdpfas_val"]
    all_results = []

    for val_set in validation_sets:
        # Check if validation set exists
        val_dir = EXTERNAL_DATA_DIR / "validation" / val_set
        if not val_dir.exists():
            logger.warning(f"Validation set directory not found: {val_dir}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating models on {val_set.upper()}")
        logger.info("=" * 60)

        output_path = Path(output_dir) if output_dir else None
        results_df = evaluate_external_validation_predictions(
            val_set=val_set, save_results=save_results, output_dir=output_path
        )

        if not results_df.empty:
            all_results.append(results_df)
            best_model = results_df.iloc[0]["model_name"]
            best_rmse = results_df.iloc[0]["rmse"]
            logger.info(
                f"Best model on {val_set}: {best_model} (RMSE: {best_rmse:.4f})"
            )
        else:
            logger.warning(f"No models could be evaluated on {val_set}")

    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        if save_results:
            if output_dir is None:
                output_dir = EXTERNAL_DATA_DIR / "validation"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Save combined results
            combined_path = output_dir / "all_external_validation_results.csv"
            combined_results.to_csv(combined_path, index=False)
            logger.info(f"\nCombined validation results saved to {combined_path}")

        typer.echo(
            f"\nSuccessfully evaluated models on {len(validation_sets)} validation sets."
        )
        typer.echo(f"Total evaluations: {len(combined_results)}")
    else:
        typer.echo("No validation sets could be evaluated.", err=True)
        raise typer.Exit(1)


@app.command()
def main(
    model_type: str = typer.Argument(
        ..., help="Model type to use (lightgbm, fnn, end2end)"
    ),
    representation: str = typer.Argument(
        ..., help="Feature representation (e.g. ecfp, original_enriched, string, etc.)"
    ),
    wandb_log: bool = False,
    use_finetuned: bool = typer.Option(
        False, help="Use fine-tuned model (applies to end2end models only)"
    ),
):
    """
    Main function to load model and perform predictions.
    Determines all file paths based on model_type and representation.
    """
    run_prediction(model_type, representation, wandb_log, use_finetuned)


@app.command("predict")
def run_prediction(
    model_type: str,
    representation: str,
    wandb_log: bool = False,
    use_finetuned: bool = False,
) -> None:
    """
    Core prediction logic for a single model_type and representation.

    Args:
        model_type: Type of model (lgbm, fnn, end2end)
        representation: Feature representation
        wandb_log: Whether to log to wandb
        use_finetuned: Whether to use fine-tuned model (applies to end2end models only)
    """
    data_source = "OpenCycloDB"

    # Handle special case for end2end model with string representation
    if model_type == "end2end":
        model_dir = MODELS_DIR / data_source / "string"
        processed_dir = PROCESSED_DATA_DIR / data_source / "string"
    else:
        model_dir = MODELS_DIR / data_source / representation
        processed_dir = PROCESSED_DATA_DIR / data_source / representation

    features_path = processed_dir / "test_features.csv"
    # labels_path = processed_dir / "test_labels.csv"
    predictions_dir = model_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "lgbm" or model_type == "lightgbm":
        model_path = model_dir / f"lgbm_{representation}.pkl"
        predictions_path = predictions_dir / f"lgbm_{representation}_predictions.csv"
        scaler_path = None
        label_scaler_path = None
        config_path = None
        predictor = LightGBMPredictor(
            features_path, model_path, predictions_path, wandb_log
        )
    elif model_type == "fnn":
        model_path = model_dir / f"fnn_{representation}.pth"
        predictions_path = predictions_dir / f"fnn_{representation}_predictions.csv"
        scaler_path = model_dir / f"fnn_{representation}_scaler.pkl"
        label_scaler_path = model_dir / f"fnn_{representation}_label_scaler.pkl"
        config_path = model_dir / f"fnn_{representation}_config.json"
        predictor = FNNPredictor(
            features_path,
            model_path,
            predictions_path,
            wandb_log,
            scaler_path,
            label_scaler_path,
            config_path,
        )
    elif model_type == "end2end":
        # Handle end2end model with support for fine-tuned variants
        if use_finetuned:
            model_basename = "end2end_string_finetuned_final"
        else:
            model_basename = "end2end_string"

        model_path = model_dir / f"{model_basename}.pth"
        predictions_path = predictions_dir / f"{model_basename}_predictions.csv"
        scaler_path = model_dir / f"{model_basename}_scaler.pkl"
        label_scaler_path = model_dir / f"{model_basename}_label_scaler.pkl"
        config_path = model_dir / f"{model_basename}_config.json"

        # Fall back to base model config if fine-tuned config doesn't exist
        if not config_path.exists() and use_finetuned:
            base_config_path = model_dir / "end2end_string_config.json"
            if base_config_path.exists():
                config_path = base_config_path
                logger.info(f"Using base model config: {base_config_path}")

        predictor = End2EndPredictor(
            features_path,
            model_path,
            predictions_path,
            wandb_log,
            scaler_path,
            label_scaler_path,
            config_path,
        )
    else:
        msg = f"Unsupported model type: {model_type}"
        raise ValueError(msg)

    predictor.check_directories()
    logger.info("Loading data...")
    features = pd.read_csv(features_path)
    logger.info("Loading model...")
    predictor.load_model()
    logger.info("Making predictions...")
    predictions = predictor.predict(features)
    predictor.save_predictions(predictions)
    logger.info("Predictions saved successfully")


@app.command("predict-finetuned")
def predict_finetuned(
    wandb_log: bool = False,
):
    """
    Run predictions using the fine-tuned end2end model.

    Args:
        wandb_log: Whether to log to wandb
    """
    logger.info("Running predictions with fine-tuned end2end model")
    run_prediction("end2end", "string", wandb_log, use_finetuned=True)


@app.command("predict-all-models")
def predict_all_models(
    wandb_log: bool = False,
):
    """
    Run predictions for all model types and feature representations.
    Logs progress and errors for each combination.
    """
    # Define supported model types and representations
    model_types: list[str] = ["lgbm", "fnn", "end2end"]
    representations: list[str] = [
        "ecfp",
        "original_enriched",
        "ecfp_plus",
        "grover",
        "unimol",
        "string",  # Add string representation for end2end
    ]
    errors: list[str] = []
    successes: list[str] = []
    for model_type in model_types:
        for representation in representations:
            # Skip invalid combinations
            if model_type == "end2end" and representation != "string":
                continue
            if model_type in ["lgbm", "fnn"] and representation == "string":
                continue

            try:
                logger.info(
                    f"Running prediction for model_type={model_type}, representation={representation}"
                )
                run_prediction(model_type, representation, wandb_log)
                successes.append(f"{model_type}:{representation}")
            except Exception as e:
                logger.error(f"Failed for {model_type}:{representation}: {e}")
                errors.append(f"{model_type}:{representation} ({e})")
    logger.info(
        f"Prediction completed. Successes: {len(successes)}, Failures: {len(errors)}"
    )
    if errors:
        logger.warning("Failures:")
        for err in errors:
            logger.warning(f" - {err}")
    else:
        logger.info("All predictions completed successfully.")


@app.command("get-wandb-model")
def get_wandb_model(model_type: str, representation: str) -> None:
    """
    Retrieve the set of hyperparameters and the trained model artifact from wandb
    for a given model type and data representation, and save the model to the models folder.

    Args:
        model_type (str): The type of model (e.g., 'lightgbm', 'fnn').
        representation (str): The data representation (e.g., 'enriched', 'ecfp').
    """
    # Validate input types
    if not isinstance(model_type, str):
        msg = "model_type must be a string."
        raise TypeError(msg)
    if not isinstance(representation, str):
        msg = "representation must be a string."
        raise TypeError(msg)

    # Set up wandb API
    try:
        api = wandb.Api()
    except Exception as e:
        logger.error(f"Failed to initialize wandb API: {e}")
        raise

    # Define project and artifact name conventions
    project_name = f"{model_type}_{representation}"

    # Try to get the artifact
    try:
        logger.info(
            f"Fetching artifact {project_name} from wandb project {project_name}"
        )
        artifact = api.runs(f"{project_name}")
    except Exception as e:
        logger.error(f"Failed to fetch artifact: {e}")
        msg = f"Project {project_name} not found in wandb."
        raise FileNotFoundError(msg) from None

    # First try to find runs with val_rmse, then fall back to val_loss
    finished_runs = [run for run in artifact if run.state == "finished"]

    # Check for val_rmse first
    val_rmse_runs = [run for run in finished_runs if "val_rmse" in run.summary]
    if val_rmse_runs:
        sorted_runs = sorted(val_rmse_runs, key=lambda r: r.summary["val_rmse"])
        metric_used = "val_rmse"
    else:
        # Fall back to val_loss if val_rmse is not available
        val_loss_runs = [run for run in finished_runs if "val_loss" in run.summary]
        if val_loss_runs:
            sorted_runs = sorted(val_loss_runs, key=lambda r: r.summary["val_loss"])
            metric_used = "val_loss"
        else:
            sorted_runs = []
            metric_used = None

    best_run = sorted_runs[0] if sorted_runs else None
    best_hyperparams = best_run.config if best_run else None
    if best_hyperparams is None:
        logger.error(
            "No finished runs with 'val_rmse' or 'val_loss' found in the artifact."
        )
        msg = f"No valid runs found for project {project_name}."
        raise ValueError(msg)

    logger.info(
        f"Best run selected using metric: {metric_used} = {best_run.summary[metric_used]:.4f}"
    )

    # Save the config file to the models directory
    if model_type == "end2end":
        config_path = (
            MODELS_DIR / "OpenCycloDB" / "string" / f"{model_type}_string_config.json"
        )
    else:
        config_path = (
            MODELS_DIR
            / "OpenCycloDB"
            / representation
            / f"{model_type}_{representation}_config.json"
        )
    try:
        # Ensure the parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(best_hyperparams, f, indent=4)
        logger.info(f"Saved hyperparameters to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save hyperparameters: {e}")
        raise


@app.command("get-all-wandb-configs")
def get_all_wandb_configs() -> None:
    """
    Iterate through all projects in wandb and save the best hyperparameter config file of each
    using the get_wandb_model function. List any projects that failed to be found or saved.
    """
    failed_projects: list[str] = []

    # Set up wandb API
    try:
        api = wandb.Api()
    except Exception as e:
        logger.error(f"Failed to initialize wandb API: {e}")
        return

    # Get all projects for the current entity
    try:
        entity = api.default_entity
        projects = api.projects(entity=entity)
    except Exception as e:
        logger.error(f"Failed to fetch projects from wandb: {e}")
        return

    # Iterate through all projects
    for project in projects:
        project_name = project.name
        # Attempt to parse model_type and representation from project name
        if "_" not in project_name:
            logger.warning(
                f"Skipping project '{project_name}' (invalid naming convention)"
            )
            failed_projects.append(project_name)
            continue
        parts = project_name.split("_", 1)
        model_type, representation = parts[0], parts[1]
        try:
            get_wandb_model(model_type, representation)
        except Exception as e:
            logger.error(f"Failed to process project '{project_name}': {e}")
            failed_projects.append(project_name)

    if failed_projects:
        logger.warning("The following projects failed to be processed:")
        for proj in failed_projects:
            logger.warning(f" - {proj}")
    else:
        logger.info("All wandb projects processed successfully.")


@app.command("compare-models")
def compare_models_command(
    data_source: str = typer.Option("OpenCycloDB", help="Data source to use"),
    use_finetuned: bool = typer.Option(
        False, help="Include fine-tuned end2end model in comparison"
    ),
    save_results: bool = typer.Option(True, help="Save results to CSV files"),
    output_dir: str | None = typer.Option(None, help="Directory to save results"),
):
    """Compare predictions from all available models."""
    output_path = Path(output_dir) if output_dir else None
    results_df = compare_all_model_predictions(
        data_source=data_source,
        use_finetuned_end2end=use_finetuned,
        save_results=save_results,
        output_dir=output_path,
    )

    if results_df.empty:
        typer.echo("No models could be evaluated.", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"Successfully compared {len(results_df)} models. Best model (by RMSE): {results_df.iloc[0]['model_name']}"
    )


@app.command("comprehensive-metrics")
def comprehensive_metrics_command(
    data_source: str = typer.Option("OpenCycloDB", help="Data source to use"),
    save_results: bool = typer.Option(True, help="Save results to CSV files"),
    output_dir: str | None = typer.Option(None, help="Directory to save results"),
):
    """Get comprehensive training, validation, and test metrics for all models."""
    output_path = Path(output_dir) if output_dir else None

    try:
        results_df = get_comprehensive_model_metrics(
            data_source=data_source,
            save_results=save_results,
            output_dir=output_path,
        )

        if results_df.empty:
            typer.echo("No model metrics could be collected.", err=True)
            raise typer.Exit(1)

        # Display summary
        valid_models = results_df[
            pd.to_numeric(results_df.get("test_rmse", []), errors="coerce").notna()
        ]
        if len(valid_models) > 0:
            best_model = valid_models.iloc[0]
            typer.echo(
                f"✅ Successfully collected metrics for {len(results_df)} models"
            )
            typer.echo(f"🏆 Best model by test RMSE: {best_model['model_name']}")
            typer.echo(f"📊 Test RMSE: {float(best_model['test_rmse']):.4f}")
            if "test_r2_score" in best_model:
                typer.echo(f"📊 Test R²: {float(best_model['test_r2_score']):.4f}")
        else:
            typer.echo(
                f"✅ Collected metrics for {len(results_df)} models (no valid test metrics)"
            )

    except Exception as e:
        typer.echo(f"❌ Failed to collect comprehensive metrics: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()

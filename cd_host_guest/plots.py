from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import shap
import typer
from config import (
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    umap_hyperparameters,
)
from loguru import logger
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage{textgreek}\usepackage{upgreek}\usepackage{amsmath}",
    }
)
plt.rcParams["mathtext.default"] = "regular"

# matplotlib.use("pgf")
# matplotlib.rcParams.update(
#     {
#         "pgf.texsystem": "pdflatex",
#         "font.family": "serif",
#         "text.usetex": True,
#         "pgf.rcfonts": False,
#     }
# )

style = scienceplots
plt.style.use(["science", "bright"])
app = typer.Typer()


@app.command()
def create_parity_plot(
    labels_path: Path, predictions_path: Path, output_path: Path | None = None
) -> None:
    """
    Create a parity plot comparing true labels vs predictions.

    Args:
        labels_path (Path): Path to CSV file containing true labels
        predictions_path (Path): Path to CSV file containing predictions
        output_path (Path): Optional path to save the plot

    Raises:
        ValueError: If the input files contain null values or different lengths
    """
    # Load the data
    logger.info("Loading labels and predictions")
    labels = pd.read_csv(labels_path)
    predictions = pd.read_csv(predictions_path)

    # Convert to numeric and check for nulls
    labels_numeric = labels["DeltaG"].astype(float)
    predictions_numeric = predictions["DeltaG"].astype(float)

    if labels_numeric.isnull().any() or predictions_numeric.isnull().any():
        msg = "Input data contains null values."
        raise ValueError(msg)

    if len(labels_numeric) != len(predictions_numeric):
        msg = "Labels and predictions must have the same length."
        raise ValueError(msg)

    # Create the plot
    logger.info("Creating parity plot")
    plt.figure(figsize=(10, 7))

    # Plot the predictions vs actual
    plt.scatter(labels_numeric, predictions_numeric, alpha=0.5)

    # Add the ideal 1:1 line
    min_val = min(labels_numeric.min(), predictions_numeric.min())
    max_val = max(labels_numeric.max(), predictions_numeric.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal (1:1)")

    # Add axis labels for true vs predicted
    plt.xlabel("True Value (DeltaG)")
    plt.ylabel("Predicted Value (DeltaG)")
    plt.title("Parity Plot")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save if output path provided
    if output_path is not None:
        logger.info(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close()


@app.command("perform-pca-and-plot-variance")
def perform_pca_and_plot_variance(input_csv: Path) -> None:
    """
    Perform PCA on the input dataset and plot the variance captured as a function of the number of principal components.

    Args:
        input_csv (Path): Path to the input CSV file containing the dataset.
    """
    try:
        # Load input data
        df = pd.read_csv(input_csv)

        # Drop non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Scale the features using StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_df)

        # Perform PCA
        pca = PCA()
        pca.fit(scaled_features)

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Plot the variance captured as a function of the number of PCs
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o",
            linestyle="--",
        )
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA - Variance Captured by Principal Components")
        plt.grid(True)
        plt.show()

    except Exception as e:
        logger.error(f"Error performing PCA and plotting variance: {e}")
        raise


@app.command("shap-feature-importance")
def shap_feature_importance(
    data_source: str,
    representation: str,
    model_type: str,
    max_features: int = 25,
    save_plot: bool = False,
) -> None:
    """
    Calculate and visualize SHAP values to explain feature importance for a model.

    Args:
        data_path (Path): Path to the CSV file containing the dataset used for explaining.
        model_path (Path): Path to the saved model file.
        output_path (Path): Optional path to save the SHAP plot.
        max_features (int): Maximum number of features to display in the plot. Default is 10.
        summary_plot (bool): Whether to create a summary plot (True) or bar plot (False). Default is True.

    Raises:
        ValueError: If the input files contain null values or cannot be loaded.
        FileNotFoundError: If the model file does not exist.
    """
    try:
        # Set paths
        data_path = (
            PROCESSED_DATA_DIR / data_source / representation / "train_features.csv"
        )
        # Determine model file extension based on model type
        if model_type.lower() == "fnn":
            model_filename = f"{model_type}_{representation}.pth"
        else:
            model_filename = f"{model_type}_{representation}.pkl"

        model_path = MODELS_DIR / data_source / representation / model_filename

        # Check if model file exists
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Load the dataset
        logger.info(f"Loading dataset from {data_path}")
        data = pd.read_csv(data_path)

        # Check for null values
        if data.isnull().any().any():
            msg = "Input dataset contains null values. Please clean the data before proceeding."
            raise ValueError(msg)

        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Create the SHAP explainer based on model type
        logger.info("Creating SHAP explainer")
        explainer = shap.Explainer(model)

        # Calculate SHAP values
        logger.info("Computing SHAP values")
        shap_values = explainer(data)

        # Create SHAP visualization
        logger.info("Creating SHAP visualization")

        # Create the plot - SHAP creates its own figure
        plt.figure(figsize=(12, 8))

        # For saving the plot, use matplotlib's savefig before showing
        if save_plot:
            logger.info("Saving SHAP plot")
            save_path = model_path.parent / f"shap_{model_type}_{representation}.png"
            # Draw the plot but don't display yet
            shap.plots.beeswarm(shap_values, max_display=max_features, show=False)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            logger.info(f"SHAP plot saved to {save_path}")

        # Create a new plot for display
        shap.plots.beeswarm(shap_values, max_display=max_features)

    except ImportError:
        logger.error("The 'shap' package is required for this functionality.")
        raise
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        raise


@app.command("lgbm-feature-importance")
def lgbm_feature_importance(
    data_source: str,
    representation: str,
    model_type: str,
    top_features: int = 25,
    save_plot: bool = False,
) -> None:
    """
    Extract and plot the top feature importances from a trained LightGBM model,
    coloring host and guest features differently.

    Args:
        data_source (str): Name of the data source directory.
        representation (str): Name of the representation directory.
        model_type (str): Model type (should be 'lgbm' or similar).
        top_features (int): Number of top features to plot. Default is 25.
        save_plot (bool): Whether to save the plot as a PNG file.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model is not a LightGBM model or feature importances are unavailable.
    """
    try:
        # Set model and data paths
        model_path = (
            MODELS_DIR
            / data_source
            / representation
            / f"{model_type}_{representation}.pkl"
        )
        data_path = (
            PROCESSED_DATA_DIR / data_source / representation / "train_features.csv"
        )

        # Check if model file exists
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Load feature names from data
        logger.info(f"Loading feature names from {data_path}")
        data = pd.read_csv(data_path)
        feature_names = list(data.columns)

        # Extract feature importances
        importances = model.feature_importance(importance_type="split")
        if len(importances) != len(feature_names):
            msg = "Mismatch between number of features in model and data."
            raise ValueError(msg)

        # Create a DataFrame for sorting
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )

        # Sort by importance and select top features
        top_df = importance_df.sort_values(by="importance", ascending=False).head(
            top_features
        )

        # Determine host/guest assignment
        total_features = len(feature_names)
        half = total_features // 2

        def get_owner(idx: int) -> str:
            return "Host" if idx < half else "Guest"

        # Map feature to owner
        feature_to_owner = {
            feature: get_owner(feature_names.index(feature))
            for feature in top_df["feature"]
        }
        top_df["owner"] = top_df["feature"].map(feature_to_owner)

        # Assign colors
        color_map = {"Host": "#1f77b4", "Guest": "#ff7f0e"}
        bar_colors = [color_map[owner] for owner in top_df["owner"]]

        # Plot
        logger.info("Plotting feature importances with host/guest coloring")
        plt.figure(figsize=(12, 8))
        # Plot horizontal bar chart for feature importances
        plt.barh(
            top_df["feature"][::-1],
            top_df["importance"][::-1],
            color=bar_colors[::-1],
            edgecolor="black",
        )
        plt.xlabel("Importance")
        plt.title(f"Top {top_features} Feature Importances ({model_type})")
        plt.tight_layout()
        plt.grid(axis="x")

        # Create legend
        legend_handles = [
            Patch(color=color_map["Host"], label="Host Feature"),
            Patch(color=color_map["Guest"], label="Guest Feature"),
        ]
        plt.legend(handles=legend_handles, loc="lower right")

        # Save plot if requested
        if save_plot:
            save_path = (
                model_path.parent / f"lgbm_feature_importance_{representation}.png"
            )
            plt.savefig(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error extracting or plotting feature importances: {e}")
        raise


@app.command("lgbm-total-importance")
def lgbm_total_importance(
    data_source: str,
    representation: str,
    model_type: str,
    save_plot: bool = False,
) -> None:
    """
    Plot the total (cumulative) feature importance as a function of features,
    ranked from highest to lowest, for a trained LightGBM model.
    The y-axis is scaled cumulative importance (0 to 1), with tick marks at every 0.1.

    Args:
        data_source (str): Name of the data source directory.
        representation (str): Name of the representation directory.
        model_type (str): Model type (should be 'lgbm' or similar).
        save_plot (bool): Whether to save the plot as a PNG file.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model is not a LightGBM model or feature importances are unavailable.
    """
    try:
        # Set model and data paths
        model_path = (
            MODELS_DIR
            / data_source
            / representation
            / f"{model_type}_{representation}.pkl"
        )
        data_path = (
            PROCESSED_DATA_DIR / data_source / representation / "train_features.csv"
        )

        # Check if model file exists
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Load feature names from data
        logger.info(f"Loading feature names from {data_path}")
        data = pd.read_csv(data_path)
        feature_names = list(data.columns)

        # Extract feature importances
        importances = model.feature_importance(importance_type="split")
        if len(importances) != len(feature_names):
            msg = "Mismatch between number of features in model and data."
            raise ValueError(msg)

        # Create a DataFrame for sorting
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )

        # Sort by importance descending
        importance_df = importance_df.sort_values(
            by="importance", ascending=False
        ).reset_index(drop=True)

        # Calculate cumulative importance
        importance_df["cumulative_importance"] = importance_df["importance"].cumsum()

        # Scale cumulative importance to [0, 1]
        total_importance = importance_df["importance"].sum()
        if total_importance == 0:
            msg = "Total feature importance is zero. Cannot scale."
            raise ValueError(msg)
        importance_df["scaled_cumulative_importance"] = (
            importance_df["cumulative_importance"] / total_importance
        )

        # Plot scaled cumulative importance as a line plot
        logger.info("Plotting scaled cumulative feature importance")
        plt.figure(figsize=(12, 8))
        plt.plot(
            range(1, len(importance_df) + 1),
            importance_df["scaled_cumulative_importance"],
            marker="o",
            linestyle="-",
            color="#1f77b4",
        )
        plt.xlabel("Feature Rank")
        plt.ylabel("Scaled Cumulative Importance")
        plt.title(f"Cumulative Feature Importance ({model_type})")
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.grid(True)
        plt.tight_layout()

        # Save plot if requested
        if save_plot:
            save_path = (
                model_path.parent / f"lgbm_total_importance_{representation}.png"
            )
            plt.savefig(save_path)
            logger.info(f"Cumulative importance plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        logger.error(
            f"Error extracting or plotting cumulative feature importances: {e}"
        )
        raise


@app.command("plot-dataset-overview")
def plot_dataset_overview(
    data_source: str = "OpenCycloDB",
    output_filename: str | None = None,
    save_as_pgf: bool = False,
) -> None:
    """
    Create a two-panel figure showing dataset overview:
    - Panel A: Distribution of binding energies (DeltaG) as a histogram
    - Panel B: Distribution of host molecules broken down by alpha, beta, gamma cyclodextrins and their derivatives

    Args:
        data_source (str): Data source name (default: "OpenCycloDB")
        output_filename (str, optional): Filename to save the plot. If not provided, the plot is shown.
        save_as_pgf (bool): If True, save as PGF format for LaTeX. Also saves a PNG version.
    """

    # Configure matplotlib for PGF output if requested
    if save_as_pgf:
        import matplotlib as mpl

        mpl.use("pgf")
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
                "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage{textgreek}",
            }
        )

    # Use the 'bright' color cycle from scienceplots
    plt.style.use(["science", "bright"])

    # Load the raw data to get Host and DeltaG information
    raw_data_path = Path("data/raw") / data_source / "Data" / "CDEnrichedData.csv"

    if not raw_data_path.exists():
        logger.error(f"Raw data file not found at {raw_data_path}")
        return

    try:
        df = pd.read_csv(raw_data_path)
        logger.info(f"Loaded dataset with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load data from {raw_data_path}: {e}")
        return

    # Create figure with two subplots using textwidth-based sizing
    textwidth = 7  # LaTeX textwidth in inches
    aspect_ratio = 6 / 12  # Height/width ratio
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(width, height))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Panel A: Binding energy distribution
    ax_a = axes[0]
    binding_energies = df["DeltaG"].dropna()

    # Create histogram
    n_bins = 30
    counts, bins, patches = ax_a.hist(
        binding_energies,
        bins=n_bins,
        color=color_cycle[0],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    ax_a.set_xlabel(r"Binding Energy $\Delta$G (kJ/mol)", fontsize=12)
    ax_a.set_ylabel("Count", fontsize=12)
    ax_a.set_title(r"$\mathbf{A}$", fontsize=14, loc="left")
    ax_a.grid(True, alpha=0.3)

    # Add statistics text
    # mean_val = binding_energies.mean()
    # std_val = binding_energies.std()
    # median_val = binding_energies.median()
    # stats_text = f"Mean: {mean_val:.2f} kJ/mol\nStd: {std_val:.2f} kJ/mol\nMedian: {median_val:.2f} kJ/mol\nN = {len(binding_energies)}"
    # ax_a.text(
    #     0.03,
    #     0.97,
    #     stats_text,
    #     transform=ax_a.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.8},
    # )

    # Panel B: Host molecule distribution
    ax_b = axes[1]

    # Categorize hosts
    def categorize_host(host_name):
        host_lower = str(host_name).lower()
        if "alpha" in host_lower:
            if any(
                keyword in host_lower
                for keyword in [
                    "methyl",
                    "hydroxy",
                    "hp-",
                    "sulfo",
                    "acetyl",
                    "carboxy",
                ]
            ):
                return "α-CD\nderiv."  # noqa: RUF001
            return "α-CD"  # noqa: RUF001
        if "beta" in host_lower:
            if any(
                keyword in host_lower
                for keyword in [
                    "methyl",
                    "hydroxy",
                    "hp-",
                    "sulfo",
                    "acetyl",
                    "carboxy",
                ]
            ):
                return "β-CD\nderiv."
            return "β-CD"
        if "gamma" in host_lower:
            if any(
                keyword in host_lower
                for keyword in [
                    "methyl",
                    "hydroxy",
                    "hp-",
                    "sulfo",
                    "acetyl",
                    "carboxy",
                ]
            ):
                return "γ-CD\nderiv."  # noqa: RUF001
            return "γ-CD"  # noqa: RUF001
        return "Other"

    df["Host_Category"] = df["Host"].apply(categorize_host)
    host_counts = df["Host_Category"].value_counts()

    # Define colors for each category - same color for CD and derivatives
    category_colors = {
        "α-CD": color_cycle[0],  # noqa: RUF001
        "α-CD\nderiv.": color_cycle[0],  # noqa: RUF001
        "β-CD": color_cycle[1],
        "β-CD\nderiv.": color_cycle[1],
        "γ-CD": color_cycle[2],  # noqa: RUF001
        "γ-CD\nderiv.": color_cycle[2],  # noqa: RUF001
        "Other": color_cycle[3] if len(color_cycle) > 3 else "gray",
    }

    # Create bar plot
    categories = host_counts.index
    counts = host_counts.values
    colors = [category_colors.get(cat, "gray") for cat in categories]

    bars = ax_b.bar(
        categories, counts, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5
    )

    ax_b.set_xlabel("Host Molecule Type", fontsize=12)
    ax_b.set_ylabel("Count", fontsize=12)
    ax_b.set_title(r"$\mathbf{B}$", fontsize=14, loc="left")
    ax_b.grid(True, alpha=0.3)

    # Don't rotate x-axis labels since we're using newlines
    ax_b.tick_params(axis="x", rotation=0)

    # Add count labels on top of bars
    for bar, count in zip(bars, counts, strict=False):
        height = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add slightly more space above the bars to prevent count labels from being cut off
    ax_b.set_ylim(0, max(counts) * 1.15)

    # Add total count
    # total_hosts = len(df)
    # ax_b.text(
    #     0.97,
    #     0.97,
    #     f"Total: {total_hosts} complexes",
    #     transform=ax_b.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    #     bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.8},
    # )

    plt.tight_layout()

    # Save or show
    if output_filename is not None:
        save_path = FIGURES_DIR / output_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_as_pgf:
            # Save as PGF for LaTeX
            pgf_path = save_path.with_suffix(".pgf")
            logger.info(f"Saving dataset overview plot as PGF to {pgf_path}")
            plt.savefig(pgf_path, bbox_inches="tight")

            # Also save as PNG for preview
            png_path = save_path.with_suffix(".pdf")
            logger.info(f"Saving dataset overview plot as PNG to {png_path}")
            plt.savefig(png_path, bbox_inches="tight", dpi=300)
        else:
            # Save in the original format specified by the filename extension
            logger.info(f"Saving dataset overview plot to {save_path}")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.close()
    else:
        plt.show()


def generate_umap_embeddings(
    train_dir: str = "data/processed/OpenCycloDB",
    embedding_type: str = "both",
    use_config_hyperparams: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    force_regenerate: bool = False,
) -> dict:
    """
    Generate UMAP embeddings for all representations and validation sets.
    Saves embeddings to disk and returns paths to the saved files.

    Args:
        train_dir: Directory containing training set representations
        val_dir: Directory containing validation set representations
        embedding_type: Type of embeddings ('host', 'guest', or 'both')
        use_config_hyperparams: Whether to use hyperparameters from config file
        n_neighbors: Default number of neighbors for UMAP
        min_dist: Default minimum distance for UMAP
        random_state: Random state for reproducible UMAP
        force_regenerate: Whether to regenerate even if embeddings exist

    Returns:
        Dictionary containing paths to saved UMAP embeddings and metadata
    """
    try:
        from umap import UMAP
    except ImportError:
        logger.error("UMAP not installed. Please install with: pip install umap-learn")
        return {}

    # Create output directory for UMAP embeddings
    umap_dir = DATA_DIR / "interim" / "umap_embeddings" / embedding_type
    umap_dir.mkdir(parents=True, exist_ok=True)

    representations = [
        "ecfp",
        "unimol",
        "grover",
        "string",
        "string_finetuned",
    ]

    val_sets = [
        ("OpenCycloDB Test", "data/processed/OpenCycloDB", "OpenCycloDB_Test"),
        ("Ext. PFAS Val.", "data/external/validation/pfas_val", "PFAS_Validation"),
        ("Ext. CD Val.", "data/external/validation/cd_val", "CD_Validation"),
    ]

    def get_umap_hyperparams(rep: str, val_label: str) -> dict:
        """Get UMAP hyperparameters for a specific representation-validation set combination."""
        if not use_config_hyperparams:
            return {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "random_state": random_state,
                "metric": "euclidean",
            }

        defaults = umap_hyperparameters.get(
            "default",
            {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "random_state": 42,
                "metric": "euclidean",
            },
        )

        rep_config = umap_hyperparameters.get(rep, {})
        params = defaults.copy()

        if "default" in rep_config:
            params.update(rep_config["default"])

        if val_label in rep_config:
            params.update(rep_config[val_label])

        params["random_state"] = random_state
        return params

    def load_embeddings(file_path: Path, embedding_type: str, rep: str) -> np.ndarray:
        """Load embeddings based on type and representation."""
        try:
            if rep in ["string", "string_finetuned"]:
                df = pd.read_csv(file_path)
                if embedding_type == "both":
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    return numeric_data
                if embedding_type == "host":
                    host_cols = [
                        col
                        for col in df.columns
                        if col.startswith("host_") and col != "host_smiles"
                    ]
                    if host_cols:
                        return df[host_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    if numeric_data.shape[1] > 1:
                        return numeric_data[:, : numeric_data.shape[1] // 2]
                    return numeric_data
                if embedding_type == "guest":
                    guest_cols = [
                        col
                        for col in df.columns
                        if col.startswith("guest_") and col != "guest_smiles"
                    ]
                    if guest_cols:
                        return df[guest_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    if numeric_data.shape[1] > 1:
                        return numeric_data[:, numeric_data.shape[1] // 2 :]
                    return numeric_data
            else:
                df = pd.read_csv(file_path)
                if embedding_type == "both":
                    guest_cols = [col for col in df.columns if col.startswith("Guest_")]
                    host_cols = [col for col in df.columns if col.startswith("Host_")]

                    if guest_cols and host_cols:
                        guest_data = df[guest_cols].values
                        host_data = df[host_cols].values
                        return np.concatenate([guest_data, host_data], axis=1)

                    numeric_data = df.select_dtypes(include=[np.number]).values
                    midpoint = numeric_data.shape[1] // 2

                    if rep == "grover":
                        host_data = numeric_data[:, :midpoint]
                        guest_data = numeric_data[:, midpoint:]
                        return np.concatenate([guest_data, host_data], axis=1)
                    return numeric_data
                if embedding_type == "host":
                    host_cols = [col for col in df.columns if col.startswith("Host_")]
                    if host_cols:
                        return df[host_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    midpoint = numeric_data.shape[1] // 2
                    if rep == "grover":
                        return numeric_data[:, :midpoint]
                    return numeric_data[:, midpoint:]
                if embedding_type == "guest":
                    guest_cols = [col for col in df.columns if col.startswith("Guest_")]
                    if guest_cols:
                        return df[guest_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    midpoint = numeric_data.shape[1] // 2
                    if rep == "grover":
                        return numeric_data[:, midpoint:]
                    return numeric_data[:, :midpoint]

        except Exception as e:
            logger.warning(f"Error loading embeddings from {file_path}: {e}")
            return np.array([])

    results = {}

    logger.info(
        f"Generating UMAP embeddings for {len(representations)} representations"
    )

    for rep in representations:
        logger.info(f"Processing representation: {rep}")

        # Check if embeddings already exist and don't need regeneration
        rep_dir = umap_dir / rep
        rep_dir.mkdir(parents=True, exist_ok=True)

        umap_model_path = rep_dir / "umap_model.pkl"
        train_umap_path = rep_dir / "train_umap.csv"

        if (
            not force_regenerate
            and umap_model_path.exists()
            and train_umap_path.exists()
        ):
            logger.info(
                f"  UMAP embeddings already exist for {rep}, skipping generation"
            )
            continue

        # Load training data
        if rep == "string":
            train_path = Path(train_dir) / rep / "train_features_embeddings.csv"
        elif rep == "string_finetuned":
            train_path = (
                Path(train_dir) / "string" / "train_features_finetuned_embeddings.csv"
            )
        else:
            train_path = Path(train_dir) / rep / "train_features.csv"

        logger.debug(f"  Loading training embeddings from: {train_path}")
        train_X = load_embeddings(train_path, embedding_type, rep)

        if train_X.size == 0 or train_X.shape[0] < 5:
            logger.warning(f"  Skipping {rep}: insufficient training data")
            continue

        # Fit UMAP model
        first_val_label = val_sets[0][0]
        umap_params = get_umap_hyperparams(rep, first_val_label)
        current_n_neighbors = umap_params["n_neighbors"]
        current_min_dist = umap_params["min_dist"]
        current_metric = umap_params.get("metric", "euclidean")

        logger.debug(
            f"  Fitting UMAP - n_neighbors={current_n_neighbors}, min_dist={current_min_dist}"
        )

        umap_model = UMAP(
            n_components=2,
            n_neighbors=min(current_n_neighbors, train_X.shape[0] - 1),
            min_dist=current_min_dist,
            random_state=random_state,
            metric=current_metric,
        )
        train_umap = umap_model.fit_transform(train_X)

        # Save UMAP model and training embeddings
        joblib.dump(umap_model, umap_model_path)
        train_umap_df = pd.DataFrame(train_umap, columns=["UMAP1", "UMAP2"])
        train_umap_df.to_csv(train_umap_path, index=False)

        logger.debug(f"  Saved UMAP model and training embeddings for {rep}")

        # Generate validation embeddings for all validation sets
        for val_label, val_dir_path, file_identifier in val_sets:
            val_clean_label = file_identifier.replace("\n", "_").replace(" ", "_")
            val_umap_path = rep_dir / f"{val_clean_label}_umap.csv"

            if not force_regenerate and val_umap_path.exists():
                logger.debug(
                    f"    Validation embeddings already exist for {rep}-{val_clean_label}"
                )
                continue

            try:
                # Determine validation file path
                if rep == "string":
                    if "OCDB" in val_label or "OpenCycloDB" in val_label:
                        val_path = (
                            Path(train_dir) / rep / "test_features_embeddings.csv"
                        )
                    elif "base_cdpfas_val" in val_dir_path:
                        val_path = (
                            Path(val_dir_path)
                            / rep
                            / "base_cdpfas_val_canonical_embeddings.csv"
                        )
                    elif "cd_val" in val_dir_path:
                        val_path = (
                            Path(val_dir_path) / rep / "cd_val_canonical_embeddings.csv"
                        )
                    elif "pfas_val" in val_dir_path:
                        val_path = (
                            Path(val_dir_path)
                            / rep
                            / "pfas_val_canonical_embeddings.csv"
                        )
                elif rep == "string_finetuned":
                    if "OCDB" in val_label or "OpenCycloDB" in val_label:
                        val_path = (
                            Path(train_dir)
                            / "string"
                            / "test_features_finetuned_embeddings.csv"
                        )
                    elif "base_cdpfas_val" in val_dir_path:
                        val_path = (
                            Path(val_dir_path)
                            / "string"
                            / "base_cdpfas_val_canonical_finetuned_embeddings.csv"
                        )
                    elif "cd_val" in val_dir_path:
                        val_path = (
                            Path(val_dir_path)
                            / "string"
                            / "cd_val_canonical_finetuned_embeddings.csv"
                        )
                    elif "pfas_val" in val_dir_path:
                        val_path = (
                            Path(val_dir_path)
                            / "string"
                            / "pfas_val_canonical_finetuned_embeddings.csv"
                        )
                else:
                    if "OCDB" in val_label or "OpenCycloDB" in val_label:
                        val_path = Path(train_dir) / rep / "test_features.csv"
                    else:
                        canonical_val_path = (
                            Path(val_dir_path)
                            / rep
                            / f"{Path(val_dir_path).name}_canonical_{rep}.csv"
                        )
                        default_val_path = Path(val_dir_path) / rep / "val_features.csv"
                        val_path = (
                            canonical_val_path
                            if canonical_val_path.exists()
                            else default_val_path
                        )

                # Load and transform validation data
                val_X = load_embeddings(val_path, embedding_type, rep)

                if val_X.size == 0 or val_X.shape[0] < 1:
                    logger.warning(
                        f"    Skipping {rep}-{val_clean_label}: insufficient validation data"
                    )
                    continue

                val_umap = umap_model.transform(val_X)
                val_umap_df = pd.DataFrame(val_umap, columns=["UMAP1", "UMAP2"])
                val_umap_df.to_csv(val_umap_path, index=False)

                logger.debug(
                    f"    Generated validation embeddings for {rep}-{val_clean_label}"
                )

            except Exception as e:
                logger.warning(
                    f"    Failed to generate validation embeddings for {rep}-{val_clean_label}: {e}"
                )

        results[rep] = {
            "umap_model_path": umap_model_path,
            "train_umap_path": train_umap_path,
            "representation_dir": rep_dir,
        }

    logger.info(
        f"UMAP embedding generation completed. Generated embeddings for {len(results)} representations"
    )
    return results


@app.command("generate-umap-embeddings")
def generate_umap_embeddings_cli(
    train_dir: str = typer.Option(
        "data/processed/OpenCycloDB",
        help="Directory containing training set representations",
    ),
    embedding_type: str = typer.Option(
        "both", help="Type of embeddings: 'host', 'guest', or 'both'"
    ),
    use_config_hyperparams: bool = typer.Option(
        True, help="Whether to use hyperparameters from config file"
    ),
    n_neighbors: int = typer.Option(15, help="Default number of neighbors for UMAP"),
    min_dist: float = typer.Option(0.1, help="Default minimum distance for UMAP"),
    random_state: int = typer.Option(42, help="Random state for reproducible UMAP"),
    force_regenerate: bool = typer.Option(
        False, help="Force regeneration of existing embeddings"
    ),
) -> None:
    """Generate UMAP embeddings for all representations and validation sets."""
    results = generate_umap_embeddings(
        train_dir=train_dir,
        embedding_type=embedding_type,
        use_config_hyperparams=use_config_hyperparams,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        force_regenerate=force_regenerate,
    )

    successful = len([r for r in results.values() if r])
    typer.echo(
        f"UMAP embedding generation complete: {successful} representations processed"
    )


@app.command("plot-train-val-representation-umap-grid")
def plot_train_val_representation_umap_grid(
    train_dir: str = typer.Option(
        "data/processed/OpenCycloDB",
        help="Directory containing training set representations (CSV files)",
    ),
    output_filename: str | None = None,
    embedding_type: str = typer.Option(
        "both", help="Type of embeddings to plot: 'host', 'guest', or 'both'"
    ),
    use_config_hyperparams: bool = typer.Option(
        True,
        help="Whether to use hyperparameters from config file (True) or command line defaults (False)",
    ),
    n_neighbors: int = typer.Option(
        15,
        help="Default number of neighbors parameter for UMAP (used if use_config_hyperparams=False)",
    ),
    min_dist: float = typer.Option(
        0.1,
        help="Default minimum distance parameter for UMAP (used if use_config_hyperparams=False)",
    ),
    random_state: int = typer.Option(42, help="Random state for reproducible UMAP"),
    regenerate_embeddings: bool = typer.Option(
        False, help="Regenerate UMAP embeddings even if they already exist"
    ),
) -> None:
    """
    Visualize the difference between training and validation set representations using UMAP.
    Uses pre-computed UMAP embeddings if available, otherwise generates them first.

    Args:
        train_dir: Directory containing training set representations
        output_filename: Filename to save the plot (if None, plot is shown)
        embedding_type: Type of embeddings to plot ('host', 'guest', or 'both')
        use_config_hyperparams: Whether to use hyperparameters from config file
        n_neighbors: Default number of neighbors for UMAP
        min_dist: Default minimum distance for UMAP
        random_state: Random state for reproducible UMAP
        regenerate_embeddings: Whether to regenerate embeddings even if they exist
    """
    # Generate UMAP embeddings if needed
    generate_umap_embeddings(
        train_dir=train_dir,
        embedding_type=embedding_type,
        use_config_hyperparams=use_config_hyperparams,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        force_regenerate=regenerate_embeddings,
    )

    # Now plot using the pre-computed embeddings
    plot_umap_grid_from_embeddings(
        train_dir=train_dir,
        output_filename=output_filename,
        embedding_type=embedding_type,
    )


def plot_umap_grid_from_embeddings(
    train_dir: str = "data/processed/OpenCycloDB",
    output_filename: str | None = None,
    embedding_type: str = "both",
) -> None:
    """
    Plot UMAP grid using pre-computed embeddings.

    Args:
        train_dir: Directory containing training set representations
        output_filename: Filename to save the plot (if None, plot is shown)
        embedding_type: Type of embeddings plotted ('host', 'guest', or 'both')
    """
    import numpy as np

    # Check if embeddings exist
    umap_dir = DATA_DIR / "interim" / "umap_embeddings" / embedding_type
    if not umap_dir.exists():
        logger.error(f"UMAP embeddings directory not found: {umap_dir}")
        logger.error("Please run generate-umap-embeddings first")
        return

    representations = ["ecfp", "unimol", "grover", "string", "string_finetuned"]
    rep_labels = ["ECFP", "UniMol2", "GROVER", "ChemBERTa", "ChemBERTa\nFinetuned"]

    val_sets = [
        (
            r"$\text{OCDB}_{\text{test}}$",
            "data/processed/OpenCycloDB",
            "OpenCycloDB_Test",
        ),
        (
            r"$\text{E-PFAS}_{\text{test}}$",
            "data/external/validation/pfas_val",
            "PFAS_Validation",
        ),
        # Line 2478
        (
            r"$\text{E-}\beta\text{-CD}_{\text{test}}$",
            "data/external/validation/cd_val",
            "CD_Validation",
        ),
    ]

    n_rows = len(representations)
    n_cols = len(val_sets)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def average_pairwise_cosine_similarity(
        train_X: np.ndarray, val_X: np.ndarray, batch_size: int = 1000
    ) -> float:
        """Compute average cosine similarity between validation and training sets."""
        train_norm = train_X / (np.linalg.norm(train_X, axis=1, keepdims=True) + 1e-12)
        val_norm = val_X / (np.linalg.norm(val_X, axis=1, keepdims=True) + 1e-12)

        total_similarity = 0.0
        total_pairs = 0

        for start in range(0, val_X.shape[0], batch_size):
            end = min(start + batch_size, val_X.shape[0])
            batch_similarities = val_norm[start:end] @ train_norm.T
            total_similarity += np.sum(batch_similarities)
            total_pairs += batch_similarities.size

        return float(total_similarity / total_pairs)

    def average_pairwise_tanimoto_similarity(
        train_X: np.ndarray, val_X: np.ndarray, batch_size: int = 1000
    ) -> float:
        """Compute average Tanimoto similarity between validation and training sets."""
        total_similarity = 0.0
        total_pairs = 0

        for start in range(0, val_X.shape[0], batch_size):
            end = min(start + batch_size, val_X.shape[0])
            val_batch = val_X[start:end]

            # Calculate Tanimoto similarity for each validation sample against all training samples
            # Tanimoto = (A · B) / (||A||² + ||B||² - A · B)
            dot_products = val_batch @ train_X.T  # Shape: (batch_size, n_train)
            val_norms_sq = np.sum(
                val_batch**2, axis=1, keepdims=True
            )  # Shape: (batch_size, 1)
            train_norms_sq = np.sum(train_X**2, axis=1)  # Shape: (n_train,)

            denominators = val_norms_sq + train_norms_sq - dot_products
            # Avoid division by zero
            denominators = np.maximum(denominators, 1e-12)

            batch_similarities = dot_products / denominators
            total_similarity += np.sum(batch_similarities)
            total_pairs += batch_similarities.size

        return float(total_similarity / total_pairs)

    def load_original_embeddings(
        file_path: Path, embedding_type: str, rep: str
    ) -> np.ndarray:
        """Load original embeddings for cosine similarity calculation."""
        try:
            if rep in ["string", "string_finetuned"]:
                df = pd.read_csv(file_path)
                if embedding_type == "both":
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    return numeric_data
                if embedding_type == "host":
                    host_cols = [
                        col
                        for col in df.columns
                        if col.startswith("host_") and col != "host_smiles"
                    ]
                    if host_cols:
                        return df[host_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    if numeric_data.shape[1] > 1:
                        return numeric_data[:, : numeric_data.shape[1] // 2]
                    return numeric_data
                if embedding_type == "guest":
                    guest_cols = [
                        col
                        for col in df.columns
                        if col.startswith("guest_") and col != "guest_smiles"
                    ]
                    if guest_cols:
                        return df[guest_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    if numeric_data.shape[1] > 1:
                        return numeric_data[:, numeric_data.shape[1] // 2 :]
                    return numeric_data
            else:
                df = pd.read_csv(file_path)
                if embedding_type == "both":
                    guest_cols = [col for col in df.columns if col.startswith("Guest_")]
                    host_cols = [col for col in df.columns if col.startswith("Host_")]
                    if guest_cols and host_cols:
                        guest_data = df[guest_cols].values
                        host_data = df[host_cols].values
                        return np.concatenate([guest_data, host_data], axis=1)
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    midpoint = numeric_data.shape[1] // 2
                    if rep == "grover":
                        host_data = numeric_data[:, :midpoint]
                        guest_data = numeric_data[:, midpoint:]
                        return np.concatenate([guest_data, host_data], axis=1)
                    return numeric_data
                if embedding_type == "host":
                    host_cols = [col for col in df.columns if col.startswith("Host_")]
                    if host_cols:
                        return df[host_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    midpoint = numeric_data.shape[1] // 2
                    if rep == "grover":
                        return numeric_data[:, :midpoint]
                    return numeric_data[:, midpoint:]
                if embedding_type == "guest":
                    guest_cols = [col for col in df.columns if col.startswith("Guest_")]
                    if guest_cols:
                        return df[guest_cols].values
                    numeric_data = df.select_dtypes(include=[np.number]).values
                    midpoint = numeric_data.shape[1] // 2
                    if rep == "grover":
                        return numeric_data[:, midpoint:]
                    return numeric_data[:, :midpoint]
        except Exception as e:
            logger.warning(f"Error loading original embeddings from {file_path}: {e}")
            return np.array([])

    # Create the grid plot
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5, 1.6 * n_rows),
        sharex=True,  # Share x-axes along columns
        sharey=True,  # Share y-axes across all plots
        dpi=300,
    )

    logger.info(
        f"Creating UMAP grid plot for {n_rows} representations and {n_cols} validation sets"
    )

    # First pass: collect all UMAP coordinates to calculate global limits
    all_x_coords = []
    all_y_coords = []

    for rep in representations:
        rep_dir = umap_dir / rep
        train_umap_path = rep_dir / "train_umap.csv"

        if train_umap_path.exists():
            train_umap_df = pd.read_csv(train_umap_path)
            train_umap = train_umap_df[["UMAP1", "UMAP2"]].values
            all_x_coords.append(train_umap[:, 0])
            all_y_coords.append(train_umap[:, 1])

            for _, _, val_file_path in val_sets:
                val_umap_path = rep_dir / f"{val_file_path}_umap.csv"
                if val_umap_path.exists():
                    val_umap_df = pd.read_csv(val_umap_path)
                    val_umap = val_umap_df[["UMAP1", "UMAP2"]].values
                    all_x_coords.append(val_umap[:, 0])
                    all_y_coords.append(val_umap[:, 1])

    # Calculate global min/max for all embeddings
    if all_x_coords and all_y_coords:
        global_x = np.concatenate(all_x_coords)
        global_y = np.concatenate(all_y_coords)
        x_min, x_max = global_x.min(), global_x.max()
        y_min, y_max = global_y.min(), global_y.max()

        # Calculate margins based on global range
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Standard margin: 15%
        x_margin = 0.15 * x_range
        y_margin = 0.15 * y_range

        # Extra margin on left and top for similarity metric: 25% total
        x_margin_left = 0.15 * x_range
        y_margin_top = 0.45 * y_range

        global_xlim = (x_min - x_margin_left, x_max + x_margin)
        global_ylim = (y_min - y_margin, y_max + y_margin_top)

        logger.info(f"Global x-limits: {global_xlim}, y-limits: {global_ylim}")
    else:
        logger.warning("No UMAP embeddings found, using default limits")
        global_xlim = (-10, 10)
        global_ylim = (-10, 10)

    for row, (rep, rep_label) in enumerate(
        zip(representations, rep_labels, strict=False)
    ):
        logger.info(f"Processing representation {row + 1}/{n_rows}: {rep_label}")

        rep_dir = umap_dir / rep
        train_umap_path = rep_dir / "train_umap.csv"

        # Check if embeddings exist for this representation
        if not train_umap_path.exists():
            logger.warning(
                f"Training UMAP embeddings not found for {rep}, skipping row"
            )
            for col in range(n_cols):
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.axis("off")
            continue

        # Load training UMAP embeddings
        train_umap_df = pd.read_csv(train_umap_path)
        train_umap = train_umap_df[["UMAP1", "UMAP2"]].values

        # Load original training embeddings for cosine similarity
        if rep == "string":
            train_orig_path = Path(train_dir) / rep / "train_features_embeddings.csv"
        elif rep == "string_finetuned":
            train_orig_path = (
                Path(train_dir) / "string" / "train_features_finetuned_embeddings.csv"
            )
        else:
            train_orig_path = Path(train_dir) / rep / "train_features.csv"

        train_X_orig = load_original_embeddings(train_orig_path, embedding_type, rep)
        logger.debug(
            f"  Loaded training embeddings from {train_orig_path}: shape {train_X_orig.shape}"
        )

        # Process each validation set
        for col, (val_label, val_dir_path, val_file_path) in enumerate(val_sets):
            logger.debug(
                f"    Processing validation set {col + 1}/{n_cols}: {val_label}"
            )
            ax = axes[row, col] if n_rows > 1 else axes[col]

            val_umap_path = rep_dir / f"{val_file_path}_umap.csv"

            if not val_umap_path.exists():
                logger.warning(
                    f"Validation UMAP embeddings not found for {rep}-{val_file_path}"
                )
                ax.axis("off")
                continue

            try:
                # Load validation UMAP embeddings
                val_umap_df = pd.read_csv(val_umap_path)
                val_umap = val_umap_df[["UMAP1", "UMAP2"]].values

                # Load original validation embeddings for cosine similarity
                if rep == "string":
                    if "OCDB" in val_label or "OpenCycloDB" in val_label:
                        val_orig_path = (
                            Path(train_dir) / rep / "test_features_embeddings.csv"
                        )
                    elif "base_cdpfas_val" in val_dir_path:
                        val_orig_path = (
                            Path(val_dir_path)
                            / rep
                            / "base_cdpfas_val_canonical_embeddings.csv"
                        )
                    elif "cd_val" in val_dir_path:
                        val_orig_path = (
                            Path(val_dir_path) / rep / "cd_val_canonical_embeddings.csv"
                        )
                    elif "pfas_val" in val_dir_path:
                        val_orig_path = (
                            Path(val_dir_path)
                            / rep
                            / "pfas_val_canonical_embeddings.csv"
                        )
                elif rep == "string_finetuned":
                    if "OCDB" in val_label or "OpenCycloDB" in val_label:
                        val_orig_path = (
                            Path(train_dir)
                            / "string"
                            / "test_features_finetuned_embeddings.csv"
                        )
                    elif "base_cdpfas_val" in val_dir_path:
                        val_orig_path = (
                            Path(val_dir_path)
                            / "string"
                            / "base_cdpfas_val_canonical_finetuned_embeddings.csv"
                        )
                    elif "cd_val" in val_dir_path:
                        val_orig_path = (
                            Path(val_dir_path)
                            / "string"
                            / "cd_val_canonical_finetuned_embeddings.csv"
                        )
                    elif "pfas_val" in val_dir_path:
                        val_orig_path = (
                            Path(val_dir_path)
                            / "string"
                            / "pfas_val_canonical_finetuned_embeddings.csv"
                        )
                else:
                    if "OCDB" in val_label or "OpenCycloDB" in val_label:
                        val_orig_path = Path(train_dir) / rep / "test_features.csv"
                    else:
                        canonical_val_path = (
                            Path(val_dir_path)
                            / rep
                            / f"{Path(val_dir_path).name}_canonical_{rep}.csv"
                        )
                        default_val_path = Path(val_dir_path) / rep / "val_features.csv"
                        val_orig_path = (
                            canonical_val_path
                            if canonical_val_path.exists()
                            else default_val_path
                        )

                val_X_orig = load_original_embeddings(
                    val_orig_path, embedding_type, rep
                )
                logger.debug(
                    f"  Loaded validation embeddings from {val_orig_path}: shape {val_X_orig.shape}"
                )

                # Plot
                ax.scatter(
                    train_umap[:, 0],
                    train_umap[:, 1],
                    alpha=0.5,
                    color=color_cycle[0],
                    s=20,
                    marker="o",
                    label="Train" if (row == 0 and col == 0) else None,
                )
                ax.scatter(
                    val_umap[:, 0],
                    val_umap[:, 1],
                    alpha=0.5,
                    color=color_cycle[1],
                    s=20,
                    marker="o",
                    label="Test" if (row == 0 and col == 0) else None,
                )

                # Set axis limits using global limits (shared axes)
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)

                # Calculate similarity using original embeddings
                if train_X_orig.size > 0 and val_X_orig.size > 0:
                    if rep == "ecfp":
                        # Use Tanimoto similarity for ECFP
                        similarity = average_pairwise_tanimoto_similarity(
                            train_X_orig, val_X_orig
                        )
                        sim_label = r"$\overline{\text{TS}}$"
                    else:
                        # Use cosine similarity for other representations
                        similarity = average_pairwise_cosine_similarity(
                            train_X_orig, val_X_orig
                        )
                        sim_label = r"$\overline{\text{CS}}$"
                else:
                    similarity = 0.0
                    sim_label = (
                        r"$\overline{\text{CS}}$"
                        if rep != "ecfp"
                        else r"$\overline{\text{TS}}$"
                    )

                ax.text(
                    0.06,
                    0.94,
                    f"{sim_label}={similarity:.3f}",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="left",
                    va="top",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "black",
                        "pad": 4,
                    },
                )

                ax.grid(True, alpha=0.3)

                # Set axis labels
                if row == n_rows - 1:  # Only show x-axis labels on bottom row
                    ax.set_xlabel("UMAP 1", fontsize=11)
                else:
                    ax.set_xlabel("")  # Remove x-axis label for non-bottom rows
                if col == 0:
                    ax.set_ylabel("UMAP 2", fontsize=11)
                else:
                    ax.set_ylabel("")  # Remove y-axis label for non-first columns

                # Set validation set label as title
                if row == 0:
                    ax.set_title(val_label, fontsize=16, pad=10)

                logger.debug(f"    Successfully plotted {rep_label} - {val_label}")

            except Exception as e:
                ax.axis("off")
                logger.warning(f"Failed to plot {rep_label} {val_label}: {e}")

    # Hide y-tick labels for all columns except the first
    for row in range(n_rows):
        for col in range(1, n_cols):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.tick_params(labelleft=False)

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_cycle[0],
            markersize=8,
            label=r"$\text{OCDB}_{\text{train}}$",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_cycle[1],
            markersize=8,
            label="Test Set",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=16,
        frameon=True,
    )

    plt.tight_layout(
        rect=[0.16, 0.03, 1, 1], h_pad=0.8, w_pad=0.8
    )  # Adjust bottom margin for legend

    # Add row labels
    fig.canvas.draw()
    tight_bbox = fig.subplotpars
    plot_bottom = tight_bbox.bottom * 0.001
    plot_top = tight_bbox.top * 1.07

    for row, rep_label_str in enumerate(rep_labels):
        bbox = axes[row, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        y_center_scaled = (y_center - plot_bottom) / (plot_top - plot_bottom)
        fig.text(
            0.05,
            y_center_scaled,
            rep_label_str,
            va="center",
            ha="center",
            fontsize=16,
            rotation=0,
            fontweight="bold",
        )

    # # Create title
    # embedding_title = {
    #     "both": "Host + Guest Embeddings",
    #     "host": "Host Embeddings Only",
    #     "guest": "Guest Embeddings Only",
    # }[embedding_type]

    # plt.suptitle(
    #     f"UMAP of Train/Val Representations ({embedding_title})",
    #     fontsize=16,
    #     x=0.55,
    #     y=0.98,
    # )

    # Save or show
    if output_filename is not None:
        save_path = FIGURES_DIR / output_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving UMAP grid plot to {save_path}")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        logger.info("Displaying UMAP grid plot")
        plt.show()

    logger.info("UMAP grid plot generation completed successfully")


@app.command("umap-nearest-train-for-val")
def umap_nearest_train_for_val(
    representation: str = typer.Option(
        "ecfp",
        help="Representation (ecfp, unimol, grover, string_finetuned, string)",
    ),
    validation_set: str = typer.Option(
        "OpenCycloDB", help="Validation set (OpenCycloDB, PFAS, CD)"
    ),
    train_dir: str = typer.Option(
        "data/processed/OpenCycloDB",
        help="Directory containing training set representations (CSV files)",
    ),
    embedding_type: str = typer.Option(
        "both", help="Embeddings to use: 'host', 'guest', or 'both'"
    ),
    k: int = typer.Option(5, help="Top-k nearest training neighbors to record"),
    metric: str = typer.Option(
        "auto",
        help="Distance metric: 'auto' (ecfp->tanimoto, others->cosine), 'cosine', 'tanimoto', or 'euclidean'",
    ),
    output_smiles_csv: str | None = typer.Option(
        None,
        help="Optional path to save CSV with SMILES strings and chemical names of nearest neighbors",
    ),
) -> None:
    """
    For a given representation and validation set, compute top-k nearest training samples
    in the embedding space using an appropriate metric (Tanimoto for ecfp, cosine otherwise by default).
    Outputs a CSV with SMILES strings and chemical names.

    When embedding_type is 'host', only analyzes host embeddings and outputs host-to-host comparisons.
    When embedding_type is 'guest', only analyzes guest embeddings and outputs guest-to-guest comparisons.
    When embedding_type is 'both', analyzes combined embeddings and outputs full pair comparisons.
    """

    # Validation set mapping (reuse same as plot_single_umap)
    validation_set_mapping = {
        "OpenCycloDB": ("OpenCycloDB\nValidation", "data/processed/OpenCycloDB"),
        "PFAS": ("PFAS\nValidation", "data/external/validation/pfas_val"),
        "CD": ("CD\nValidation", "data/external/validation/cd_val"),
        "base_cdpfas": (
            "Base CDPFAS\nValidation",
            "data/external/validation/base_cdpfas_val",
        ),
    }

    if validation_set not in validation_set_mapping:
        logger.error(
            f"Invalid validation set: {validation_set}. Must be one of: {list(validation_set_mapping.keys())}"
        )
        return

    if embedding_type not in {"both", "host", "guest"}:
        logger.error("embedding_type must be one of: 'both', 'host', 'guest'")
        return

    # Metric resolution
    metric_l = metric.lower()
    if metric_l == "auto":
        resolved_metric = "tanimoto" if representation == "ecfp" else "cosine"
    else:
        resolved_metric = metric_l
    if resolved_metric not in {"cosine", "tanimoto", "euclidean"}:
        logger.error("metric must be one of: 'auto', 'cosine', 'tanimoto', 'euclidean'")
        return

    val_label, val_dir_path = validation_set_mapping[validation_set]

    # Loader consistent with ordering rules used elsewhere
    def _load_embeddings(file_path: Path, emb_type: str, rep: str) -> np.ndarray:
        try:
            if rep in ["string", "string_finetuned"]:
                df = pd.read_csv(file_path)
                if emb_type == "both":
                    return df.select_dtypes(include=[np.number]).values
                if emb_type == "host":
                    host_cols = [
                        c
                        for c in df.columns
                        if c.startswith("host_") and c != "host_smiles"
                    ]
                    if host_cols:
                        return df[host_cols].values
                    numeric = df.select_dtypes(include=[np.number]).values
                    return numeric[:, : numeric.shape[1] // 2]
                if emb_type == "guest":
                    guest_cols = [
                        c
                        for c in df.columns
                        if c.startswith("guest_") and c != "guest_smiles"
                    ]
                    if guest_cols:
                        return df[guest_cols].values
                    numeric = df.select_dtypes(include=[np.number]).values
                    return numeric[:, numeric.shape[1] // 2 :]
            else:
                df = pd.read_csv(file_path)
                if emb_type == "both":
                    guest_cols = [c for c in df.columns if c.startswith("Guest_")]
                    host_cols = [c for c in df.columns if c.startswith("Host_")]
                    if rep == "ecfp" and guest_cols and host_cols:
                        guest_data = df[guest_cols].values
                        host_data = df[host_cols].values
                        return np.concatenate([guest_data, host_data], axis=1)
                    return df.select_dtypes(include=[np.number]).values
                if emb_type == "host":
                    host_cols = [c for c in df.columns if c.startswith("Host_")]
                    if rep == "ecfp" and host_cols:
                        return df[host_cols].values
                    numeric = df.select_dtypes(include=[np.number]).values
                    mid = numeric.shape[1] // 2
                    if rep == "ecfp":
                        return numeric[:, mid:]
                    return numeric[:, :mid]
                if emb_type == "guest":
                    guest_cols = [c for c in df.columns if c.startswith("Guest_")]
                    if rep == "ecfp" and guest_cols:
                        return df[guest_cols].values
                    numeric = df.select_dtypes(include=[np.number]).values
                    mid = numeric.shape[1] // 2
                    if rep == "ecfp":
                        return numeric[:, :mid]
                    return numeric[:, mid:]
        except Exception as e:
            logger.warning(f"Error loading embeddings from {file_path}: {e}")
            return np.array([])

    # Helper: discover file paths similar to plot_single_umap
    def _resolve_paths(rep: str) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
        if rep == "string":
            train_path = Path(train_dir) / rep / "train_features_embeddings.csv"
            val_path = Path(train_dir) / rep / "val_features_embeddings.csv"
            if "base_cdpfas_val" in val_dir_path:
                val_path = (
                    Path(val_dir_path)
                    / rep
                    / "base_cdpfas_val_canonical_embeddings.csv"
                )
            elif "cd_val" in val_dir_path:
                val_path = Path(val_dir_path) / rep / "cd_val_canonical_embeddings.csv"
            elif "pfas_val" in val_dir_path:
                val_path = (
                    Path(val_dir_path) / rep / "pfas_val_canonical_embeddings.csv"
                )
        elif rep == "string_finetuned":
            train_path = (
                Path(train_dir) / "string" / "train_features_finetuned_embeddings.csv"
            )
            val_path = (
                Path(train_dir) / "string" / "val_features_finetuned_embeddings.csv"
            )
            if "base_cdpfas_val" in val_dir_path:
                val_path = (
                    Path(val_dir_path)
                    / "string"
                    / "base_cdpfas_val_canonical_finetuned_embeddings.csv"
                )
            elif "cd_val" in val_dir_path:
                val_path = (
                    Path(val_dir_path)
                    / "string"
                    / "cd_val_canonical_finetuned_embeddings.csv"
                )
            elif "pfas_val" in val_dir_path:
                val_path = (
                    Path(val_dir_path)
                    / "string"
                    / "pfas_val_canonical_finetuned_embeddings.csv"
                )
        else:
            train_path = Path(train_dir) / rep / "train_features.csv"
            canonical_val_path = (
                Path(val_dir_path)
                / rep
                / f"{Path(val_dir_path).name}_canonical_{rep}.csv"
            )
            default_val_path = Path(val_dir_path) / rep / "val_features.csv"
            val_path = (
                canonical_val_path if canonical_val_path.exists() else default_val_path
            )
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        return train_path, val_path, train_df, val_df

    def _cosine_topk(
        train_X: np.ndarray, val_X: np.ndarray, topk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ALL cosine similarities, then select exact top-k (accurate for consolidation)."""
        eps = 1e-12
        t_norm = np.maximum(np.linalg.norm(train_X, axis=1, keepdims=True), eps)
        v_norm = np.maximum(np.linalg.norm(val_X, axis=1, keepdims=True), eps)
        train_n = train_X / t_norm
        sims_indices = []
        sims_values = []
        batch = max(1, 2048 // max(1, train_X.shape[1]))  # crude heuristic
        for start in range(0, val_X.shape[0], batch):
            end = min(start + batch, val_X.shape[0])
            v_batch = val_X[start:end] / v_norm[start:end]
            sims = v_batch @ train_n.T  # (b, n_train) - ALL similarities computed
            # Sort to get exact top-k (changed from argpartition for accuracy)
            idx_sorted = np.argsort(-sims, axis=1)[:, :topk]
            vals_sorted = np.take_along_axis(sims, idx_sorted, axis=1)
            sims_indices.append(idx_sorted)
            sims_values.append(vals_sorted)
        return np.vstack(sims_indices), np.vstack(sims_values)

    def _tanimoto_topk(
        train_X: np.ndarray, val_X: np.ndarray, topk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ALL Tanimoto similarities, then select exact top-k (accurate for consolidation)."""
        eps = 1e-12
        t_sq = np.sum(train_X * train_X, axis=1)  # (n_train,)
        sims_indices = []
        sims_values = []
        batch = max(
            1, 512 // max(1, train_X.shape[1] // 256)
        )  # smaller batches for heavy mul
        for start in range(0, val_X.shape[0], batch):
            end = min(start + batch, val_X.shape[0])
            v_batch = val_X[start:end]
            v_sq = np.sum(v_batch * v_batch, axis=1)[:, None]  # (b,1)
            dot = v_batch @ train_X.T  # (b, n_train) - ALL similarities computed
            denom = v_sq + t_sq[None, :] - dot + eps
            sims = dot / denom
            # Sort to get exact top-k (changed from argpartition for accuracy)
            idx_sorted = np.argsort(-sims, axis=1)[:, :topk]
            vals_sorted = np.take_along_axis(sims, idx_sorted, axis=1)
            sims_indices.append(idx_sorted)
            sims_values.append(vals_sorted)
        return np.vstack(sims_indices), np.vstack(sims_values)

    def _euclidean_topk(
        train_X: np.ndarray, val_X: np.ndarray, topk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute ALL Euclidean distances, then select exact top-k smallest (accurate for consolidation)."""
        # Return indices of smallest distances, but we will also return similarity as -distance for consistency
        sims_indices = []
        sims_values = []
        t_sq = np.sum(train_X * train_X, axis=1)
        batch = max(1, 2048 // max(1, train_X.shape[1]))
        for start in range(0, val_X.shape[0], batch):
            end = min(start + batch, val_X.shape[0])
            v_batch = val_X[start:end]
            v_sq = np.sum(v_batch * v_batch, axis=1)[:, None]
            dot = v_batch @ train_X.T
            d2 = np.clip(v_sq + t_sq[None, :] - 2.0 * dot, a_min=0.0, a_max=None)
            dist = np.sqrt(d2)  # ALL distances computed
            # Sort to get exact top-k smallest distances (changed from argpartition for accuracy)
            idx_sorted = np.argsort(dist, axis=1)[:, :topk]
            vals_sorted = np.take_along_axis(dist, idx_sorted, axis=1)
            # convert to similarity as negative distance for ranking consistency
            sims_indices.append(idx_sorted)
            sims_values.append(-vals_sorted)
        return np.vstack(sims_indices), np.vstack(sims_values)

    # Resolve file paths and load raw DataFrames
    try:
        train_path, val_path, train_df_raw, val_df_raw = _resolve_paths(representation)
    except Exception as e:
        logger.error(f"Failed resolving/reading files: {e}")
        return

    # Load numeric embeddings with correct host/guest handling
    logger.info(f"Loading training embeddings from: {train_path}")
    train_X = _load_embeddings(train_path, embedding_type, representation)
    logger.info(f"Loading validation embeddings from: {val_path}")
    val_X = _load_embeddings(val_path, embedding_type, representation)

    if train_X.size == 0 or val_X.size == 0:
        logger.error("Empty embeddings encountered. Aborting.")
        return
    if train_X.shape[1] != val_X.shape[1]:
        logger.error(
            f"Feature dimension mismatch: train {train_X.shape[1]} vs val {val_X.shape[1]}"
        )
        return

    # Compute top-k neighbors
    # We compute more than k to account for potential duplicates in the training set
    # (e.g., same guest molecule with different hosts)
    search_k = min(
        k * 3, train_X.shape[0]
    )  # Search up to 3x k neighbors, but not more than training set size
    logger.info(
        f"Computing top-{search_k} neighbors (to find {k} unique) using metric={resolved_metric}"
    )
    if resolved_metric == "cosine":
        nbr_idx, nbr_sim = _cosine_topk(train_X, val_X, search_k)
    elif resolved_metric == "tanimoto":
        nbr_idx, nbr_sim = _tanimoto_topk(train_X, val_X, search_k)
    else:  # euclidean
        nbr_idx, nbr_sim = _euclidean_topk(train_X, val_X, search_k)

    # Load SMILES from string representation files (consistent location)
    def _load_smiles_data() -> tuple[dict[str, list], dict[str, list]]:
        """Load SMILES strings and chemical names from the string representation files"""
        try:
            # Load chemical names mapping from CDEnrichedData.csv for training data
            cd_enriched_path = Path("data/raw/OpenCycloDB/Data/CDEnrichedData.csv")
            train_host_name_map = {}
            train_guest_name_map = {}

            if cd_enriched_path.exists():
                cd_enriched_df = pd.read_csv(cd_enriched_path)
                # Create mappings from SMILES to chemical names
                if (
                    "IsomericSMILES_Host" in cd_enriched_df.columns
                    and "Host" in cd_enriched_df.columns
                ):
                    train_host_name_map = dict(
                        zip(
                            cd_enriched_df["IsomericSMILES_Host"].astype(str),
                            cd_enriched_df["Host"].astype(str),
                            strict=False,
                        )
                    )
                if (
                    "IsomericSMILES" in cd_enriched_df.columns
                    and "Guest" in cd_enriched_df.columns
                ):
                    train_guest_name_map = dict(
                        zip(
                            cd_enriched_df["IsomericSMILES"].astype(str),
                            cd_enriched_df["Guest"].astype(str),
                            strict=False,
                        )
                    )

            # Load training SMILES
            train_smiles_path = Path(train_dir) / "string" / "train_features.csv"
            train_smiles_df = pd.read_csv(train_smiles_path)

            train_host_smiles = train_smiles_df["Host_SMILES"].astype(str).tolist()
            train_guest_smiles = train_smiles_df["Guest_SMILES"].astype(str).tolist()

            # Map training data to chemical names
            train_host_names = [
                train_host_name_map.get(smiles, "Unknown")
                for smiles in train_host_smiles
            ]
            train_guest_names = [
                train_guest_name_map.get(smiles, "Unknown")
                for smiles in train_guest_smiles
            ]

            train_smiles = {
                "train_host_smiles": train_host_smiles,
                "train_guest_smiles": train_guest_smiles,
                "train_host_names": train_host_names,
                "train_guest_names": train_guest_names,
            }

            # Load validation SMILES and names
            val_host_name_map = {}
            val_guest_name_map = {}

            if validation_set == "OpenCycloDB":
                val_smiles_path = Path(train_dir) / "string" / "val_features.csv"
                val_smiles_df = pd.read_csv(val_smiles_path)
                val_host_smiles = val_smiles_df["Host_SMILES"].astype(str).tolist()
                val_guest_smiles = val_smiles_df["Guest_SMILES"].astype(str).tolist()
                # For OpenCycloDB validation, use the same training name maps
                val_host_names = [
                    train_host_name_map.get(smiles, "Unknown")
                    for smiles in val_host_smiles
                ]
                val_guest_names = [
                    train_guest_name_map.get(smiles, "Unknown")
                    for smiles in val_guest_smiles
                ]
            else:
                # For external validation sets (cd_val, pfas_val), load names from canonical CSV
                val_canonical_path = (
                    Path(val_dir_path) / f"{Path(val_dir_path).name}_canonical.csv"
                )
                if val_canonical_path.exists():
                    val_canonical_df = pd.read_csv(val_canonical_path)
                    # Create mappings from SMILES to chemical names from canonical CSV
                    if "Host_SMILES" in val_canonical_df.columns:
                        if "CD" in val_canonical_df.columns:
                            val_host_name_map = dict(
                                zip(
                                    val_canonical_df["Host_SMILES"].astype(str),
                                    val_canonical_df["CD"].astype(str),
                                    strict=False,
                                )
                            )
                        elif "PFAS" in val_canonical_df.columns:
                            # For pfas_val where host is CD
                            val_host_name_map = dict(
                                zip(
                                    val_canonical_df["Host_SMILES"].astype(str),
                                    val_canonical_df["CD"].astype(str)
                                    if "CD" in val_canonical_df.columns
                                    else val_canonical_df["PFAS"].astype(str),
                                    strict=False,
                                )
                            )

                    if "Guest_SMILES" in val_canonical_df.columns:
                        if "PFAS" in val_canonical_df.columns:
                            val_guest_name_map = dict(
                                zip(
                                    val_canonical_df["Guest_SMILES"].astype(str),
                                    val_canonical_df["PFAS"].astype(str),
                                    strict=False,
                                )
                            )
                        elif "CD" in val_canonical_df.columns:
                            # For pfas_val where guest is PFAS
                            val_guest_name_map = dict(
                                zip(
                                    val_canonical_df["Guest_SMILES"].astype(str),
                                    val_canonical_df["PFAS"].astype(str)
                                    if "PFAS" in val_canonical_df.columns
                                    else val_canonical_df["CD"].astype(str),
                                    strict=False,
                                )
                            )

                # Load SMILES from canonical file (same file for both SMILES and names)
                val_smiles_df = val_canonical_df
                val_host_smiles = val_smiles_df["Host_SMILES"].astype(str).tolist()
                val_guest_smiles = val_smiles_df["Guest_SMILES"].astype(str).tolist()

                # Map validation data to chemical names
                val_host_names = [
                    val_host_name_map.get(smiles, "Unknown")
                    for smiles in val_host_smiles
                ]
                val_guest_names = [
                    val_guest_name_map.get(smiles, "Unknown")
                    for smiles in val_guest_smiles
                ]

            val_smiles = {
                "val_host_smiles": val_host_smiles,
                "val_guest_smiles": val_guest_smiles,
                "val_host_names": val_host_names,
                "val_guest_names": val_guest_names,
            }

            return val_smiles, train_smiles

        except Exception as e:
            logger.warning(f"Could not load SMILES data: {e}")
            return {}, {}

    val_opt, train_opt = _load_smiles_data()

    # Generate SMILES CSV with chemical names (only output)
    smiles_rows = []
    n_val = val_X.shape[0]

    # Track unique molecules to avoid duplicates
    seen_molecules = set()
    unique_count = 0
    duplicate_count = 0

    for i in range(n_val):
        # Get validation SMILES and names if available
        val_host_smiles = (
            val_opt.get("val_host_smiles", [None] * n_val)[i] if val_opt else None
        )
        val_guest_smiles = (
            val_opt.get("val_guest_smiles", [None] * n_val)[i] if val_opt else None
        )
        val_host_name = (
            val_opt.get("val_host_names", [None] * n_val)[i] if val_opt else None
        )
        val_guest_name = (
            val_opt.get("val_guest_names", [None] * n_val)[i] if val_opt else None
        )

        # Determine the molecule identifier based on embedding_type
        if embedding_type == "host":
            molecule_id = val_host_smiles
        elif embedding_type == "guest":
            molecule_id = val_guest_smiles
        else:  # both
            molecule_id = f"{val_host_smiles}|{val_guest_smiles}"

        # Skip if we've already analyzed this molecule
        if molecule_id in seen_molecules:
            duplicate_count += 1
            continue

        seen_molecules.add(molecule_id)
        unique_count += 1

        # Track unique training neighbors for this validation molecule
        seen_train_neighbors = set()
        actual_rank = 0  # Track the actual rank after deduplication

        # Continue searching until we find k unique neighbors
        # We may need to search beyond k initial neighbors to get k unique ones
        search_rank = 0
        while actual_rank < k and search_rank < nbr_idx.shape[1]:
            j = int(nbr_idx[i, search_rank])
            sim = float(nbr_sim[i, search_rank])

            # Get training SMILES and names if available
            train_host_smiles = (
                train_opt.get("train_host_smiles", [None] * train_X.shape[0])[j]
                if train_opt
                else None
            )
            train_guest_smiles = (
                train_opt.get("train_guest_smiles", [None] * train_X.shape[0])[j]
                if train_opt
                else None
            )

            # Determine training neighbor identifier based on embedding_type
            if embedding_type == "host":
                train_neighbor_id = train_host_smiles
            elif embedding_type == "guest":
                train_neighbor_id = train_guest_smiles
            else:  # both
                train_neighbor_id = f"{train_host_smiles}|{train_guest_smiles}"

            # Skip if we've already seen this training neighbor for this validation molecule
            if train_neighbor_id in seen_train_neighbors:
                search_rank += 1
                continue

            seen_train_neighbors.add(train_neighbor_id)
            actual_rank += 1

            # Create row based on embedding_type
            if embedding_type == "host":
                # Only include host data for host-only analysis
                smiles_row = {
                    "val_index": i,
                    "neighbor_rank": actual_rank,
                    "train_index": j,
                    "similarity": sim,
                    "val_smiles": val_host_smiles,
                    "val_name": val_host_name,
                    "train_smiles": train_host_smiles,
                    "train_name": train_host_smiles,  # Use SMILES as identifier
                    "representation": representation,
                    "validation_set": validation_set,
                    "metric": resolved_metric,
                    "embedding_type": embedding_type,
                }
            elif embedding_type == "guest":
                # Only include guest data for guest-only analysis
                smiles_row = {
                    "val_index": i,
                    "neighbor_rank": actual_rank,
                    "train_index": j,
                    "similarity": sim,
                    "val_smiles": val_guest_smiles,
                    "val_name": val_guest_name,
                    "train_smiles": train_guest_smiles,
                    "train_name": train_guest_smiles,  # Use SMILES as identifier
                    "representation": representation,
                    "validation_set": validation_set,
                    "metric": resolved_metric,
                    "embedding_type": embedding_type,
                }
            else:  # embedding_type == "both"
                # Include both host and guest data for full pair analysis
                # Create combined SMILES identifier for host+guest pairs
                train_pair_smiles = f"{train_host_smiles}|{train_guest_smiles}"
                smiles_row = {
                    "val_index": i,
                    "neighbor_rank": actual_rank,
                    "train_index": j,
                    "similarity": sim,
                    "val_host_smiles": val_host_smiles,
                    "val_guest_smiles": val_guest_smiles,
                    "val_host_name": val_host_name,
                    "val_guest_name": val_guest_name,
                    "train_host_smiles": train_host_smiles,
                    "train_guest_smiles": train_guest_smiles,
                    "train_host_name": train_host_smiles,  # Use host SMILES
                    "train_guest_name": train_guest_smiles,  # Use guest SMILES
                    "train_name": train_pair_smiles,  # Combined identifier for pairs
                    "representation": representation,
                    "validation_set": validation_set,
                    "metric": resolved_metric,
                    "embedding_type": embedding_type,
                }
            smiles_rows.append(smiles_row)

            # Move to next candidate
            search_rank += 1

    smiles_df = pd.DataFrame(smiles_rows)

    # Determine SMILES CSV path
    if output_smiles_csv is None:
        default_smiles_dir = (
            Path(REPORTS_DIR) / "val_to_train_neighbors" / representation
        )
        default_smiles_dir.mkdir(parents=True, exist_ok=True)
        smiles_csv_name = f"{representation}_{validation_set}_k{k}_{embedding_type}_{resolved_metric}_smiles.csv"
        smiles_csv_path = default_smiles_dir / smiles_csv_name
    else:
        smiles_csv_path = Path(output_smiles_csv)
        smiles_csv_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_df.to_csv(smiles_csv_path, index=False)
    logger.info(f"Saved SMILES neighbor mapping CSV to: {smiles_csv_path}")
    logger.info(
        f"Analyzed {unique_count} unique molecules (skipped {duplicate_count} duplicates)"
    )


@app.command("run-all-neighbor-analysis")
def run_all_neighbor_analysis(
    k: int = typer.Option(5, help="Top-k nearest training neighbors to record"),
    embedding_type: str = typer.Option(
        "both", help="Embeddings to use: 'host', 'guest', or 'both'"
    ),
) -> None:
    """
    Run nearest neighbor analysis for all representations and validation sets.
    Excludes ecfp_plus as it's typically redundant, uses k=5 by default,
    and organizes outputs by representation type.
    """

    # Representations to analyze (excluding ecfp_plus)
    representations = [
        "ecfp",
        "unimol",
        "grover",
        "string",
        "string_finetuned",
    ]

    # Validation sets to analyze
    validation_sets = ["CD", "PFAS"]

    logger.info(
        f"Running nearest neighbor analysis for {len(representations)} representations x {len(validation_sets)} validation sets"
    )
    logger.info(f"Parameters: k={k}, embedding_type={embedding_type}")
    logger.info("=" * 80)

    success_count = 0
    total_count = 0
    failed_analyses = []

    for rep in representations:
        logger.info(f"\n--- Processing representation: {rep} ---")

        for val_set in validation_sets:
            total_count += 1
            logger.info(f"\n{total_count}. {rep} + {val_set}")

            try:
                # Call the umap_nearest_train_for_val function directly
                umap_nearest_train_for_val(
                    representation=rep,
                    validation_set=val_set,
                    train_dir="data/processed/OpenCycloDB",
                    embedding_type=embedding_type,
                    k=k,
                    metric="auto",
                    output_smiles_csv=None,
                )

                logger.info(f"✓ Successfully processed {rep} + {val_set}")
                success_count += 1

            except Exception as e:
                logger.error(f"✗ Failed to process {rep} + {val_set}: {e}")
                failed_analyses.append(f"{rep} + {val_set}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Analysis complete: {success_count}/{total_count} successful")

    if failed_analyses:
        logger.warning(f"⚠️  {len(failed_analyses)} analyses failed:")
        for failed in failed_analyses:
            logger.warning(f"  - {failed}")
    else:
        logger.info("✓ All analyses completed successfully!")

        # Show the organized output structure
        reports_dir = Path(REPORTS_DIR) / "val_to_train_neighbors"
        if reports_dir.exists():
            logger.info(f"\nOutput files organized in: {reports_dir}")
            for rep_dir in sorted(reports_dir.iterdir()):
                if rep_dir.is_dir():
                    csv_files = list(rep_dir.glob("*.csv"))
                    main_files = [
                        f for f in csv_files if not f.name.endswith("_smiles.csv")
                    ]
                    smiles_files = [
                        f for f in csv_files if f.name.endswith("_smiles.csv")
                    ]
                    logger.info(
                        f"  {rep_dir.name}/: {len(main_files)} neighbor files, {len(smiles_files)} SMILES files"
                    )


@app.command("consolidate-neighbor-analysis")
def consolidate_neighbor_analysis(
    k: int = typer.Option(5, help="Number of neighbors to analyze"),
    _metric: str = typer.Option(
        "auto", help="Distance metric used (auto to detect from files)"
    ),
    rank_by: str = typer.Option(
        "mean", help="Ranking metric: 'mean' or 'median' for similarity scores"
    ),
) -> None:
    """
    Consolidate individual neighbor analysis files into a single ranked list per representation.

    For each representation and embedding type, creates a CSV file:
    - {rep}_{embedding_type}_consolidated_neighbor_avg_similarity.csv

    Each file contains a single ranked list of training neighbors with:
    - training_neighbor_smiles: The SMILES string of the training neighbor
    - average_similarity: Average similarity score across all validation molecules (if rank_by='mean')
    - median_similarity: Median similarity score across all validation molecules (if rank_by='median')
    - occurrence_count: Number of times this neighbor appeared in top-k
    - validation_sets: Which validation sets this neighbor appeared in

    The list is sorted by the selected ranking metric (descending) and handles duplicates
    by using SMILES as the unique identifier.
    """
    logger.info(
        "Starting neighbor analysis consolidation with average similarity scores"
    )

    # Define embedding representations and validation sets
    representations = [
        "ecfp",
        "unimol",
        "grover",
        "string",
        "string_finetuned",
    ]
    validation_sets = ["CD", "PFAS"]
    embedding_types = ["host", "guest"]  # Process host and guest separately

    reports_base_dir = Path(REPORTS_DIR) / "val_to_train_neighbors"

    if not reports_base_dir.exists():
        logger.error(f"Reports directory not found: {reports_base_dir}")
        logger.info("Please run 'run-all-neighbor-analysis' first")
        return

    successful_consolidations = 0
    failed_consolidations = []

    for rep in representations:
        logger.info(f"Processing representation: {rep}")
        rep_dir = reports_base_dir / rep

        if not rep_dir.exists():
            logger.warning(f"Skipping {rep}: directory not found at {rep_dir}")
            failed_consolidations.append(rep)
            continue

        # Process host and guest separately
        for embedding_type in embedding_types:
            logger.info(f"  Processing {embedding_type} embeddings for {rep}")

            try:
                # Dictionary to store all neighbor data: {smiles: {similarities: [], val_sets: set()}}
                all_neighbor_data = {}

                for val_set in validation_sets:
                    logger.debug(f"    Processing validation set: {val_set}")

                    # Find the neighbor file for this validation set and embedding type
                    pattern = f"{rep}_{val_set}_k{k}_{embedding_type}_*_smiles.csv"
                    matching_files = list(rep_dir.glob(pattern))

                    if not matching_files:
                        logger.warning(
                            f"      No neighbor file found for {rep} + {val_set} + {embedding_type} (pattern: {pattern})"
                        )
                        continue

                    if len(matching_files) > 1:
                        logger.warning(
                            f"      Multiple files found for {rep} + {val_set} + {embedding_type}, using first: {matching_files[0].name}"
                        )

                    neighbor_file = matching_files[0]
                    logger.debug(f"      Reading neighbor file: {neighbor_file.name}")

                    try:
                        df = pd.read_csv(neighbor_file)

                        # Validate required columns exist
                        if (
                            "train_name" not in df.columns
                            or "similarity" not in df.columns
                        ):
                            logger.warning(
                                f"      Missing required columns in {neighbor_file.name}"
                            )
                            continue

                        # Process each neighbor occurrence
                        for _, row in df.iterrows():
                            train_smiles = row[
                                "train_name"
                            ]  # This is now the SMILES string
                            similarity = row["similarity"]

                            if pd.isna(train_smiles) or pd.isna(similarity):
                                continue

                            # Initialize entry if new neighbor
                            if train_smiles not in all_neighbor_data:
                                all_neighbor_data[train_smiles] = {
                                    "similarities": [],
                                    "val_sets": set(),
                                }

                            # Add similarity score and track validation set
                            all_neighbor_data[train_smiles]["similarities"].append(
                                float(similarity)
                            )
                            all_neighbor_data[train_smiles]["val_sets"].add(val_set)

                        logger.debug(
                            f"      Processed {len(df)} neighbor entries for {val_set}"
                        )

                    except Exception as e:
                        logger.error(f"      Error reading {neighbor_file.name}: {e}")
                        continue

                if not all_neighbor_data:
                    logger.warning(
                        f"    No neighbor data found for {rep} ({embedding_type}), skipping"
                    )
                    continue

                # Calculate average/median similarity for each unique neighbor
                consolidated_results = []

                for train_smiles, data in all_neighbor_data.items():
                    similarities = data["similarities"]
                    val_sets = data["val_sets"]

                    avg_similarity = np.mean(similarities)
                    median_similarity = np.median(similarities)
                    occurrence_count = len(similarities)
                    val_sets_str = ", ".join(sorted(val_sets))

                    consolidated_results.append(
                        {
                            "training_neighbor_smiles": train_smiles,
                            "average_similarity": avg_similarity,
                            "median_similarity": median_similarity,
                            "occurrence_count": occurrence_count,
                            "validation_sets": val_sets_str,
                            "std_similarity": np.std(similarities),
                            "min_similarity": np.min(similarities),
                            "max_similarity": np.max(similarities),
                        }
                    )

                # Create DataFrame and sort by selected ranking metric (descending)
                consolidated_df = pd.DataFrame(consolidated_results)
                rank_column = (
                    "median_similarity" if rank_by == "median" else "average_similarity"
                )
                consolidated_df = consolidated_df.sort_values(
                    rank_column, ascending=False
                )
                consolidated_df = consolidated_df.reset_index(drop=True)

                # Save to CSV
                output_file = (
                    rep_dir
                    / f"{rep}_{embedding_type}_consolidated_neighbor_avg_similarity.csv"
                )
                consolidated_df.to_csv(output_file, index=False)

                logger.info(
                    f"    ✓ Saved {embedding_type} consolidated analysis to: {output_file}"
                )
                logger.info(f"      - {len(consolidated_df)} unique training neighbors")
                logger.info(f"      - Ranking by: {rank_by}")
                logger.info(
                    f"      - Average similarity range: {consolidated_df['average_similarity'].min():.4f} to {consolidated_df['average_similarity'].max():.4f}"
                )
                logger.info(
                    f"      - Median similarity range: {consolidated_df['median_similarity'].min():.4f} to {consolidated_df['median_similarity'].max():.4f}"
                )
                rank_value = consolidated_df.iloc[0][rank_column]
                logger.info(
                    f"      - Top neighbor: {consolidated_df.iloc[0]['training_neighbor_smiles'][:50]}... ({rank_by} sim: {rank_value:.4f})"
                )

            except Exception as e:
                logger.error(f"    ✗ Failed to process {rep} ({embedding_type}): {e}")
                continue

        successful_consolidations += 1

    # Summary
    logger.info(
        "================================================================================"
    )
    logger.info(
        f"Consolidation complete: {successful_consolidations}/{len(representations)} successful"
    )

    if failed_consolidations:
        logger.warning(f"⚠️  {len(failed_consolidations)} consolidations failed:")
        for failed in failed_consolidations:
            logger.warning(f"  - {failed}")
    else:
        logger.info("✓ All consolidations completed successfully!")


@app.command("plot-consolidated-similarity")
def plot_consolidated_similarity(
    k: int = typer.Option(
        50, help="Number of top neighbors to plot for each representation"
    ),
    embedding_type: str = typer.Option(
        "guest", help="Embedding type: 'host', 'guest', or 'both'"
    ),
    rank_by: str = typer.Option(
        "mean", help="Ranking metric to use: 'mean' or 'median'"
    ),
    output_dir: Path = typer.Option(
        Path("reports/figures"),
        help="Directory to save the plot",
    ),
    save_figure: bool = typer.Option(
        True,
        help="Whether to save the figure to file (if False, displays interactively)",
    ),
    _train_dir: str = typer.Option(
        "data/processed/OpenCycloDB",
        help="Directory containing training set representations",
    ),
):
    """
    Plot average/median similarity scores for top k neighbors from consolidated analysis.

    Creates a 3x2 grid with max 2 columns, one per representation, showing the similarity
    values for the top k training neighbors. The horizontal axis shows neighbor ranks
    (1 to k) without listing individual SMILES strings.

    Reads from: reports/val_to_train_neighbors/{rep}/{rep}_{embedding_type}_consolidated_neighbor_avg_similarity.csv

    Each panel shows:
    - Average or median similarity on y-axis (depending on rank_by parameter)
    - Neighbor rank (1 to k) on x-axis
    - Error bars showing std deviation (optional)
    - Red dashed line for overall mean calculated from external validation sets
    """
    logger.info("=" * 80)
    logger.info("PLOT CONSOLIDATED SIMILARITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Top k neighbors: {k}")
    logger.info(f"Embedding type: {embedding_type}")
    logger.info(f"Ranking metric: {rank_by}")
    logger.info(f"Output directory: {output_dir}")

    # Define representations
    # Order: ECFP, GROVER, UniMol2 (with zoom), ChemBERTa, ChemBERTa Finetuned
    representations = ["ecfp", "grover", "unimol", "string", "string_finetuned"]
    rep_labels = {
        "ecfp": "ECFP",
        "unimol": "UniMol2",
        "grover": "GROVER",
        "string": "ChemBERTa",
        "string_finetuned": "ChemBERTa\nFinetuned",
    }

    # Define y-axis labels with appropriate metrics
    y_labels = {
        "ecfp": r"$\overline{\text{TS}}_{\text{V}}$",
        "unimol": r"$\overline{\text{CS}}_{\text{V}}$",
        "grover": r"$\overline{\text{CS}}_{\text{V}}$",
        "string": r"$\overline{\text{CS}}_{\text{V}}$",
        "string_finetuned": r"$\overline{\text{CS}}_{\text{V}}$",
    }

    # Collect data for all representations
    plot_data = {}

    for rep in representations:
        rep_dir = Path("reports/val_to_train_neighbors") / rep
        csv_file = (
            rep_dir / f"{rep}_{embedding_type}_consolidated_neighbor_avg_similarity.csv"
        )

        if not csv_file.exists():
            logger.warning(f"  ⚠️  File not found: {csv_file}")
            continue

        try:
            df = pd.read_csv(csv_file)

            if len(df) == 0:
                logger.warning(f"  ⚠️  Empty dataframe for {rep}")
                continue

            # Select the appropriate similarity column based on rank_by
            similarity_column = (
                "median_similarity" if rank_by == "median" else "average_similarity"
            )

            if similarity_column not in df.columns:
                logger.warning(
                    f"  ⚠️  Column '{similarity_column}' not found in {csv_file}"
                )
                logger.warning(f"      Available columns: {', '.join(df.columns)}")
                logger.warning(
                    f"      Please re-run consolidate_neighbor_analysis with rank_by='{rank_by}'"
                )
                continue

            # Sort by the selected metric and get top k neighbors
            df_sorted = df.sort_values(similarity_column, ascending=False)
            top_k_df = df_sorted.head(k)

            plot_data[rep] = {
                "avg_similarity": top_k_df[similarity_column].values,
                "std_similarity": top_k_df["std_similarity"].values
                if "std_similarity" in top_k_df.columns
                else None,
                "n_neighbors": len(top_k_df),
                "smiles": top_k_df[
                    "training_neighbor_smiles"
                ].values,  # Store SMILES for identification
            }

            logger.info(f"  ✓ Loaded {len(top_k_df)} neighbors for {rep}")

        except Exception as e:
            logger.error(f"  ✗ Error loading {rep}: {e}")
            continue

    if not plot_data:
        logger.error("No data loaded for any representation. Exiting.")
        return

    # Create the grid plot with 3 rows x 2 columns, total width 7 inches
    # Row 1: ECFP (left), GROVER (right)
    # Row 2: UniMol2 full range (left), UniMol2 zoomed (right)
    # Row 3: ChemBERTa (left), ChemBERTa Finetuned (right)
    logger.info("Creating grid plot with special layout for UniMol2 zoom...")

    # Get color cycle first for colorblind-friendly colors
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Define PFAS molecules to highlight with different colors from color cycle
    # Using indices 3 (red), 2 (green), 9 (cyan) for maximum visual contrast
    highlighted_smiles = {
        "O=C([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F": {
            "color": color_cycle[1],
            "label": "Mol. 1",
            "found": False,
        },
        "O=C(NC1=CC=C([N+](=O)[O-])C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F": {
            "color": "cyan",
            "label": "Mol. 2",
            "found": False,
        },
        "O=C(NC1=CC=CC=C1)C(F)(F)F": {
            "color": "orange",
            "label": "Mol. 3",
            "found": False,
        },
    }

    nrows = 3
    ncols = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7, 3 * nrows + 0.5),
        squeeze=False,  # Extra space for legend
    )

    # Define the layout mapping: representation -> (row, col)
    # ECFP: (0,0), GROVER: (0,1), UniMol2: (1,0) and (1,1) zoomed, ChemBERTa: (2,0), ChemBERTa FT: (2,1)
    layout_map = {
        "ecfp": (0, 0),
        "grover": (0, 1),
        "unimol": (1, 0),  # Will also use (1,1) for zoomed version
        "string": (2, 0),
        "string_finetuned": (2, 1),
    }

    # Plot each representation
    for rep in representations:
        if rep not in plot_data:
            continue

        row, col = layout_map[rep]
        ax = axes[row, col]
        data = plot_data[rep]

        x = np.arange(1, data["n_neighbors"] + 1)
        y = data["avg_similarity"]
        smiles_list = data["smiles"]

        # Separate regular points from highlighted points
        regular_mask = np.ones(len(x), dtype=bool)

        # Plot highlighted molecules first (so they appear on top)
        for smiles_str, info in highlighted_smiles.items():
            mask = np.array([s == smiles_str for s in smiles_list])
            if np.any(mask):
                ax.scatter(
                    x[mask],
                    y[mask],
                    marker="^",
                    s=28,
                    alpha=0.9,
                    color=info["color"],
                    zorder=3,
                    edgecolors="black",
                    linewidths=0.5,
                )
                regular_mask &= ~mask
                info["found"] = True  # Track that this molecule was found

        # Plot regular points
        ax.scatter(
            x[regular_mask],
            y[regular_mask],
            marker="o",
            s=16,
            alpha=0.8,
            color=color_cycle[0],
            zorder=2,
        )

        # Add error bars if std is available
        if data["std_similarity"] is not None:
            ax.fill_between(
                x,
                y - data["std_similarity"],
                y + data["std_similarity"],
                alpha=0.2,
                color=color_cycle[0],
            )

        # Styling
        ax.set_title(rep_labels[rep], fontsize=12, fontweight="bold")
        ax.set_xlabel("Sorted Training Neighbor", fontsize=10)
        ax.set_ylabel(y_labels[rep], fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, data["n_neighbors"] + 1)

        # Set y-axis limits to [0, 1] for all plots
        ax.set_ylim(0, 1)

        # Special handling for UniMol2: create zoomed version in right column
        if rep == "unimol":
            ax_zoom = axes[1, 1]

            # Separate regular points from highlighted points for zoom plot
            regular_mask_zoom = np.ones(len(x), dtype=bool)

            # Plot highlighted molecules first
            for smiles_str, info in highlighted_smiles.items():
                mask = np.array([s == smiles_str for s in smiles_list])
                if np.any(mask):
                    ax_zoom.scatter(
                        x[mask],
                        y[mask],
                        marker="^",
                        s=28,
                        alpha=0.9,
                        color=info["color"],
                        zorder=3,
                        edgecolors="black",
                        linewidths=0.5,
                    )
                    regular_mask_zoom &= ~mask

            # Plot regular points
            ax_zoom.scatter(
                x[regular_mask_zoom],
                y[regular_mask_zoom],
                marker="o",
                s=16,
                alpha=0.8,
                color=color_cycle[0],
                zorder=2,
            )

            if data["std_similarity"] is not None:
                ax_zoom.fill_between(
                    x,
                    y - data["std_similarity"],
                    y + data["std_similarity"],
                    alpha=0.2,
                    color=color_cycle[0],
                )

            # Styling for zoomed plot
            ax_zoom.set_title(
                f"{rep_labels[rep]} (Zoomed)", fontsize=12, fontweight="bold"
            )
            ax_zoom.set_xlabel("Sorted Training Neighbor", fontsize=10)
            ax_zoom.set_ylabel(y_labels[rep], fontsize=11)
            ax_zoom.grid(True, alpha=0.3, linestyle="--")
            ax_zoom.set_xlim(0, data["n_neighbors"] + 1)

            # Set zoomed y-axis limits with padding
            y_min, y_max = y.min(), y.max()
            y_range = y_max - y_min
            ax_zoom.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    # Add legend for highlighted molecules below the grid
    from matplotlib.lines import Line2D

    legend_elements = []

    # Add highlighted molecules first
    for _smiles_str, info in highlighted_smiles.items():
        if info["found"]:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor=info["color"],
                    markersize=7,
                    label=info["label"],
                    alpha=0.9,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            )

    # Add 'Other training molecules' last
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_cycle[0],
            markersize=6,
            label="Other training molecules",
            alpha=0.8,
        )
    )

    # Create legend below the grid
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])  # Leave space for legend

    # Save or display the figure
    if save_figure:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir / f"consolidated_similarity_top{k}_{embedding_type}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Plot saved to: {output_file}")

        # Also save as PDF
        output_file_pdf = (
            output_dir / f"consolidated_similarity_top{k}_{embedding_type}.pdf"
        )
        plt.savefig(output_file_pdf, bbox_inches="tight")
        logger.info(f"✓ PDF saved to: {output_file_pdf}")

        plt.close()
    else:
        logger.info("Displaying plot interactively...")
        plt.show()

    logger.info("=" * 80)
    logger.info("PLOT COMPLETE")
    logger.info("=" * 80)


@app.command("plot-unimol-feature-distributions")
def plot_unimol_feature_distributions(
    train_features_path: Path = typer.Option(
        "data/processed/OpenCycloDB/unimol/train_features.csv",
        help="Path to the UniMol training features CSV file",
    ),
    output_path: Path = typer.Option(
        None,
        help="Path to save the output figure (if None, uses default in reports/figures/)",
    ),
    n_features_to_plot: int = typer.Option(
        100,
        help="Number of features to plot (useful for subset visualization). Set to -1 for all features.",
    ),
    plot_type: str = typer.Option(
        "violin",
        help="Type of plot: 'violin', 'box', or 'hist' for histogram grid",
    ),
) -> None:
    """
    Examine UniMol2 embedding feature distributions across all training data points.

    This function loads the UniMol2 embeddings from the training set and creates
    visualizations showing the distribution of each feature dimension. Useful for
    understanding the range, variance, and characteristics of the embedding space.

    Args:
        train_features_path: Path to the CSV file containing UniMol embeddings
        output_path: Where to save the generated plot
        n_features_to_plot: Number of features to visualize (or -1 for all)
        plot_type: Type of visualization ('violin', 'box', or 'hist')
    """
    logger.info(f"Loading UniMol training features from {train_features_path}")

    # Load the data
    df = pd.read_csv(train_features_path)
    n_samples, n_features = df.shape

    logger.info(f"Loaded {n_samples} samples with {n_features} features")

    # Determine how many features to plot
    if n_features_to_plot == -1:
        n_features_to_plot = n_features
    else:
        n_features_to_plot = min(n_features_to_plot, n_features)

    logger.info(f"Plotting distributions for {n_features_to_plot} features")

    # Calculate statistics for each feature
    feature_stats = {
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
        "median": df.median(),
    }

    logger.info("Feature statistics:")
    logger.info(
        f"  Mean range: [{feature_stats['mean'].min():.3f}, {feature_stats['mean'].max():.3f}]"
    )
    logger.info(
        f"  Std range: [{feature_stats['std'].min():.3f}, {feature_stats['std'].max():.3f}]"
    )
    logger.info(
        f"  Value range: [{feature_stats['min'].min():.3f}, {feature_stats['max'].max():.3f}]"
    )

    # Create the visualization
    if plot_type in ["violin", "box"]:
        # Create a single panel showing mean and std per feature
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))

        # Get color cycle
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # Select features to plot (evenly spaced if subset)
        if n_features_to_plot < n_features:
            feature_indices = np.linspace(
                0, n_features - 1, n_features_to_plot, dtype=int
            )
            data_to_plot = df.iloc[:, feature_indices]
            x_labels = [str(i) for i in feature_indices]
        else:
            data_to_plot = df
            x_labels = df.columns.tolist()

        # Plot mean and std per feature
        feature_means = data_to_plot.mean().values
        feature_stds = data_to_plot.std().values
        feature_positions = np.arange(data_to_plot.shape[1])

        # Filter to only show error bars for features with std above threshold
        std_threshold = 0.1
        high_std_mask = feature_stds >= std_threshold

        logger.info(
            f"Displaying error bars for {high_std_mask.sum()}/{len(feature_stds)} features (std >= {std_threshold})"
        )

        # Plot points without error bars first (for low std features)
        low_std_mask = ~high_std_mask
        if low_std_mask.any():
            ax.scatter(
                feature_positions[low_std_mask],
                feature_means[low_std_mask],
                s=10,
                color=color_cycle[0],
                alpha=0.7,
                zorder=2,
            )

        # Plot error bars only for high std features
        if high_std_mask.any():
            ax.errorbar(
                feature_positions[high_std_mask],
                feature_means[high_std_mask],
                yerr=feature_stds[high_std_mask],
                fmt="o",
                markersize=3,
                color=color_cycle[0],
                ecolor=color_cycle[0],
                elinewidth=1,
                capsize=3,
                alpha=0.7,
                label="Mean ± Std",
                zorder=3,
            )
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)

        ax.set_xlabel("Feature Index", fontsize=20)
        ax.set_ylabel("Value", fontsize=20)
        ax.legend(loc="upper right", fontsize=20)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_xlim(0, data_to_plot.shape[1])

        # Set x-axis ticks
        tick_spacing = max(1, len(x_labels) // 20)
        ax.set_xticks(np.arange(0, len(x_labels), tick_spacing))
        ax.set_xticklabels(
            [x_labels[i] for i in range(0, len(x_labels), tick_spacing)],
            rotation=45,
            fontsize=14,
        )
        ax.set_yticklabels(ax.get_yticks(), fontsize=14)

        plt.tight_layout()

    elif plot_type == "hist":
        # Create a grid of histograms with density curves
        n_cols = 10
        n_rows = int(np.ceil(n_features_to_plot / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
        axes = axes.flatten() if n_features_to_plot > 1 else [axes]

        # Select features to plot (evenly spaced if subset)
        if n_features_to_plot < n_features:
            feature_indices = np.linspace(
                0, n_features - 1, n_features_to_plot, dtype=int
            )
        else:
            feature_indices = list(range(n_features))

        for idx, feat_idx in enumerate(feature_indices):
            ax = axes[idx]
            feature_name = df.columns[feat_idx]
            feature_data = df.iloc[:, feat_idx].values

            # Plot histogram with density normalization
            n, bins, patches = ax.hist(
                feature_data,
                bins=30,
                alpha=0.6,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
                density=True,  # Normalize to show density
                label="Density",
            )

            # Add KDE curve for smooth density estimate
            from scipy.stats import gaussian_kde

            try:
                kde = gaussian_kde(feature_data)
                x_range = np.linspace(feature_data.min(), feature_data.max(), 100)
                ax.plot(
                    x_range, kde(x_range), "r-", linewidth=1.5, alpha=0.8, label="KDE"
                )
            except Exception as e:
                logger.debug(f"Could not compute KDE for feature {feature_name}: {e}")

            ax.set_title(f"Feat {feature_name}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.3)

            # Add mean line
            ax.axvline(
                feature_data.mean(),
                color="green",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

        # Hide unused subplots
        for idx in range(n_features_to_plot, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            f"UniMol2 Feature Distributions (n={n_samples} samples)", fontsize=16
        )
        plt.tight_layout()

    else:
        error_msg = f"Unknown plot_type: {plot_type}. Use 'violin', 'box', or 'hist'."
        raise ValueError(error_msg)

    # Save the figure
    if output_path is None:
        output_path = FIGURES_DIR / f"unimol_feature_distributions_{plot_type}.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved PNG to {output_path}")

    # Also save as PDF
    output_path_pdf = output_path.with_suffix(".pdf")
    plt.savefig(output_path_pdf, bbox_inches="tight")
    logger.info(f"Saved PDF to {output_path_pdf}")

    plt.close()


@app.command()
def main():
    typer.echo("Enter plot command")


if __name__ == "__main__":
    app()

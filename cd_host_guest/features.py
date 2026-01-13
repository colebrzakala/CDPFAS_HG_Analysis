from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


class DirectoryManager:
    """Class to manage directory setup and paths."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.dirs = {
            "smiles": self.base_dir / "OpenCycloDB" / "string",
            "fingerprints": self.base_dir / "OpenCycloDB" / "ECFP",
            "labels": self.base_dir / "OpenCycloDB" / "labels",
            "original_enriched": self.base_dir / "OpenCycloDB" / "original_enriched",
        }
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create directories if they do not exist."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_directories(self) -> dict[str, Path]:
        """Return the directory paths."""
        return self.dirs


class ECFPGenerator:
    """Class to generate Extended-Connectivity Fingerprints (ECFP)."""

    @staticmethod
    def generate(
        smiles: str, radius: int = 2, length: int = 2**10, use_chirality: bool = False
    ) -> pd.Series | None:
        """
        Generate ECFP from a SMILES string.

        Args:
            smiles (str): SMILES string of the compound.
            radius (int): Maximum radius of circular substructures.
            length (int): Fingerprint length.
            use_chirality (bool): Whether to include chirality information.

        Returns:
            Optional[pd.Series]: ECFP as a pandas Series or None if generation fails.
        """
        try:
            # Convert SMILES to RDKit molecule
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                logger.error(f"Invalid SMILES string: {smiles}")
                return None

            # Initialize MorganGenerator with specified parameters
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, includeChirality=use_chirality, fpSize=length
            )

            # Generate the fingerprint as a bit vector
            feature_list = generator.GetCountFingerprint(molecule)

            # Convert the bit vector to a pandas Series
            return pd.Series(list(feature_list))

        except Exception as e:
            logger.error(f"Error generating ECFP: {e}")
            return None


class StringFeatureGenerator:
    """Class to generate string representations (SMILES) from molecular data."""

    @staticmethod
    def generate_string_representations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract IsomericSMILES strings for guest and host molecules.

        Args:
            df (pd.DataFrame): DataFrame containing SMILES strings.

        Returns:
            pd.DataFrame: DataFrame with columns for guest and host IsomericSMILES.
        """
        string_df = df[
            [
                "IsomericSMILES",
                "IsomericSMILES_Host",
            ]
        ].copy()

        string_df = string_df.rename(columns={"IsomericSMILES": "IsomericSMILES_Guest"})

        return string_df


class ECFPFeatureGenerator:
    """Class to generate Extended-Connectivity Fingerprints (ECFP) from molecular data."""

    def __init__(
        self, radius: int = 2, length: int = 2**10, use_chirality: bool = False
    ):
        """
        Initialize ECFP generator with specific parameters.

        Args:
            radius (int): Maximum radius of circular substructures.
            length (int): Fingerprint length.
            use_chirality (bool): Whether to include chirality information.
        """
        self.radius = radius
        self.length = length
        self.use_chirality = use_chirality
        self.ecfp_generator = ECFPGenerator()

    def generate_ecfp_for_smiles(self, smiles: str) -> pd.Series | None:
        """
        Generate ECFP from a single SMILES string.

        Args:
            smiles (str): SMILES string of the compound.

        Returns:
            Optional[pd.Series]: ECFP as a pandas Series or None if generation fails.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            logger.error(f"Invalid SMILES: {smiles}")
            return None
        return self.ecfp_generator.generate(
            smiles,
            radius=self.radius,
            length=self.length,
            use_chirality=self.use_chirality,
        )

    def generate_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ECFP fingerprints from SMILES strings in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing SMILES strings.

        Returns:
            pd.DataFrame: DataFrame with columns for each fingerprint value.
        """
        guest_fingerprints = df["IsomericSMILES"].apply(self.generate_ecfp_for_smiles)
        host_fingerprints = df["IsomericSMILES_Host"].apply(
            self.generate_ecfp_for_smiles
        )

        # Create a new DataFrame to store the fingerprint values
        fingerprint_df = pd.DataFrame()

        # Extract guest fingerprints and create column labels
        guest_fingerprint_matrix = np.vstack(guest_fingerprints.dropna().values)
        guest_columns = [f"Guest_{i}" for i in range(guest_fingerprint_matrix.shape[1])]
        guest_fingerprint_df = pd.DataFrame(
            guest_fingerprint_matrix, columns=guest_columns
        )

        # Extract host fingerprints and create column labels
        host_fingerprint_matrix = np.vstack(host_fingerprints.dropna().values)
        host_columns = [f"Host_{i}" for i in range(host_fingerprint_matrix.shape[1])]
        host_fingerprint_df = pd.DataFrame(
            host_fingerprint_matrix, columns=host_columns
        )

        # Combine guest and host fingerprint DataFrames
        fingerprint_df = pd.concat([guest_fingerprint_df, host_fingerprint_df], axis=1)

        return fingerprint_df


class EnrichedFeatureGenerator:
    """Class to process enriched molecular data by extracting numeric features."""

    @staticmethod
    def process_enriched_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the enriched data by removing unnecessary columns.

        Args:
            df (pd.DataFrame): DataFrame containing the enriched data.

        Returns:
            pd.DataFrame: Processed DataFrame with only numeric features.
        """
        # Drop unnecessary columns
        non_numeric_columns = df.select_dtypes(include=["object"]).columns.tolist()

        numeric_df = df.copy()

        # Exclude non-numeric columns
        numeric_df = numeric_df.drop(columns=non_numeric_columns)

        # Exclude irrelevant columns
        irrelevant_columns = ["Unnamed: 0", "CID_Host", "CID_Guest", "K", "logK"]
        numeric_df = numeric_df.drop(columns=irrelevant_columns)

        # Exclude target columns
        target_columns = ["DeltaG", "Erreur"]
        numeric_df = numeric_df.drop(columns=target_columns)

        return numeric_df


class FeatureGenerator:
    """Orchestrator class to coordinate feature generation using modular generators."""

    def __init__(self):
        self.dir_manager = DirectoryManager(INTERIM_DATA_DIR)
        self.string_generator = StringFeatureGenerator()
        self.ecfp_generator = ECFPFeatureGenerator()
        self.enriched_generator = EnrichedFeatureGenerator()

    def generate_features(self, input_path: Path) -> None:
        """
        Generate and save molecular features using modular generators.

        Args:
            input_path (Path): Path to the input CSV file containing SMILES strings.
        """
        logger.info("Starting feature generation pipeline...")

        # Setup directories
        dirs = self.dir_manager.get_directories()

        try:
            # Load input data
            df = pd.read_csv(input_path)

            # Generate string representations (IsomericSMILES only)
            smiles_df = self.string_generator.generate_string_representations(df)

            # Generate fingerprints
            fingerprints_df = self.ecfp_generator.generate_fingerprints(df)

            # Create label vector if available
            labels = self.create_label_vector(df, "DeltaG")

            # Generate original enriched data
            original_enriched_df = self.enriched_generator.process_enriched_data(df)

            # Save all representations
            smiles_df.to_csv(dirs["smiles"] / "string_representations.csv", index=False)
            fingerprints_df.to_csv(
                dirs["fingerprints"] / "fingerprints.csv", index=False
            )

            original_enriched_df.to_csv(
                dirs["original_enriched"] / "original_enriched_data.csv", index=False
            )

            if labels is not None:
                labels.to_csv(dirs["labels"] / "labels.csv", index=False)

            logger.success("Feature generation pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Error in feature generation pipeline: {e}")
            raise

    @staticmethod
    def create_label_vector(df: pd.DataFrame, label_column: str) -> pd.DataFrame | None:
        """
        Create a label vector from a DataFrame column.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            label_column (str): Name of the column containing the labels.

        Returns:
            Optional[pd.DataFrame]: Label vector as a DataFrame or None if generation fails.
        """
        try:
            if label_column not in df.columns:
                logger.error(f"Label column '{label_column}' not found in DataFrame.")
                return None

            labels = df[[label_column]].copy()
            return labels
        except Exception as e:
            logger.error(f"Error creating label vector: {e}")
            return None


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "OpenCycloDB/Data/CDEnrichedData.csv",
) -> None:
    """Generate and save molecular features"""
    feature_generator = FeatureGenerator()
    feature_generator.generate_features(input_path)


@app.command("compare_PFAS_in_guests")
def compare_PFAS_in_guests() -> None:
    """
    Compare SMILES strings from list of all PFAS with guest compounds in string representations.

    Args:
        input_csv (Path): Path to the input CSV file containing SMILES strings.
    """
    try:
        # Load input data
        raw_df = pd.read_csv(RAW_DATA_DIR / "Chemical List epapfasinv-2025-01-08.csv")
        if "SMILES" not in raw_df.columns:
            logger.error("Column 'SMILES' not found in the input CSV file.")
            return

        # Extract SMILES column
        raw_smiles = raw_df["SMILES"]

        # Load string representations data
        string_representations_path = (
            INTERIM_DATA_DIR / "OpenCycloDB" / "string" / "string_representations.csv"
        )
        string_df = pd.read_csv(string_representations_path)
        if "IsomericSMILES_Guest" not in string_df.columns:
            logger.error(
                "Column 'IsomericSMILES_Guest' not found in the string representations CSV file."
            )
            return

        # Extract IsomericSMILES_Guest column
        guest_smiles = string_df["IsomericSMILES_Guest"]

        # Convert both SMILES columns to Canonical SMILES
        raw_canon_smiles = raw_smiles.apply(
            lambda x: Chem.CanonSmiles(x)
            if isinstance(x, str) and Chem.CanonSmiles(x)
            else None
        )
        guest_canon_smiles = guest_smiles.apply(
            lambda x: Chem.CanonSmiles(x)
            if isinstance(x, str) and Chem.CanonSmiles(x)
            else None
        )

        # Compare and find common compounds
        common_smiles = set(raw_canon_smiles).intersection(set(guest_canon_smiles))

        # Calculate and list the number of instances of common compounds
        common_smiles_instances = raw_canon_smiles.isin(common_smiles).sum()
        logger.info(
            f"Number of instances of common compounds: {common_smiles_instances}"
        )

        if common_smiles:
            logger.info(f"Common compounds found: {common_smiles}")
            logger.info(f"Number of common compounds: {len(common_smiles)}")

            # Calculate and list the number of unique SMILES in the common compounds
            unique_common_smiles = set(common_smiles)
            logger.info(
                f"Number of unique common compounds: {len(unique_common_smiles)}"
            )
        else:
            logger.info("No common compounds found.")

    except Exception as e:
        logger.error(f"Error comparing SMILES: {e}")
        raise


@app.command("generate_external_validation_all_embeddings")
def generate_external_validation_all_embeddings() -> None:
    """
    Generate ECFP and ECFP+ embeddings for external validation datasets.
    Saves the resulting features as CSVs in the same directory as the input files.
    Handles both standard and alternative column names for SMILES/temp/pH.
    """

    val_files = [
        EXTERNAL_DATA_DIR / "validation" / "cd_val" / "cd_val_canonical.csv",
        EXTERNAL_DATA_DIR / "validation" / "pfas_val" / "pfas_val_canonical.csv",
        EXTERNAL_DATA_DIR
        / "validation"
        / "base_cdpfas_val"
        / "base_cdpfas_val_canonical.csv",
    ]

    col_map_options = {
        "guest": ["IsomericSMILES", "Guest_SMILES"],
        "host": ["IsomericSMILES_Host", "Host_SMILES"],
        "temp": ["T"],
        "pH": ["pH"],
    }

    def find_column(df: pd.DataFrame, options: list[str]) -> str | None:
        for col in options:
            if col in df.columns:
                return col
        return None

    # Initialize the ECFPFeatureGenerator
    ecfp_feature_gen = ECFPFeatureGenerator()

    for val_path in val_files:
        logger.info(f"Processing validation file: {val_path}")
        if not val_path.exists():
            logger.error(f"Validation file not found: {val_path}")
            continue
        try:
            df = pd.read_csv(val_path)
        except Exception as e:
            logger.error(f"Failed to read {val_path}: {e}")
            continue

        guest_col = find_column(df, col_map_options["guest"])
        host_col = find_column(df, col_map_options["host"])
        temp_col = find_column(df, col_map_options["temp"])
        pH_col = find_column(df, col_map_options["pH"])

        if guest_col is None or host_col is None:
            logger.error(
                f"Required SMILES columns missing in {val_path}. Skipping file."
            )
            continue

        # Prepare output directories for each embedding type
        base_dir = val_path.parent
        ecfp_dir = base_dir / "ecfp"
        ecfp_plus_dir = base_dir / "ecfp_plus"
        ecfp_dir.mkdir(exist_ok=True)
        ecfp_plus_dir.mkdir(exist_ok=True)

        # --- ECFP/ECFP+ Generation using ECFPFeatureGenerator ---
        guest_ecfp = df[guest_col].apply(ecfp_feature_gen.generate_ecfp_for_smiles)
        host_ecfp = df[host_col].apply(ecfp_feature_gen.generate_ecfp_for_smiles)
        # Ensure valid_idx is always a 1D boolean Series
        if isinstance(guest_ecfp, pd.DataFrame):
            guest_valid = guest_ecfp.notnull().all(axis=1)
        else:
            guest_valid = guest_ecfp.notnull()
        if isinstance(host_ecfp, pd.DataFrame):
            host_valid = host_ecfp.notnull().all(axis=1)
        else:
            host_valid = host_ecfp.notnull()
        valid_idx = guest_valid & host_valid

        guest_ecfp = guest_ecfp[valid_idx].reset_index(drop=True)
        host_ecfp = host_ecfp[valid_idx].reset_index(drop=True)
        guest_matrix = np.vstack(guest_ecfp.values)
        host_matrix = np.vstack(host_ecfp.values)
        guest_cols = [f"Guest_{i}" for i in range(guest_matrix.shape[1])]
        host_cols = [f"Host_{i}" for i in range(host_matrix.shape[1])]
        ecfp_df = pd.DataFrame(guest_matrix, columns=guest_cols)
        ecfp_df = pd.concat(
            [ecfp_df, pd.DataFrame(host_matrix, columns=host_cols)], axis=1
        )
        ecfp_out = ecfp_dir / (val_path.stem + "_ecfp.csv")
        ecfp_df.to_csv(ecfp_out, index=False)
        logger.info(f"ECFP features saved to {ecfp_out}")
        # ECFP+
        if temp_col is not None and pH_col is not None:
            logger.debug(f"Detected temp_col: {temp_col}, pH_col: {pH_col}")
            # Always use the original columns from the input DataFrame, not from df_valid
            temp_series = df[temp_col]
            pH_series = df[pH_col]
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(temp_series):
                logger.warning(
                    f"Column '{temp_col}' is not numeric in {val_path}, attempting conversion."
                )
                temp_series = pd.to_numeric(temp_series, errors="coerce")
            if not pd.api.types.is_numeric_dtype(pH_series):
                logger.warning(
                    f"Column '{pH_col}' is not numeric in {val_path}, attempting conversion."
                )
                pH_series = pd.to_numeric(pH_series, errors="coerce")
            # valid_idx is a 1D boolean Series, use .loc for robust indexing
            temp_valid = temp_series.loc[valid_idx].reset_index(drop=True)
            pH_valid = pH_series.loc[valid_idx].reset_index(drop=True)
            # Now filter for rows where both temp and pH are not null
            valid_env = temp_valid.notnull() & pH_valid.notnull()
            logger.debug(
                f"Rows with valid temp/pH: {valid_env.sum()} / {len(temp_valid)}"
            )
            if not valid_env.all():
                logger.warning(
                    f"Dropping {len(temp_valid) - valid_env.sum()} rows with missing temp/pH in {val_path}"
                )
            ecfp_plus_df = ecfp_df.loc[valid_env].reset_index(drop=True)
            temp_ph_df = pd.DataFrame(
                {
                    temp_col: temp_valid[valid_env].reset_index(drop=True),
                    pH_col: pH_valid[valid_env].reset_index(drop=True),
                }
            )
            logger.debug(
                f"ecfp_plus_df shape: {ecfp_plus_df.shape}, temp_ph_df shape: {temp_ph_df.shape}"
            )
            ecfp_plus_df = pd.concat([ecfp_plus_df, temp_ph_df], axis=1)
            logger.debug(
                f"Final ecfp_plus_df shape (after concat): {ecfp_plus_df.shape}"
            )
            # Reorder columns: pH, T, Guest_*, Host_*
            guest_cols_order = [
                col for col in ecfp_plus_df.columns if col.startswith("Guest_")
            ]
            host_cols_order = [
                col for col in ecfp_plus_df.columns if col.startswith("Host_")
            ]
            ordered_cols = [*([pH_col, temp_col]), *guest_cols_order, *host_cols_order]
            ordered_cols += [
                col for col in ecfp_plus_df.columns if col not in ordered_cols
            ]
            ecfp_plus_df = ecfp_plus_df[ordered_cols]
            ecfp_plus_out = ecfp_plus_dir / (val_path.stem + "_ecfp_plus.csv")
            if not ecfp_plus_df.empty:
                ecfp_plus_df.to_csv(ecfp_plus_out, index=False)
                logger.info(f"ECFP+ features saved to {ecfp_plus_out}")
            else:
                logger.warning(f"No valid rows for ECFP+ in {val_path}, skipping save.")
        else:
            logger.warning(
                f"Skipping ECFP+ embedding for {val_path} (missing temp or pH column)"
            )

    logger.success("External validation ECFP and ECFP+ embedding generation complete.")


@app.command("canonicalize_smiles")
def canonicalize_smiles(
    input_csv: Path = typer.Argument(..., help="Path to input CSV file"),
    output_csv: Path = typer.Option(
        None, help="Path to save output CSV file (optional)"
    ),
) -> None:
    """
    Replace SMILES columns in a CSV file with canonical isomeric SMILES using RDKit.

    Args:
        input_csv (Path): Path to input CSV file.
        output_csv (Path, optional): Path to save output CSV file. If not provided, '_canonical' is appended to input filename.
    """
    # Predefined list of possible SMILES column names
    smiles_columns = ["Guest_SMILES", "Host_SMILES"]

    # Load CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        logger.error(f"Failed to read input CSV: {e}")
        return

    # Find which columns to canonicalize
    found_cols = [col for col in smiles_columns if col in df.columns]
    if not found_cols:
        logger.error("No SMILES columns found in input CSV.")
        return

    # Function to canonicalize SMILES
    def to_canonical_smiles(smiles: str) -> str | None:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Canonical SMILES (isomeric)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception as e:
            logger.error(f"Error canonicalizing SMILES '{smiles}': {e}")
            return None

    # Replace each found column with its canonical isomeric SMILES
    for col in found_cols:
        logger.info(f"Canonicalizing column: {col}")
        df[col] = df[col].apply(to_canonical_smiles)

    # Determine output path
    if output_csv is None:
        output_csv = input_csv.parent / f"{input_csv.stem}_canonical.csv"

    # Save the modified DataFrame
    try:
        df.to_csv(output_csv, index=False)
        logger.success(f"Canonicalized CSV saved to {output_csv}")
    except Exception as e:
        logger.error(f"Failed to save canonicalized CSV: {e}")


if __name__ == "__main__":
    app()

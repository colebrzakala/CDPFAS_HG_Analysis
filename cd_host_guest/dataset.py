from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
import typer
from loguru import logger
from tqdm import tqdm

# Import necessary configurations and utilities
from cd_host_guest.config import RAW_DATA_DIR

# Initialize Typer app
app = typer.Typer()

# Define the default argument and option outside the function
dataset_argument = typer.Argument(..., help="Dataset to download")
force_option = typer.Option(
    False, "--force", "-f", help="Force download even if files exist"
)


@dataclass
class DatasetConfig:
    """Configuration for a dataset source"""

    name: str
    urls: dict[str, str]  # filename -> url mapping
    description: str
    requires_extraction: bool = False
    extract_dir: Path | None = None


# Define dataset configurations
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "OpenCycloDB": DatasetConfig(
        name="OpenCycloDB",
        urls={"Data.zip": "https://zenodo.org/records/7575539/files/Data.zip"},
        description="Host-guest binding affinity data for cyclodextrins",
        requires_extraction=True,
        extract_dir=RAW_DATA_DIR / "OpenCycloDB",
    )
    # Add more dataset configurations as needed
}


@app.command()
def download(
    dataset: str = dataset_argument,
    force: bool = force_option,
) -> None:
    """
    Download and prepare a specific dataset.

    Args:
        dataset (str): Dataset to download
        force (bool): Force download even if files exist
    """
    config = DATASET_CONFIGS[dataset]
    logger.info(f"Downloading {config.name} dataset...")

    output_dir = RAW_DATA_DIR / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in config.urls.items():
        output_path = output_dir / filename

        if output_path.exists() and not force:
            logger.info(f"File {filename} already exists. Skipping download.")
            continue

        try:
            download_file(url, output_path)

            if config.requires_extraction and filename.endswith(".zip"):
                extract_dir = config.extract_dir or output_dir
                extract_zip(output_path, extract_dir)

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e!s}")
            raise

    logger.success(f"Successfully downloaded {config.name} dataset.")


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL and save it to the specified path.

    Args:
        url (str): URL of the file to download
        output_path (Path): Path where the file should be saved
        chunk_size (int): Size of chunks for streaming download
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            open(output_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=f"Downloading {output_path.name}",
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = f.write(chunk)
                pbar.update(size)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e!s}")
        raise


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip file to the specified directory.

    Args:
        zip_path (Path): Path to the zip file
        extract_dir (Path): Directory to extract files to
    """
    logger.info(f"Extracting {zip_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    logger.success(f"Successfully extracted {zip_path.name}")


if __name__ == "__main__":
    app()

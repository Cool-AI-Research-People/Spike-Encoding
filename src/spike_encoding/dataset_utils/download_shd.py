import urllib.request
import zipfile
from pathlib import Path


def load_shd():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Dataset URLs
    urls = {
        "train": "https://zenkelab.org/datasets/shd_train.h5.zip",
        "test": "https://zenkelab.org/datasets/shd_test.h5.zip",
    }

    extracted_files = {}

    for dataset_type, url in urls.items():
        zip_path = data_dir / f"shd_{dataset_type}.h5.zip"
        h5_path = data_dir / f"shd_{dataset_type}.h5"

        # Skip if already extracted
        if h5_path.exists():
            print(f"SHD {dataset_type} dataset already exists: {h5_path}")
            extracted_files[dataset_type] = h5_path
            continue

        print(f"Downloading SHD {dataset_type} dataset...")

        # Download the zip file
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded: {zip_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

        # Extract the zip file
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted: {zip_path}")
            extracted_files[dataset_type] = h5_path
        except Exception as e:
            raise RuntimeError(f"Failed to extract {zip_path}: {e}")

        # Delete the zip file
        try:
            zip_path.unlink()
            print(f"Cleaned up: {zip_path}")
        except Exception as e:
            print(f"Warning: Could not delete {zip_path}: {e}")

    print(f"SHD dataset ready in: {data_dir}")
    return extracted_files["train"], extracted_files["test"]
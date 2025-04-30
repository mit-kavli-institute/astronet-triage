from __future__ import annotations
from typing import Optional
import yaml
from pathlib import Path

"""
HOW TO USE THIS API

The config_parser is meant to be used to create a DatasetConfig which is an
input to a DataManager.

By filling out config.yaml, you can maintain a single dataset object to use in
every notebook rather than filling out individually all of the sheets you need.
Sheets can also be loaded dynamically live from Google to keep the latest
source of truth.

"""

import yaml
from pathlib import Path
from pydantic import BaseModel


class Constants:
    """String constants used throughout the project."""

    CONFIG_FILE: str = "config.yaml"

    # Dataset keys
    DATASET: str = "dataset"
    RAW_DATA_SOURCE_TYPE: str = "raw_data_source_type"
    RAW_DATA_DIR: str = "raw_data_dir"
    IMAGES_DIR: str = "images_dir"
    REPORTS_DIR: str = "reports_dir"

    # Source Types
    LOCAL: str = "local"
    REMOTE: str = "remote"

    # Sheets
    LABELS_SHEET: str = "labels_sheet"
    PROPERTIES_SHEET: str = "properties_sheet"
    DATASET_SPLIT_SHEET: str = "dataset_split_sheet"


class DatasetConfig(BaseModel):
    """Pydantic model to store dataset configuration."""
    properties_sheet: str
    raw_data_source_type: Optional[str] = None
    raw_data_dir: Optional[Path]= None
    images_dir: Optional[Path] = None
    reports_dir: Optional[Path] = None
    labels_sheet: Optional[str] = None
    dataset_split_sheet: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str = Constants.CONFIG_FILE) -> DatasetConfig:
        """Reads YAML configuration and returns a DatasetConfig instance."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        if Constants.DATASET not in config:
            raise ValueError(f"Invalid config format: Missing '{Constants.DATASET}' section.")

        dataset = config[Constants.DATASET]
        return cls(**dataset)

    def __str__(self) -> str:
        return "DatasetConfig(\n\t" \
            f"raw_data_source_type={self.raw_data_source_type}\n\t" \
            f"raw_data_dir={self.raw_data_dir}\n\t" \
            f"reports_dir={self.reports_dir}\n\t" \
            f"images_dir={self.images_dir}\n\t" \
            f"labels_sheet={self.labels_sheet}\n\t" \
            f"properties_sheet={self.properties_sheet}\n\t" \
            f"dataset_split_sheet={self.dataset_split_sheet}\n)"


if __name__ == "__main__":
    dataset_config = DatasetConfig.from_yaml("config.yaml")
    print(f'Loaded dataset config: {dataset_config}')
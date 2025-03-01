import yaml
from dataclasses import dataclass
from pathlib import Path
import os

"""
HOW TO USE THIS API

The config_parser is meant to be used to create a DatasetConfig which is an
input to a DataManager.

By filling out config.yaml, you can maintain a single dataset object to use in
every notebook rather than filling out individually all of the sheets you need.
Sheets can also be loaded dynamically live from Google to keep the latest
source of truth.

"""

class Constants:
    """String constants used throughout the project."""

    CONFIG_FILE: str = "config.yaml"

    # Dataset keys
    DATASET: str = "dataset"
    RAW_DATA_SOURCE_TYPE: str = "raw_data_source_type"
    RAW_DATA_DIR: str = "raw_data_dir"

    # Source Types
    LOCAL: str = "local"
    REMOTE: str = "remote"

    # Sheets
    LABELS_SHEET: str = "labels_sheet"
    PROPERTIES_SHEET: str = "properties_sheet"
    DATASET_SPLIT_SHEET: str = "dataset_split_sheet"

@dataclass
class DatasetConfig:
    """Dataclass to store dataset configuration."""
    raw_data_source_type: str
    raw_data_dir: Path
    labels_sheet: str
    properties_sheet: str
    dataset_split_sheet: str

    @staticmethod
    def from_yaml(config_path: str = Constants.CONFIG_FILE) -> "DatasetConfig":
        """Reads YAML configuration and returns a DatasetConfig instance."""

        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        if Constants.DATASET not in config:
            raise ValueError(f"Invalid config format: Missing '{Constants.DATASET}' section.")

        dataset = config[Constants.DATASET]

        return DatasetConfig(
            raw_data_source_type=dataset.get(Constants.RAW_DATA_SOURCE_TYPE),
            raw_data_dir=Path(dataset.get(Constants.RAW_DATA_DIR)),
            labels_sheet=dataset.get(Constants.LABELS_SHEET),
            properties_sheet=dataset.get(Constants.PROPERTIES_SHEET),
            dataset_split_sheet=dataset.get(Constants.DATASET_SPLIT_SHEET),
        )

    def __str__(self) -> str:
        return "DatasetConfig(\n\t" \
            f"raw_data_source_type={self.raw_data_source_type}\n\t" \
            f"raw_data_dir={self.raw_data_dir}\n\t" \
            f"labels_sheet={self.labels_sheet}\n\t" \
            f"properties_sheet={self.properties_sheet}\n\t" \
            f"dataset_split_sheet={self.dataset_split_sheet}\n)"

if __name__ == "__main__":
    dataset_config = DatasetConfig.from_yaml("config.yaml")
    print(f'Loaded dataset config: {dataset_config}')

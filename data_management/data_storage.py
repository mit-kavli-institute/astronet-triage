import re
from pathlib import Path
from collections import defaultdict
from typing import Optional


class DataStorage:
    def __init__(self, data_dir: Path, images_dir: Path, reports_dir: Path) -> None:
        self.tic_id_to_path: dict[int, str] = {}
        self.tic_id_to_images_path: dict[int, str] = {}
        self.tic_id_to_reports_path: dict[str, list[str]] = defaultdict(list)
        if not data_dir:
            return
        if not data_dir.is_absolute(): # enable both local and full paths
            data_dir = data_dir.resolve()
        if not images_dir.is_absolute(): # enable both local and full paths
            images_dir = images_dir.resolve()
        if not reports_dir.is_absolute(): # enable both local and full paths
            reports_dir = reports_dir.resolve()

        # load raw data
        for file in Path(data_dir).glob("*.fits"):
            tic_id = self.extract_tic_id(file.name)
            if tic_id is None:
                print(f"Skipping {file.name}: TIC ID could not be extracted.")
                continue

            full_path = str(file.resolve())  # Get absolute OS path
            self.tic_id_to_path[tic_id] = full_path

        # load image
        for folder in Path(images_dir).iterdir():
            if folder.is_dir():
                tic_id = int(folder.name)
                full_path = str(folder.resolve())
                self.tic_id_to_images_path[tic_id] = full_path

        # load reports
        self.tic_id_to_reports_path: dict[str, list[str]] = defaultdict(list)
        # there can be N reports in the form <tic_id>.page<n>.png
        for file in Path(reports_dir).glob("*.png"):
            tic_id = int(str(file.name).split(".")[0])
            if tic_id is None:
                print(f"Skipping {file.name}: TIC ID could not be extracted.")
                continue

            full_path = str(file.resolve())
            self.tic_id_to_reports_path[tic_id].append(full_path)

    def extract_tic_id(self, filename: str) -> int:
        """Extract TIC ID from the filename using regex."""
        match = re.search(r'-(\d{9,})_', filename)
        if match:
            return int(match.group(1))
        return None

    def get_path(self, tic_id: int) -> Optional[str]:
        return self.tic_id_to_path.get(tic_id)

    def get_images_path(self, tic_id: int) -> Optional[Path]:
        return self.tic_id_to_images_path.get(tic_id)
    
    def get_reports_path(self, tic_id: int) -> Optional[Path]:
        return self.tic_id_to_reports_path.get(tic_id, [])
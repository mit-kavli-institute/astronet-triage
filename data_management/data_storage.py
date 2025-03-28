import re
from pathlib import Path


class DataStorage:
    def __init__(self, data_dir: Path) -> None:
        self.tic_id_to_path: dict[int, str] = {}
        if not data_dir.is_absolute(): # enable both local and full paths
            data_dir = data_dir.resolve()
        for file in Path(data_dir).glob("*.fits"):
            tic_id = self.extract_tic_id(file.name)
            if tic_id is None:
                print(f"Skipping {file.name}: TIC ID could not be extracted.")
                continue

            full_path = str(file.resolve())  # Get absolute OS path
            self.tic_id_to_path[tic_id] = full_path


    def extract_tic_id(self, filename: str) -> int:
        """Extract TIC ID from the filename using regex."""
        match = re.search(r'-(\d{9,})_', filename)
        if match:
            return int(match.group(1))
        return None

    def get_path(self, tic_id: int) -> str | None:
        return self.tic_id_to_path.get(tic_id)

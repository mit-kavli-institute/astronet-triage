"""
Helper functions for finding files and directories related to Astronet.
"""

from pathlib import Path
from typing import Optional


def find_checkpoint_paths(base_dir: Path, nruns: Optional[int] = None) -> list[Path]:
    """
    Find directories containing models, assuming structure of training checkpoints.
    """
    if nruns is None:
        nruns = len(list(base_dir.iterdir()))
    return [next((base_dir / str(i)).iterdir()) for i in range(1, nruns + 1)]

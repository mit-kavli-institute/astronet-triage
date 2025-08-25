from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class Split(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    UNALLOCATED = 'unallocated'

    @classmethod
    def from_str(cls, s: str) -> Split:
        s = str(s)
        if s.lower() == cls.TRAIN.value:
            return cls.TRAIN
        elif s.lower() == cls.VALIDATION.value or s.lower() == 'val':
            return cls.VALIDATION
        elif s.lower() == cls.TEST:
            return cls.TEST
        else:
            return cls.UNALLOCATED

@dataclass
class AstroData:
    astro_id: int
    tic_id: int
    fits_path: str
    report_paths: list[str]
    images_path: Optional[str]
    properties: dict[str, object]
    split: Split
    label: Optional[str] = None
    label_simplified: Optional[str] = None

    @property
    def id(self) -> int | None:
        return self.properties.get('id')

    @property
    def version_id(self) -> int | None:
        return self.properties.get('version_id')

    @property
    def ra(self) -> Optional[float]:
        return self.properties.get('ra')

    @property
    def dec(self) -> Optional[float]:
        return self.properties.get('dec')

    @property
    def tmag(self) -> Optional[float]:
        return self.properties.get('tmag')

    @property
    def epoc(self) -> Optional[float]:
        return self.properties.get('epoc')

    @property
    def period(self) -> Optional[float]:
        return self.properties.get('period')

    @property
    def duration(self) -> Optional[float]:
        return self.properties.get('duration')

    @property
    def transit_depth(self) -> Optional[float]:
        return self.properties.get('transit_depth')

    @property
    def sectors(self) -> Optional[object]:
        return self.properties.get('sectors')

    @property
    def star_rad(self) -> Optional[float]:
        return self.properties.get('star_rad')

    @property
    def star_mass(self) -> Optional[float]:
        return self.properties.get('star_mass')

    @property
    def teff(self) -> Optional[float]:
        return self.properties.get('teff')

    @property
    def logg(self) -> Optional[float]:
        return self.properties.get('logg')

    @property
    def sn(self) -> Optional[float]:
        return self.properties.get('sn')

    @property
    def qingress(self) -> Optional[float]:
        return self.properties.get('qingress')

    @property
    def star_rad_est(self) -> Optional[float]:
        return self.properties.get('star_rad_est')

    @property
    def filename(self) -> Optional[str]:
        return self.properties.get('filename')

    @property
    def comment(self) -> Optional[str]:
        return self.properties.get('comment')

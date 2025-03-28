from __future__ import annotations
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
    images_path: str | None
    properties: dict[str, object]
    split: Split
    label: str | None = None
    label_simplified: str | None = None

    @property
    def id(self) -> int | None:
        return self.properties.get('id')

    @property
    def version_id(self) -> int | None:
        return self.properties.get('version_id')

    @property
    def ra(self) -> float | None:
        return self.properties.get('ra')

    @property
    def dec(self) -> float | None:
        return self.properties.get('dec')

    @property
    def tmag(self) -> float | None:
        return self.properties.get('tmag')

    @property
    def epoc(self) -> float | None:
        return self.properties.get('epoc')

    @property
    def period(self) -> float | None:
        return self.properties.get('period')

    @property
    def duration(self) -> float | None:
        return self.properties.get('duration')

    @property
    def transit_depth(self) -> float | None:
        return self.properties.get('transit_depth')

    @property
    def sectors(self) -> object | None:
        return self.properties.get('sectors')

    @property
    def star_rad(self) -> float | None:
        return self.properties.get('star_rad')

    @property
    def star_mass(self) -> float | None:
        return self.properties.get('star_mass')

    @property
    def teff(self) -> float | None:
        return self.properties.get('teff')

    @property
    def logg(self) -> float | None:
        return self.properties.get('logg')

    @property
    def sn(self) -> float | None:
        return self.properties.get('sn')

    @property
    def qingress(self) -> float | None:
        return self.properties.get('qingress')

    @property
    def star_rad_est(self) -> float | None:
        return self.properties.get('star_rad_est')

    @property
    def filename(self) -> str | None:
        return self.properties.get('filename')

    @property
    def comment(self) -> str | None:
        return self.properties.get('comment')

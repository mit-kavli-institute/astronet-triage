from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from light_curve_util import tess_io
from light_curve_util import keplersplinev2
from light_curve_util import median_filter2
from light_curve_util import util
import numpy as np

@dataclass
class AstroData:
    astro_id: int
    tic_id: int
    fits_path: str
    report_path: Optional[str]
    properties: Dict[str, Any]
    label: Optional[str] = None

    def read_light_curve(self, flux_key: str, min_t: int, max_t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads a TESS light curve from a FITS file and filters the data based on the provided time range.
        
        Args:
            flux_key (str): The key to extract the desired flux type from the FITS file.
            min_t (int): The minimum time value for filtering.
            max_t (int): The maximum time value for filtering.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered time and flux arrays.
        
        Raises:
            AssertionError: If no data points remain after filtering.
        """
        time, flux = tess_io.read_tess_light_curve(self.fits_path, flux_key)
        mask = (time >= min_t) & (time <= max_t)

        filtered_time = time[mask]
        filtered_flux = flux[mask]

        assert len(filtered_time), "No data points remain after filtering."
        return filtered_time, filtered_flux

    def filter_outliers(self, time: np.ndarray, flux: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Removes NaN values from the flux data while keeping time and mask aligned.
        
        Args:
            time (np.ndarray): The time array of the light curve.
            flux (np.ndarray): The flux measurements corresponding to the time array.
            mask (np.ndarray): A boolean mask used to filter data points.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of time, flux, and mask after removing NaN values.
        """
        valid = ~np.isnan(flux)
        return time[valid], flux[valid], mask[valid]

    def detrend_and_filter(
        self, time: np.ndarray, flux: np.ndarray, fixed_bkspace: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detrends the light curve using the Kepler Spline algorithm and filters outliers.
        
        Args:
            time (np.ndarray): The time array of the light curve.
            flux (np.ndarray): The flux measurements corresponding to the time array.
            fixed_bkspace (float): Fixed break-space parameter for the spline fit.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and detrended flux arrays after filtering.
        """
        mask = self.get_spline_mask(time)
        spline_flux, metadata = keplersplinev2.choosekeplersplinev2(
            time, flux, input_mask=mask, fixed_bkspace=fixed_bkspace, return_metadata=True
        )

        detrended_flux = flux / spline_flux
        return self.filter_outliers(time, detrended_flux, mask)

    def phase_fold_and_sort_light_curve(
        self, time: np.ndarray, flux: np.ndarray, mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase folds the light curve based on the known period and epoch, then sorts it by ascending time.
        
        Args:
            time (np.ndarray): The time array of the light curve.
            flux (np.ndarray): The flux measurements corresponding to the time array.
            mask (np.ndarray): A boolean mask used to filter data points.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - Folded time array sorted in ascending order.
                - Corresponding sorted flux array.
                - Fold number array sorted accordingly.
                - Sorted mask array.
        """
        if time.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        folded_time, fold_num = util.phase_fold_time(time, self.period, self.epoc)
        sorted_indices = np.argsort(folded_time)

        return (
            folded_time[sorted_indices],
            flux[sorted_indices],
            fold_num[sorted_indices],
            mask[sorted_indices],
        )

    @property
    def id(self) -> Optional[int]:
        return self.properties.get('id')

    @property
    def version_id(self) -> Optional[int]:
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
    def sectors(self) -> Optional[Any]:
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

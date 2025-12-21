from io import BytesIO
from PIL import Image
import requests
from pathlib import Path
import os

PAGE_NUMBER_TO_TYPE: dict = {
    0: "Summary",
    2: "BLS Spectrum",
    3: "Depth-aperture Correlation",
    5: "Difference Images",
    6: "MCMC Fit",
    7: "Full Raw LC + Folded Detrended LC",
    8: "Matches to Known Signals",
    9: "Full Detrended LC",
    20: "TFRecord Global View",
    21: "TFRecord Local View",
    22: "TFRecord Local View Odd",
    23: "TFRecord Local View Even",
    24: "TFRecord Secondary View",
    25: "TFRecord Secondary Phase",
    26: "TFRecord Segment View",
    27: "TFRecord Segment Local View",
}

PAGE_TYPE_TO_TFRECORD_KEY: dict = {
    'TFRecord Global View': 'global_view',
    'TFRecord Local View': 'local_view',
    'TFRecord Local View Odd': 'local_view_odd',
    'TFRecord Local View Even': 'local_view_even',
    'TFRecord Secondary View': 'secondary_view',
    'TFRecord Secondary Phase': 'secondary_phase',
    'TFRecord Segment View': 'sample_segments_view',
    'TFRecord Segment Local View': 'sample_segments_local_view',


}

LOCAL_PAGE_TO_FILENAME = {
    #0: "_summary.png",
    20: "_global.png",
    21: "_local.png",
    22: "_props.png",
    23: "_secondary.png",
    24: "_segments.png"
}

ALL_PAGE_TYPES = [v for _, v in sorted(PAGE_NUMBER_TO_TYPE.items())]

class LightCurveServer:
    def __init__(self, server_url: str = f"http://localhost:5001"):
        self.server_url = server_url
        self.reports_dir = "/pdo/astronet-data/data/reports/"

    def get_report_pages(self, tic_id: int, planet_number: int) -> list:
        """
        Returns a list of available page numbers (including local TFRecord reports).
        """
        # Start with remote pages (API)
        remote_pages = []
        url = f"{self.server_url}/api/report-pages/{tic_id}"
        params = {"planet_number": planet_number}
        try:
            response = requests.get(url, params=params)
            if response.ok:
                data = response.json()
                if isinstance(data, list):
                    remote_pages = data
                else:
                    print(f"Unexpected response format: {data}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Server error: {e}")

        # Now check for local TFRecord files
        local_pages = []
        for page_number, suffix in LOCAL_PAGE_TO_FILENAME.items():
            path = os.path.join(self.reports_dir, f"{str(tic_id)[:3]}", f"{tic_id}{planet_number:02d}{suffix}")
            if os.path.exists(path):
                local_pages.append(page_number)

        return sorted(set(remote_pages + local_pages))
        
    def get_page_image(self, tic_id: int, page_number: int, planet_number: int = 1) -> Image.Image:
        """
        Returns the image for a given TIC ID and page number.
        Uses local files for TFRecord pages 20–24.
        """
        if page_number in LOCAL_PAGE_TO_FILENAME:
            # --- Handle local TFRecord image ---
            suffix = LOCAL_PAGE_TO_FILENAME[page_number]
            report_path = Path(self.reports_dir) / f"{str(tic_id)[:3]}" / f"{tic_id}{planet_number:02d}{suffix}"
            if report_path.exists():
                return Image.open(report_path)
            else:
                print(f"Report file not found: {report_path}")
                return None
        
        # --- Fallback to server API for standard pages ---
        url = f"{self.server_url}/api/report/{tic_id}"
        params = {
            "page": page_number,
            "planet_number": planet_number
        }
        response = requests.get(url, params=params)
        if response.ok:
            image_bytes = BytesIO(response.content)
            return Image.open(image_bytes)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return ""
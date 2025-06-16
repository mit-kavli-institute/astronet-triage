from io import BytesIO
from PIL import Image
import requests

PAGE_NUMBER_TO_TYPE: dict = {
    0: "Summary",
    2: "BLS Spectrum",
    3: "Depth-aperture Correlation",
    5: "Difference Images",
    6: "MCMC Fit",
    7: "Full Raw LC + Folded Detrended LC",
    8: "Matches to Known Signals",
    9: "Full Detrended LC",
}
ALL_PAGE_TYPES = ["Summary", "BLS Spectrum", "Depth-aperture Correlation", "Difference Images", "Full Detrended LC", "Full Raw LC + Folded Detrended LC", "MCMC Fit", "Matches to Known Signals"]

class LightCurveServer:
    def __init__(self, server_url: str = f"http://localhost:5001"):
        self.server_url = server_url

    def get_report_pages(self, tic_id: int, planet_number: int) -> list:
        """
        Returns a list of report paths for a given TIC ID.
        """
        url = f"{self.server_url}/api/report-pages/{tic_id}"
        params = {
            "planet_number": planet_number
        }
        response = requests.get(url, params=params)
        if response.ok:
            data = response.json()
            if isinstance(data, list):
                return data
            else:
                print(f"Unexpected response format: {data}")
                return []
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return []
        
    def get_page_image(self, tic_id: int, page_number: int, planet_number: int=1) -> Image:
        """
        Returns the URL of the image for a given TIC ID and page number.
        """
        url = f"{self.server_url}/api/report/{tic_id}"
        params = {
            "page": page_number,
            "planet_number": planet_number
        }
        response = requests.get(url, params=params)
        if response.ok:
            # convert to image
            image_bytes = BytesIO(response.content)
            img = Image.open(image_bytes) 
            return img
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return ""
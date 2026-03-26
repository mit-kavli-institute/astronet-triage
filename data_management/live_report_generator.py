import requests
import os
from pathlib import Path


class LiveReportGenerator:
    def __init__(self, live_generation_url: str = f'http://127.0.0.1:8123'):
        self.live_generation_url = live_generation_url

    def generate_summary(self, tic_id: int, cam: int, ccd: int, planetno: int, sector: int, page_num: int):

        # Attempt 0: before calling the API, check QLP
        cam = int(cam)
        ccd = int(ccd)
        if page_num > 0:
            page_folder_map = {1: 'PageOne', 2: 'PageTwo', 3: 'PageThree', 5: 'PageFive', 6: 'PageSix', 7: 'PageSeven'}
            qlp_summary_path = Path(f"/pdo/qlp-data/sector-{int(sector.split('_')[1])}/ffi/cam{cam}/ccd{ccd}/REPORTS/{page_folder_map[page_num]}/{tic_id}.Page{page_num}.png")
        else:
            qlp_summary_path = Path(f"/pdo/qlp-data/sector-{int(sector.split('_')[1])}/ffi/cam{cam}/ccd{ccd}/REPORTS/{tic_id}.png")
        print(f'Checking path {qlp_summary_path}...')
        if qlp_summary_path.is_file():
            return str(qlp_summary_path) 
        else:
            ...

        payload = {
            "tic_id": tic_id,
            "cam": cam,
            "ccd": ccd,
            "planetno": planetno,
            "sector": int(sector.split('_')[1]),
        }
        resp = requests.post(self.live_generation_url + '/summary', json=payload, timeout=600)
        print("STATUS:", resp.status_code)
        print("BODY:", resp.text)
        resp.raise_for_status()
        return resp.json()["img_path"]

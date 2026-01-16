import requests

class LiveReportGenerator:
    def __init__(self, live_generation_url: str = f'http://127.0.0.1:8123'):
        self.live_generation_url = live_generation_url

    def generate_summary(self, tic_id: int, cam: int, ccd: int, planetno: int, sector: int):
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

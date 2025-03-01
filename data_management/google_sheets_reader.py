import pandas as pd

class GoogleSheetsReader:
    @staticmethod
    def from_url(url: str, page_num: int = 0) -> pd.DataFrame:
        """
        Fetches data from a public Google Sheet and returns it as a Pandas DataFrame.

        :param url: The Google Sheets URL.
        :param page_num: The index of the sheet (default is 0, the first sheet).
        :return: Pandas DataFrame with the sheet data.
        """
        try:
            if "docs.google.com/spreadsheets/d/" not in url:
                raise ValueError("Invalid Google Sheets URL.")

            sheet_id = url.split("/d/")[1].split("/")[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={page_num}"

            df = pd.read_csv(csv_url)
            return df

        except Exception as e:
            print(f"Error fetching Google Sheet data: {e}")
            return pd.DataFrame()

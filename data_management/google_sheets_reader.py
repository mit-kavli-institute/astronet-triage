import os
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

import os
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

class GoogleSheetsReader:
    def __init__(self):
        """
        Initializes the Google Sheets Reader using an environment variable for authentication.
        """
        self.client = self.authenticate()

    def authenticate(self):
        """
        Authenticates the service account with Google Sheets API using the credentials from an environment variable.
        """
        try:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path:
                raise ValueError("Google service account credentials file not found in environment variables.")

            scopes = ["https://www.googleapis.com/auth/spreadsheets",
                      "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
            return gspread.authorize(creds)
        except Exception as e:
            print(f"Authentication Error: {e}")
            return None

    def from_url(self, url: str, sheet_name: str = None) -> pd.DataFrame:
        """
        Fetches data from a Google Sheet (public or private) and returns it as a Pandas DataFrame.

        :param url: The Google Sheets URL.
        :param sheet_name: The name of the sheet (default is the first sheet).
        :return: Pandas DataFrame with the sheet data.
        """
        try:
            if not self.client:
                raise ValueError("Google Sheets authentication failed.")

            if "docs.google.com/spreadsheets/d/" not in url:
                raise ValueError("Invalid Google Sheets URL.")

            # Extract the Google Sheet ID from the URL
            sheet_id = url.split("/d/")[1].split("/")[0]
            spreadsheet = self.client.open_by_key(sheet_id)

            # Select the specific sheet
            if sheet_name:
                worksheet = spreadsheet.worksheet(sheet_name)
            else:
                worksheet = spreadsheet.get_worksheet(0)  # Default to the first sheet

            # Get all data as records (list of dictionaries)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)

            return df

        except Exception as e:
            print(f"Error fetching Google Sheet data: {e} (did you share the service account? astronet-sheets-reader@astronet-452402.iam.gserviceaccount.com)")
            return pd.DataFrame()
